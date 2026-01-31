import os
import argparse
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# problem.py が同じディレクトリにあることを前提とします
try:
    from problem import ShiftProblemData
except ImportError:
    ShiftProblemData = None
    print("Warning: Could not import ShiftProblemData. Cost breakdown plots will be skipped.")

# ==========================================
# matplotlibの設定 (LaTeX風フォントを使用)
# ==========================================
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

# ==========================================
# データ読み込み・解析関数
# ==========================================
def build_path_from_params(base_prefix, params):
    """
    指定されたディレクトリ構造に従ってパスを生成する
    
    修正後ルール:
    すべての手法は {base_prefix}_pat_{patience} の下に配置される。
    例: results_exp1_pat_5 / std
        results_exp1_pat_5 / acc
    """
    p_val = params.get('patience')
    method = params.get('method')
    
    # 手法による分岐を削除し、統一されたディレクトリ構造に対応
    dir_name = f"{base_prefix}_pat_{p_val}"
    
    # 親ディレクトリ + 手法名ディレクトリ
    path = os.path.join(dir_name, method)
    return path

def load_metrics_from_json(base_prefix, config_path, start_week=None, end_week=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_list = json.load(f)
        
    data_list = []
    
    for entry in config_list:
        label = entry['label']
        params = entry['params']
        
        # パスを動的に生成
        target_dir = build_path_from_params(base_prefix, params)
        
        if not os.path.exists(target_dir):
            print(f"[Warning] Directory not found: {target_dir}")
            continue
            
        # reportファイルの検索パターン
        pattern = os.path.join(target_dir, "report_wk*.txt")
        files = glob.glob(pattern)
        
        for filepath in files:
            match = re.search(r"report_wk(\d+)\.txt", os.path.basename(filepath))
            if match:
                wk = int(match.group(1))
                if start_week is not None and wk < start_week: continue
                if end_week is not None and wk > end_week: continue

                metrics = parse_metrics(filepath)
                if metrics:
                    metrics['Week'] = wk
                    metrics['Method'] = label  # グラフの凡例にはJSONのlabelを使用
                    data_list.append(metrics)
                    
    return pd.DataFrame(data_list)

def parse_metrics(filepath):
    """
    レポートファイルから全ての指標を抽出する。
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'RMP': 0.0,
        'MIP': 0.0,  # MIP Time
        'MIP_Columns': 0.0, # MIP Variables
        'Filtered_Columns': 0.0, # Filtered Columns
        'Pool_Search': 0.0,
        'Solving_Shortest_Path_Problem': 0.0,
        'Total_Time': 0.0,
        'Gap_Absolute': None,
        'Pool_Hit_Rate': None,
        'Iterations': 0.0,
        'Stagnant_Iterations': 0.0
    }

    # --- 1. 基本的な指標の抽出 ---
    patterns = {
            'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
            'RMP': r"RMP Time\s*:\s*([\d\.]+)",
            'MIP': r"MIP Time\s*:\s*([\d\.]+)",
            'MIP_Columns': r"MIP Decision Variables\s*:\s*([\d]+)",
            'Filtered_Columns': r"MIP Filtered Columns\s*:\s*([\d]+)",
            'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
            'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
            'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
            'Pool_Hit_Rate': r"Pool Hit Rate\s*:\s*([\d\.]+)%",
            'Iterations': r"(?:MIP )?Iterations\s*:\s*([\d]+)"
        }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    # --- 補完・計算指標 ---

    # Total_Time補完
    if metrics['Total_Time'] == 0.0:
        metrics['Total_Time'] = (metrics['RMP'] + metrics['MIP'] + 
                                 metrics['Pool_Search'] + metrics['Solving_Shortest_Path_Problem'])

    # MIP Time per Column
    if metrics['MIP_Columns'] > 0:
        metrics['Time_Per_Column'] = metrics['MIP'] / metrics['MIP_Columns']
    else:
        metrics['Time_Per_Column'] = 0.0
        
    # Objective Value per Iteration
    if metrics['Iterations'] > 0:
        metrics['Obj_Per_Iter'] = metrics['Objective_Value'] / metrics['Iterations']
    else:
        metrics['Obj_Per_Iter'] = 0.0

    # Objective Value * MIP Time (積)
    metrics['Obj_Times_Time'] = metrics['Objective_Value'] * metrics['MIP']

    # 絶対ギャップ (Objective - LowerBound) の計算
    lower_bound = None
    match_rmp_lb = re.search(r"RMP Relaxed Value \(.*?\)\s*:\s*([\d\.,]+)", content)
    if match_rmp_lb:
        lower_bound = float(match_rmp_lb.group(1).replace(',', ''))
            
    if lower_bound is not None and metrics['Objective_Value'] > 0:
        metrics['Gap_Absolute'] = metrics['Objective_Value'] - lower_bound

    # --- 2. 停滞回数 (Stagnation) の計算 ---
    obj_values = []
    lines = content.splitlines()
    in_history_section = False
    
    for line in lines:
        if "Iteration History" in line:
            in_history_section = True
            continue
        if not in_history_section:
            continue
        
        stripped = line.strip()
        if not stripped: continue
        if "Iter" in line or "-----" in line: continue
        
        parts = line.split('|')
        if len(parts) >= 2:
            try:
                val_str = parts[1].strip().replace(',', '')
                val = float(val_str)
                obj_values.append(val)
            except ValueError:
                continue

    stagnant_count = 0
    for i in range(len(obj_values) - 1):
        if abs(obj_values[i] - obj_values[i+1]) < 1e-9:
            stagnant_count += 1
            
    metrics['Stagnant_Iterations'] = float(stagnant_count)

    return metrics

# ==========================================
# コスト内訳計算関数
# ==========================================

def load_problem_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return ShiftProblemData(config_path=config_path)

def calculate_cost_breakdown(prob, pool_csv_path, week):
    prob.generate_new_demand(period=week - 1)
    
    if not os.path.exists(pool_csv_path):
        return None

    try:
        df = pd.read_csv(pool_csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read {pool_csv_path}: {e}")
        return None
    
    if 'is_selected' not in df.columns:
        return None
        
    selected_df = df[df['is_selected'] == 1]
    
    if selected_df.empty:
        return {'Base Wage': 0.0, 'Mismatch Cost': 0.0, 'Understaffing Penalty': 0.0, 'Total Obj': 0.0}

    total_base_wage = 0.0
    total_mismatch_cost = 0.0
    supplied = np.zeros(prob.T)
    
    for _, row in selected_df.iterrows():
        emp_id = int(row['emp_id'])
        sched_str = str(row['schedule_pattern'])
        schedule = np.array([int(c) for c in sched_str])
        
        emp = prob.employees[emp_id]
        
        base_wage = np.sum(schedule * emp['hourly_wage'])
        total_base_wage += base_wage
        
        rho_cost = np.sum(schedule * emp['rho'])
        total_mismatch_cost += rho_cost
        
        supplied += schedule

    shortage = np.maximum(0, prob.demand - supplied)
    total_penalty = np.sum(shortage) * prob.big_m
    
    return {
        'Base Wage': total_base_wage,
        'Mismatch Cost': total_mismatch_cost,
        'Understaffing Penalty': total_penalty,
        'Total Obj': total_base_wage + total_mismatch_cost + total_penalty
    }

# ==========================================
# グラフ描画関数群
# ==========================================

def plot_comparison(df, prefix_name, methods, output_dir):
    """指標比較（集合棒グラフ）"""
    if df.empty: return

    # 比較したい指標一覧
    metrics_keys = [
        'Objective_Value', 'Total_Time', 'RMP', 'MIP', 
        'Pool_Search', 'Solving_Shortest_Path_Problem',
        'Gap_Absolute', 'Pool_Hit_Rate', 'Iterations',
        'MIP_Columns', 'Filtered_Columns', 'Time_Per_Column', 
        'Obj_Per_Iter', 'Obj_Times_Time'
    ]
    
    metric_titles = {
        'Objective_Value': 'Objective Value',
        'Total_Time': 'Total Computaion Time',
        'RMP': 'RMP Time',
        'MIP': 'MIP Time',
        'Pool_Search': 'Pool Search Time',
        'Solving_Shortest_Path_Problem': 'Shortest Path Calculation Time',
        'Gap_Absolute': 'Optimality Gap (Absolute Cost Difference)',
        'Pool_Hit_Rate': 'Pool Hit Rate (%)',
        'Iterations': 'Number of Iterations (Hatched = Stagnation)',
        'MIP_Columns': 'Number of MIP Columns',
        'Filtered_Columns': 'Number of Filtered Columns',
        'Time_Per_Column': 'MIP Time per Column (sec)',
        'Obj_Per_Iter': 'Objective Value per Iteration',
        'Obj_Times_Time': r'Objective Value $\times$ MIP Time'
    }

    colors = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
    ]
    all_weeks = sorted(df['Week'].unique())

    total_width = 0.8
    num_methods = len(methods)
    bar_width = total_width / num_methods

    # 出力先ディレクトリの作成
    save_dir = os.path.join(output_dir, "plots_comparison")
    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics_keys:
        if metric not in df.columns: continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        has_data = False
        
        # --- Iterationsの場合のみ積み上げ（停滞＋有効） ---
        if metric == 'Iterations':
            legend_patches = []
            
            for i, method in enumerate(methods):
                subset = df[df['Method'] == method].sort_values('Week')
                subset = subset.dropna(subset=[metric])

                if not subset.empty:
                    x_offset = (i - (num_methods - 1) / 2) * bar_width
                    x_values = subset['Week'] + x_offset
                    
                    total_iter = subset['Iterations']
                    stagnant_iter = subset.get('Stagnant_Iterations', pd.Series([0]*len(total_iter)))
                    effective_iter = total_iter - stagnant_iter
                    
                    base_color = colors[i % len(colors)]
                    
                    # 1. 有効な反復
                    ax1.bar(x_values, effective_iter,
                            width=bar_width,
                            color=base_color,
                            label=method,
                            alpha=0.9,
                            edgecolor='white',
                            linewidth=0.5)
                    
                    # 2. 停滞した反復（ハッチング）
                    ax1.bar(x_values, stagnant_iter,
                            bottom=effective_iter,
                            width=bar_width,
                            color=base_color,
                            alpha=0.4, 
                            hatch='///',
                            edgecolor='black',
                            linewidth=0.0)

                    has_data = True
                    legend_patches.append(Patch(facecolor=base_color, label=method, alpha=0.9))

            if has_data:
                legend_patches.append(Patch(facecolor='white', edgecolor='black', hatch='///', label='Stagnation', alpha=0.5))
                ax1.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

        # --- その他の指標 ---
        else:
            for i, method in enumerate(methods):
                subset = df[df['Method'] == method].sort_values('Week')
                subset = subset.dropna(subset=[metric])

                if not subset.empty:
                    x_offset = (i - (num_methods - 1) / 2) * bar_width
                    x_values = subset['Week'] + x_offset

                    ax1.bar(x_values, subset[metric],
                            width=bar_width,
                            color=colors[i % len(colors)],
                            label=method,
                            alpha=0.9,
                            edgecolor='white',
                            linewidth=0.5)
                    has_data = True
            
            if has_data:
                ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

        if has_data:
            ax1.set_title(f"{metric_titles.get(metric, metric)}", fontsize=16)
            ax1.set_xlabel("Week", fontsize=14)
            
            # Y軸ラベルの決定
            if metric == 'Pool_Hit_Rate':
                ylabel = "Rate (%)"
            elif metric in ['Objective_Value', 'Gap_Absolute']:
                ylabel = "Cost"
            elif metric in ['Iterations', 'MIP_Columns', 'Filtered_Columns']:
                ylabel = "Count"
            elif metric == 'Time_Per_Column':
                ylabel = "Time / Column (s)"
            elif metric == 'Obj_Per_Iter':
                ylabel = "Obj / Iteration"
            elif metric == 'Obj_Times_Time': 
                ylabel = r"Obj $\times$ Time"
            else:
                ylabel = "Time (s)"
            
            ax1.set_ylabel(ylabel, fontsize=14)
            ax1.set_xticks(all_weeks)
            ax1.set_xticklabels(all_weeks)
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            output_path = os.path.join(save_dir, f"compare_{metric}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved Comparison: {output_path}")

def plot_cost_breakdown_individual(data_dict, prefix_name, output_dir):
    """コスト内訳（積み上げ棒グラフ）"""
    weeks = sorted(list(set(k[0] for k in data_dict.keys())))
    methods = sorted(list(set(k[1] for k in data_dict.keys())))
    
    if not weeks: return

    components = ['Base Wage', 'Mismatch Cost', 'Understaffing Penalty']
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] 
    
    save_dir = os.path.join(output_dir, "plots_cost_breakdown")
    os.makedirs(save_dir, exist_ok=True)

    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_weeks = []
        subset_values = {c: [] for c in components}
        
        for wk in weeks:
            if (wk, method) in data_dict:
                res = data_dict[(wk, method)]
                valid_weeks.append(wk)
                for c in components:
                    subset_values[c].append(res.get(c, 0.0))
        
        if not valid_weeks:
            plt.close()
            continue
            
        indices = np.arange(len(valid_weeks))
        bottoms = np.zeros(len(valid_weeks))
        
        for i, comp in enumerate(components):
            vals = np.array(subset_values[comp])
            ax.bar(indices, vals, bottom=bottoms, 
                   label=comp, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
            bottoms += vals
            
        ax.set_title(f'Cost Breakdown: {method}', fontsize=16)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.set_xticks(indices)
        ax.set_xticklabels(valid_weeks, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), title="Cost Components")
        
        plt.tight_layout()
        output_path = os.path.join(save_dir, f"cost_breakdown_{method}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved Cost Breakdown: {output_path}")

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Prefix of the results directory (e.g. results_exp1)")
    parser.add_argument("config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--start_week", type=int, default=None, help="Start week")
    parser.add_argument("--end_week", type=int, default=None, help="End week")

    args = parser.parse_args()
    
    base_prefix = args.prefix
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config_list = json.load(f)

    # 1. Metricsの読み込みとグラフ化
    print(f"Loading metrics from directories starting with {base_prefix}...")
    df = load_metrics_from_json(base_prefix, args.config, args.start_week, args.end_week)
    
    # グラフの保存先はカレントディレクトリ（または任意のまとめフォルダ）
    output_dir = "." 
    
    if not df.empty:
        methods_in_plot = df['Method'].unique().tolist()
        print(f"Generating comparison plots for: {methods_in_plot}")
        plot_comparison(df, base_prefix, methods_in_plot, output_dir)
    else:
        print("Warning: No data found. Check directory names and week numbers.")

    # 2. コスト内訳の計算とグラフ化
    if ShiftProblemData is None:
        print("\nSkipping Cost Breakdown (ShiftProblemData not found).")
        return

    print("\n--- 2. Generating Cost Breakdown Plots ---")
    
    # Problem Configの場所を探す
    try:
        first_params = config_list[0]['params']
        first_dir = build_path_from_params(base_prefix, first_params)
        # build_path_from_params は .../method_name まで返すので、その親ディレクトリ(実験フォルダ)を取得
        exp_dir = os.path.dirname(first_dir)
        config_path = os.path.join(exp_dir, "problem_config.json")
        
        if not os.path.exists(config_path):
            config_path = "problem_config.json"
    except Exception:
         config_path = "problem_config.json"
        
    if os.path.exists(config_path):
        try:
            prob = load_problem_config(config_path)
            print(f"Loaded problem config from {config_path}")
        except Exception as e:
            print(f"Error loading problem config: {e}")
            return
    else:
        print(f"Warning: problem_config.json not found. Skipping cost breakdown.")
        return

    cost_data_store = {}
    
    if not df.empty:
        s_wk = args.start_week if args.start_week else int(df['Week'].min())
        e_wk = args.end_week if args.end_week else int(df['Week'].max())
    else:
        s_wk = args.start_week if args.start_week else 1
        e_wk = args.end_week if args.end_week else 5

    for wk in range(s_wk, e_wk + 1):
        for entry in config_list:
            label = entry['label']
            params = entry['params']
            
            method_dir = build_path_from_params(base_prefix, params)
            pool_file = os.path.join(method_dir, f"pool_wk{wk}.csv")
            
            if os.path.exists(pool_file):
                res = calculate_cost_breakdown(prob, pool_file, wk)
                if res:
                    cost_data_store[(wk, label)] = res

    if cost_data_store:
        plot_cost_breakdown_individual(cost_data_store, base_prefix, output_dir)
    else:
        print("No pool csv files found for cost breakdown.")

    print("-" * 40)
    print("Processing complete.")

if __name__ == "__main__":
    main()
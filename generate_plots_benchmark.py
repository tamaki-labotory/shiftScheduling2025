import os
import argparse
import re
import glob
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

def load_metrics_from_directory(target_dir, start_week=None, end_week=None):
    """
    指定されたディレクトリ直下のサブディレクトリ（手法名）を走査し、
    結果を読み込む
    """
    if not os.path.exists(target_dir):
        print(f"[Error] Directory not found: {target_dir}")
        return pd.DataFrame()

    data_list = []
    
    # ディレクトリ内のサブフォルダを取得（手法ディレクトリとみなす）
    # ただし、__pycache__ や 画像フォルダなどは除外する簡易フィルタを入れる
    ignore_list = ['__pycache__', 'plots_comparison', 'plots_cost_breakdown']
    subdirs = [d for d in os.listdir(target_dir) 
               if os.path.isdir(os.path.join(target_dir, d)) and d not in ignore_list]
    
    print(f"Detected methods (subdirectories): {subdirs}")

    for method in subdirs:
        method_dir = os.path.join(target_dir, method)
        
        # reportファイルの検索パターン
        pattern = os.path.join(method_dir, "report_wk*.txt")
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
                    metrics['Method'] = method  # ディレクトリ名をMethod名として使用
                    data_list.append(metrics)
                    
    return pd.DataFrame(data_list)

def parse_metrics(filepath):
    """
    レポートファイルから全ての指標を抽出する
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'RMP': 0.0,
        'MIP': 0.0,  
        'MIP_Columns': 0.0, 
        'Filtered_Columns': 0.0, 
        'Pool_Search': 0.0,
        'Solving_Shortest_Path_Problem': 0.0,
        'Total_Time': 0.0,
        'Gap_Absolute': None,
        'Pool_Hit_Rate': None,
        'Iterations': 0.0,
        'Stagnant_Iterations': 0.0
    }

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

    # ---------------------------------------------------------
    # 変更箇所: Objective Value * Total_Time (Execution Time) に変更
    # ---------------------------------------------------------
    metrics['Obj_Times_Time'] = metrics['Objective_Value'] * metrics['Total_Time']

    # 絶対ギャップ
    lower_bound = None
    match_rmp_lb = re.search(r"RMP Relaxed Value \(.*?\)\s*:\s*([\d\.,]+)", content)
    if match_rmp_lb:
        lower_bound = float(match_rmp_lb.group(1).replace(',', ''))
            
    if lower_bound is not None and metrics['Objective_Value'] > 0:
        metrics['Gap_Absolute'] = metrics['Objective_Value'] - lower_bound

    # 停滞回数
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
# RMS 指標計算 (新規追加)
# ==========================================

def calculate_rms_metric(df):
    """
    資料に基づき、対数正規化を用いたRMS評価指標を算出する
    S = sqrt(C_tilde^2 + T_tilde^2)
    
    1. 目的関数値(Objective_Value)と総計算時間(Total_Time)を常用対数変換
    2. 全データの最大値・最小値を用いて0-1に正規化
    3. RMSを算出
    """
    if df.empty:
        return df

    # 0や負の値の対策として極小値(1e-9)でクリップしてから対数をとる
    df['log_obj'] = np.log10(df['Objective_Value'].clip(lower=1e-9))
    df['log_time'] = np.log10(df['Total_Time'].clip(lower=1e-9))

    # 全週・全手法を通じた最大値・最小値を取得
    c_min = df['log_obj'].min()
    c_max = df['log_obj'].max()
    t_min = df['log_time'].min()
    t_max = df['log_time'].max()

    print("\n--- RMS Calculation Info ---")
    print(f"  Log Obj Range: [{c_min:.4f}, {c_max:.4f}]")
    print(f"  Log Time Range: [{t_min:.4f}, {t_max:.4f}]")

    # 正規化 (Max == Min の場合のゼロ除算回避)
    if c_max > c_min:
        df['norm_obj'] = (df['log_obj'] - c_min) / (c_max - c_min)
    else:
        df['norm_obj'] = 0.0 # 値が一定の場合は0とする

    if t_max > t_min:
        df['norm_time'] = (df['log_time'] - t_min) / (t_max - t_min)
    else:
        df['norm_time'] = 0.0

    # RMS算出
    df['RMS_Score'] = np.sqrt(df['norm_obj']**2 + df['norm_time']**2)
    
    return df

# ==========================================
# コスト内訳計算関数
# ==========================================

def load_problem_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return ShiftProblemData(config_path=config_path)

def calculate_cost_breakdown(prob, pool_csv_path, week):
    # 需要を更新（重要）
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

def plot_comparison(df, methods, output_dir):
    """指標比較（集合棒グラフ）"""
    if df.empty: return

    metrics_keys = [
        'Objective_Value', 'Total_Time', 'RMP', 'MIP', 
        'RMS_Score', # 追加: RMS
        'Pool_Search', 'Solving_Shortest_Path_Problem',
        'Gap_Absolute', 'Pool_Hit_Rate', 'Iterations',
        'MIP_Columns', 'Filtered_Columns', 'Time_Per_Column', 
        'Obj_Per_Iter', 'Obj_Times_Time'
    ]
    
    # ---------------------------------------------------------
    # 変更箇所: タイトルの変更とRMSの追加
    # ---------------------------------------------------------
    metric_titles = {
        'Objective_Value': 'Objective Value',
        'Total_Time': 'Total Computaion Time',
        'RMP': 'RMP Time',
        'MIP': 'MIP Time',
        'RMS_Score': 'RMS Score (Log-Normalized Cost & Time)', # 追加
        'Pool_Search': 'Pool Search Time',
        'Solving_Shortest_Path_Problem': 'Shortest Path Calculation Time',
        'Gap_Absolute': 'Optimality Gap (Absolute Cost Difference)',
        'Pool_Hit_Rate': 'Pool Hit Rate (%)',
        'Iterations': 'Number of Iterations (Hatched = Stagnation)',
        'MIP_Columns': 'Number of MIP Columns',
        'Filtered_Columns': 'Number of Filtered Columns',
        'Time_Per_Column': 'MIP Time per Column (sec)',
        'Obj_Per_Iter': 'Objective Value per Iteration',
        'Obj_Times_Time': r'Objective Value $\times$ Execution Time'
    }

    colors = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
    ]
    all_weeks = sorted(df['Week'].unique())

    total_width = 0.8
    num_methods = len(methods)
    bar_width = total_width / num_methods

    save_dir = os.path.join(output_dir, "plots_comparison")
    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics_keys:
        if metric not in df.columns: continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        has_data = False
        
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
                    
                    ax1.bar(x_values, effective_iter,
                            width=bar_width,
                            color=base_color,
                            label=method,
                            alpha=0.9,
                            edgecolor='white',
                            linewidth=0.5)
                    
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
            
            # Y軸ラベルの条件分岐
            if metric == 'RMS_Score':
                ylabel = "RMS Score (Lower is better)"
            elif metric == 'Pool_Hit_Rate':
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

def plot_cost_breakdown_individual(data_dict, output_dir):
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
    parser.add_argument("target_dir", type=str, help="Directory containing benchmark results (e.g. results_comparison_20emp)")
    parser.add_argument("--start_week", type=int, default=None, help="Start week")
    parser.add_argument("--end_week", type=int, default=None, help="End week")

    args = parser.parse_args()
    target_dir = args.target_dir
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory not found: {target_dir}")
        return

    # 1. Metricsの読み込み
    print(f"Loading metrics from {target_dir}...")
    df = load_metrics_from_directory(target_dir, args.start_week, args.end_week)
    
    if not df.empty:
        # ---------------------------------------------------------
        # 追加機能: RMS指標の計算
        # ---------------------------------------------------------
        df = calculate_rms_metric(df)
        
        methods_in_plot = sorted(df['Method'].unique().tolist())
        print(f"Generating comparison plots for: {methods_in_plot}")
        
        plot_comparison(df, methods_in_plot, target_dir)
    else:
        print("Warning: No report files found. Check directory structure.")

    # 2. コスト内訳の計算とグラフ化
    if ShiftProblemData is None:
        print("\nSkipping Cost Breakdown (ShiftProblemData not found).")
        return

    print("\n--- 2. Generating Cost Breakdown Plots ---")
    
    # problem_config.jsonを探す
    config_path = os.path.join(target_dir, "problem_config.json")
        
    if os.path.exists(config_path):
        try:
            prob = load_problem_config(config_path)
            print(f"Loaded problem config from {config_path}")
        except Exception as e:
            print(f"Error loading problem config: {e}")
            return
    else:
        if os.path.exists("problem_config.json"):
            prob = load_problem_config("problem_config.json")
            print("Loaded problem_config.json from current directory.")
        else:
            print(f"Warning: problem_config.json not found in {target_dir} or current dir. Skipping cost breakdown.")
            return

    cost_data_store = {}
    
    if not df.empty:
        s_wk = args.start_week if args.start_week else int(df['Week'].min())
        e_wk = args.end_week if args.end_week else int(df['Week'].max())
        methods = df['Method'].unique()
    else:
        s_wk = args.start_week if args.start_week else 1
        e_wk = args.end_week if args.end_week else 15
        ignore_list = ['__pycache__', 'plots_comparison', 'plots_cost_breakdown']
        methods = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and d not in ignore_list]

    for wk in range(s_wk, e_wk + 1):
        for method in methods:
            method_dir = os.path.join(target_dir, method)
            pool_file = os.path.join(method_dir, f"pool_wk{wk}.csv")
            
            if os.path.exists(pool_file):
                res = calculate_cost_breakdown(prob, pool_file, wk)
                if res:
                    cost_data_store[(wk, method)] = res

    if cost_data_store:
        plot_cost_breakdown_individual(cost_data_store, target_dir)
    else:
        print("No pool csv files found for cost breakdown.")

    print("-" * 40)
    print("Processing complete.")

if __name__ == "__main__":
    main()
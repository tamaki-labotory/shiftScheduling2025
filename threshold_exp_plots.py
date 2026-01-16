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
# 設定
# ==========================================
TARGET_METHODS = ['acc', 'pruning', 'std']
TARGET_THREADS = ['1', '1000', '100000', '10000000','ALL']
ROOT_DIR = "results_5emp/experiment_comparison_methods"

# 添付ファイル(generate_plots.py)で使用されていた色定義
COLORS = [
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
    '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
]

# ==========================================
# データ読み込み・解析関数
# ==========================================

def parse_metrics(filepath):
    """
    レポートファイルから指標を抽出する。
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'MIP': 0.0,
        'Total_Time': 0.0,
        'Gap_Absolute': None,
        'MIP_Columns': 0.0, # 新規追加: MIP列数
        # 以下は計算用またはTotal_Time補完用に取得するがプロットはしない
        'RMP': 0.0,
        'Pool_Search': 0.0,
        'Solving_Shortest_Path_Problem': 0.0
    }

    # パターン定義
    # 注意: MIP列数の正規表現はレポートの出力形式に合わせて調整が必要な場合があります
    # パターン定義
    patterns = {
            'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
            'RMP': r"RMP Time\s*:\s*([\d\.]+)",
            'MIP': r"MIP Time\s*:\s*([\d\.]+)",
            'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
            'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
            'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
            # 修正: "Variables: 1088 (Columns used in Final MIP)" の形式に対応
            'MIP_Columns': r"Variables\s*:\s*([\d]+)\s*\(Columns used in Final MIP\)"
    }

    # 補助的な検索（表記ゆれ対応用）
    # もし "Number of Columns" でヒットしなければ "Total Columns" や "Generated Columns" も試す例
    if not re.search(patterns['MIP_Columns'], content):
        patterns['MIP_Columns'] = r"(?:Total Columns|Generated Columns)\s*:\s*([\d]+)"

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    # Total_Time補完
    if metrics['Total_Time'] == 0.0:
        metrics['Total_Time'] = (metrics['RMP'] + metrics['MIP'] + 
                                 metrics['Pool_Search'] + metrics['Solving_Shortest_Path_Problem'])

    # 絶対ギャップ
    lower_bound = None
    match_rmp_lb = re.search(r"RMP Relaxed Value \(.*?\)\s*:\s*([\d\.,]+)", content)
    if match_rmp_lb:
        lower_bound = float(match_rmp_lb.group(1).replace(',', ''))
            
    if lower_bound is not None and metrics['Objective_Value'] > 0:
        metrics['Gap_Absolute'] = metrics['Objective_Value'] - lower_bound

    return metrics

def load_metrics_for_method(root_dir, method, threads, start_week=None, end_week=None):
    """
    指定された手法について、各Threadディレクトリからデータを読み込む
    """
    data_list = []

    for th in threads:
        target_dir = os.path.join(root_dir, method, f"Th_{th}")
        if not os.path.exists(target_dir):
            # print(f"  Warning: Directory not found: {target_dir}")
            continue

        pattern = os.path.join(target_dir, "report_wk*.txt")
        files = glob.glob(pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            match = re.search(r"report_wk(\d+)\.txt", filename)
            if match:
                wk = int(match.group(1))
                if start_week is not None and wk < start_week: continue
                if end_week is not None and wk > end_week: continue

                metrics = parse_metrics(filepath)
                if metrics:
                    metrics['Week'] = wk
                    metrics['Method'] = method
                    metrics['Thread'] = th
                    data_list.append(metrics)
    
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list)
    df['Thread'] = pd.Categorical(df['Thread'], categories=threads, ordered=True)
    
    return df.sort_values(by=['Thread', 'Week'])

# ==========================================
# コスト内訳計算関数
# ==========================================
# (変更なし)

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
    except Exception:
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
# グラフ描画関数群 (スレッド比較用)
# ==========================================

def plot_thread_comparison(df, method, threads, output_dir):
    """
    特定の手法について、スレッド数ごとの指標を比較（集合棒グラフ）
    """
    if df.empty: return

    # プロット対象のメトリクス（ご要望により変更）
    metrics_keys = [
        'Objective_Value', 
        'Total_Time', 
        'MIP', 
        'Gap_Absolute',
        'MIP_Columns' # 新規追加
    ]
    
    metric_titles = {
        'Objective_Value': f'Objective Value ({method})',
        'Total_Time': f'Total Computation Time ({method})',
        'MIP': f'MIP Time ({method})',
        'Gap_Absolute': f'Optimality Gap ({method})',
        'MIP_Columns': f'Number of Columns in MIP ({method})' # タイトル
    }

    # 指定された色を使用
    colors = COLORS 
    
    all_weeks = sorted(df['Week'].unique())
    total_width = 0.8
    num_threads = len(threads)
    bar_width = total_width / num_threads

    for metric in metrics_keys:
        if metric not in df.columns or (df[metric].sum() == 0 and metric != 'Objective_Value'):
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        has_data = False
        
        for i, th in enumerate(threads):
            subset = df[df['Thread'] == th].sort_values('Week')
            subset = subset.dropna(subset=[metric])

            if not subset.empty:
                x_offset = (i - (num_threads - 1) / 2) * bar_width
                x_values = subset['Week'] + x_offset

                # 色をCOLORSリストから取得して割り当て
                color_idx = i % len(colors)
                
                ax1.bar(x_values, subset[metric], width=bar_width, color=colors[color_idx],
                        label=f"Th_{th}", alpha=0.9, edgecolor='white', linewidth=0.5)
                has_data = True
        
        if has_data:
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Thread Count")
            ax1.set_title(metric_titles.get(metric, metric), fontsize=16)
            ax1.set_xlabel("Week", fontsize=14)
            
            # Y軸ラベルの設定
            if "Time" in metric: 
                ax1.set_ylabel("Time (s)", fontsize=14)
            elif "Columns" in metric:
                ax1.set_ylabel("Count", fontsize=14)
            else:
                ax1.set_ylabel("Value", fontsize=14)
            
            ax1.set_xticks(all_weeks)
            ax1.set_xticklabels(all_weeks)
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{method}_compare_{metric}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {output_path}")

def plot_cost_breakdown_individual(data_dict, file_suffix, output_dir):
    """
    コスト内訳（generate_plots.pyのロジックを流用しつつ色を調整）
    """
    weeks = sorted(list(set(k[0] for k in data_dict.keys())))
    # keys are (week, thread_key)
    thread_keys = sorted(list(set(k[1] for k in data_dict.keys()))) # e.g. "Th_1", "Th_1000"
    
    # スレッド数順に並べ替えたい場合のためのソート（文字列比較になるが概ねOK、厳密には数値変換が必要）
    # ここでは単純に文字列ソート
    
    if not weeks: return

    components = ['Base Wage', 'Mismatch Cost', 'Understaffing Penalty']
    # コスト内訳用の色は generate_plots.py に準拠 (緑、オレンジ、赤)
    cost_colors = ['#2ca02c', '#ff7f0e', '#d62728'] 
    
    for th_key in thread_keys:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_weeks = []
        subset_values = {c: [] for c in components}
        
        for wk in weeks:
            if (wk, th_key) in data_dict:
                res = data_dict[(wk, th_key)]
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
                   label=comp, color=cost_colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
            bottoms += vals
            
        ax.set_title(f'Cost Breakdown: {file_suffix} ({th_key})', fontsize=16)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.set_xticks(indices)
        ax.set_xticklabels(valid_weeks, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), title="Cost Components")
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"cost_breakdown_{th_key}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"    Saved Cost Breakdown: {output_path}")

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compare metrics across thread counts for each method.")
    parser.add_argument("--start_week", type=int, default=None, help="Start week")
    parser.add_argument("--end_week", type=int, default=None, help="End week")
    parser.add_argument("--config", type=str, default="problem_config.json", help="Path to problem_config.json")
    
    args = parser.parse_args()

    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' does not exist.")
        return

    summary_dir = "experiment_threshold_result/summary_thread_comparison"
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    print(f"Target Directory: {ROOT_DIR}")
    print(f"Methods: {TARGET_METHODS}")
    print(f"Threads: {TARGET_THREADS}")
    print(f"Output Directory: {summary_dir}\n")

    for method in TARGET_METHODS:
        print(f"--- Processing Method: {method} ---")
        
        # 1. データ読み込み
        df = load_metrics_for_method(ROOT_DIR, method, TARGET_THREADS, args.start_week, args.end_week)
        
        if df.empty:
            print(f"  No data found for method {method}. Skipping.")
            continue
        
        # 2. メトリクス比較グラフ作成 (項目変更・配色変更版)
        method_output_dir = os.path.join(summary_dir, method)
        if not os.path.exists(method_output_dir):
            os.makedirs(method_output_dir)
            
        plot_thread_comparison(df, method, TARGET_THREADS, method_output_dir)

        # 3. コスト内訳
        if ShiftProblemData is None:
            print("  Skipping Cost Breakdown (ShiftProblemData not found).")
            continue
            
        config_path = args.config
        if not os.path.exists(config_path):
             config_path = os.path.join(ROOT_DIR, "problem_config.json")
        
        try:
            prob = load_problem_config(config_path)
            
            cost_data_store = {}
            s_wk = args.start_week if args.start_week else df['Week'].min()
            e_wk = args.end_week if args.end_week else df['Week'].max()
            
            print(f"  Calculating Cost Breakdown for {method} (Weeks {s_wk}-{e_wk})...")
            
            for wk in range(int(s_wk), int(e_wk) + 1):
                for th in TARGET_THREADS:
                    pool_file = os.path.join(ROOT_DIR, method, f"Th_{th}", f"pool_wk{wk}.csv")
                    
                    if os.path.exists(pool_file):
                        res = calculate_cost_breakdown(prob, pool_file, wk)
                        if res:
                            cost_data_store[(wk, f"Th_{th}")] = res
            
            plot_cost_breakdown_individual(cost_data_store, method, method_output_dir)
            
        except Exception as e:
            print(f"  Error in Cost Breakdown for {method}: {e}")

    print("\nDone. All plots generated in:", summary_dir)

if __name__ == "__main__":
    main()
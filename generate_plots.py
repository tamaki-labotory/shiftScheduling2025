import os
import argparse
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# problem.py が同じディレクトリにあることを前提とします
try:
    from problem import ShiftProblemData
except ImportError:
    ShiftProblemData = None
    print("Warning: Could not import ShiftProblemData. Cost breakdown plots will be skipped.")

# ==========================================
# データ読み込み・解析関数
# ==========================================

def parse_metrics(filepath):
    """
    レポートファイルから全ての指標（時間・コスト・ギャップ・プールヒット率）を抽出する。
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'RMP': 0.0,
        'MIP': 0.0,
        'Pool_Search': 0.0,
        'Solving_Shortest_Path_Problem': 0.0,
        'Total_Time': 0.0,
        'Gap_Absolute': None,
        'Pool_Hit_Rate': None
    }

    # 基本的な指標の抽出
    patterns = {
            'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
            'RMP': r"RMP Time\s*:\s*([\d\.]+)",
            'MIP': r"MIP Time\s*:\s*([\d\.]+)",
            'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
            'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
            'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
            'Pool_Hit_Rate': r"Pool Hit Rate\s*:\s*([\d\.]+)%"
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

    # 絶対ギャップ (Objective - LowerBound) の計算
    lower_bound = None
    # match_mip_lb = re.search(r"MIP Best Bound \(Final\)\s*:\s*([\d\.,]+)", content)
    # if match_mip_lb:
    #     lower_bound = float(match_mip_lb.group(1).replace(',', ''))
    
    if lower_bound is None:
        match_rmp_lb = re.search(r"RMP Relaxed Value \(.*?\)\s*:\s*([\d\.,]+)", content)
        if match_rmp_lb:
            lower_bound = float(match_rmp_lb.group(1).replace(',', ''))
            
    if lower_bound is not None and metrics['Objective_Value'] > 0:
        metrics['Gap_Absolute'] = metrics['Objective_Value'] - lower_bound

    return metrics

def load_all_metrics_data(base_dir, methods, start_week=None, end_week=None):
    """
    report_wk*.txt からメトリクスデータを読み込む
    """
    data_list = []

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            continue

        pattern = os.path.join(method_dir, "report_wk*.txt")
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
                    data_list.append(metrics)
    
    if not data_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_list)
    return df.sort_values(by=['Method', 'Week'])

# ==========================================
# コスト内訳計算関数
# ==========================================

def load_problem_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return ShiftProblemData(config_path=config_path)

def calculate_cost_breakdown(prob, pool_csv_path, week):
    """
    CSVから採用されたスケジュールを読み込み、コストの内訳を計算する
    """
    prob.generate_new_demand(period=week - 1)
    
    if not os.path.exists(pool_csv_path):
        return None

    try:
        df = pd.read_csv(pool_csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read {pool_csv_path}: {e}")
        return None
    
    # is_selected 列の確認
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
        
        # 基本給
        base_wage = np.sum(schedule * emp['hourly_wage'])
        total_base_wage += base_wage
        
        # 不一致コスト
        rho_cost = np.sum(schedule * emp['rho'])
        total_mismatch_cost += rho_cost
        
        supplied += schedule

    # 欠員ペナルティ
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

def plot_time_breakdown(df, emp, method, output_dir):
    """処理時間内訳（積み上げ棒グラフ）"""
    subset = df[df['Method'] == method].sort_values('Week')
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    weeks = subset['Week']
    rmp = subset['RMP']
    mip = subset['MIP']
    pool_search = subset['Pool_Search']
    shortest_path_calculation = subset['Solving_Shortest_Path_Problem']
    total_time = subset['Total_Time']

    ax.bar(weeks, rmp, label='RMP', color='#ff9999')
    ax.bar(weeks, mip, bottom=rmp, label='MIP', color='#66b3ff')
    ax.bar(weeks, pool_search, bottom=rmp + mip, label='Pool Search', color='#99ff99')
    ax.bar(weeks, shortest_path_calculation, bottom=rmp + mip + pool_search, label='Shortest Path', color='#ffcc99')

    ax.plot(weeks, total_time, color='red', marker='o', linestyle='-', linewidth=2, label='Total Time')

    ax.set_title(f'Time Breakdown: {method} (N={emp})', fontsize=16)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_xticks(weeks)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"breakdown_n{emp}_{method}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved Time Breakdown: {output_path}")

def plot_comparison(df, emp, methods, output_dir):
    """指標比較（集合棒グラフ）: 離散的な値を明確にするため棒グラフを使用"""
    if df.empty: return

    metrics_keys = [
        'Objective_Value', 'Total_Time', 'RMP', 'MIP', 
        'Pool_Search', 'Solving_Shortest_Path_Problem',
        'Gap_Absolute', 'Pool_Hit_Rate'
    ]
    cg_only_metrics = ['MIP', 'RMP', 'Pool_Search', 'Solving_Shortest_Path_Problem', 'Pool_Hit_Rate']
    
    metric_titles = {
        'Objective_Value': 'Objective Value',
        'Total_Time': 'Total Computaion Time',
        'RMP': 'RMP Time',
        'MIP': 'MIP Time',
        'Pool_Search': 'Pool Search Time',
        'Solving_Shortest_Path_Problem': 'Shortest Path Calculation Time',
        'Gap_Absolute': 'Optimality Gap (Absolute Cost Difference)',
        'Pool_Hit_Rate': 'Pool Hit Rate (%)'
    }

    # 棒グラフ用の色設定（少し淡い色にして重なりを防ぐ）
    colors = [
        '#4E79A7', # Blue (落ち着いた青)
        '#F28E2B', # Orange (明るいオレンジ)
        '#E15759', # Red (ソフトな赤)
        '#76B7B2', # Teal (青緑)
        '#59A14F', # Green (自然な緑)
        '#EDC948', # Yellow (視認性の良い濃い黄色)
        '#B07AA1', # Purple (紫)
        '#FF9DA7', # Pink (ピンク)
        '#9C755F', # Brown (茶)
        '#BAB0AC'  # Gray (グレー)
    ]
    all_weeks = sorted(df['Week'].unique())

    # 棒グラフの幅設定
    # 手法の数に応じて幅を調整（最大0.8のスペースを分け合う）
    total_width = 0.8
    num_methods = len(methods)
    bar_width = total_width / num_methods

    for metric in metrics_keys:
        if metric not in df.columns: continue

        fig, ax1 = plt.subplots(figsize=(12, 6)) # 横幅を少し広げる
        has_data = False
        
        for i, method in enumerate(methods):
            if method == 'exact' and metric in cg_only_metrics: continue

            subset = df[df['Method'] == method].sort_values('Week')
            subset = subset.dropna(subset=[metric])

            if not subset.empty:
                # X軸の位置計算：中心から左右に展開
                # (i - (num_methods - 1) / 2) * bar_width
                x_offset = (i - (num_methods - 1) / 2) * bar_width
                x_values = subset['Week'] + x_offset

                # 棒グラフの描画
                ax1.bar(x_values, subset[metric],
                        width=bar_width,
                        color=colors[i % len(colors)],
                        label=method,
                        alpha=0.9,
                        edgecolor='white', # 棒の境界を白くして区切りを明確に
                        linewidth=0.5)
                has_data = True
        
        if has_data:
            ax1.set_title(f"{metric_titles.get(metric, metric)} (N={emp})", fontsize=16)
            ax1.set_xlabel("Week", fontsize=14)
            
            if metric == 'Pool_Hit_Rate':
                ax1.set_ylabel("Rate (%)", fontsize=14)
            elif metric in ['Objective_Value', 'Gap_Absolute']:
                ax1.set_ylabel("Cost", fontsize=14)
            else:
                ax1.set_ylabel("Time (s)", fontsize=14)
            
            # X軸の目盛りを整数（週）に固定
            ax1.set_xticks(all_weeks)
            ax1.set_xticklabels(all_weeks) # 明示的にラベルを設定
            
            # グリッドはY軸のみ（横線）に入れるのが棒グラフの定石
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
            # 凡例
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"compare_{metric}_n{emp}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved Comparison (Bar): {output_path}")

def plot_cost_breakdown_individual(data_dict, emp, output_dir):
    """
    コスト内訳（積み上げ棒グラフ）を手法ごとに別ファイルで出力
    """
    weeks = sorted(list(set(k[0] for k in data_dict.keys())))
    methods = sorted(list(set(k[1] for k in data_dict.keys())))
    
    if not weeks:
        print("  [Cost Breakdown] No valid data available to plot.")
        return

    components = ['Base Wage', 'Mismatch Cost', 'Understaffing Penalty']
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] # 緑, オレンジ, 赤
    
    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # この手法に関するデータを抽出
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
            
        ax.set_title(f'Cost Breakdown: {method} (N={emp})', fontsize=16)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Cost', fontsize=12)
        ax.set_xticks(indices)
        ax.set_xticklabels(valid_weeks, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # 凡例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), title="Cost Components")
        
        plt.tight_layout()
        
        # ファイル名を個別に設定
        output_path = os.path.join(output_dir, f"cost_breakdown_n{emp}_{method}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"  Saved Cost Breakdown: {output_path}")

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate all plots (Metrics & Cost Breakdown).")
    parser.add_argument("emp", type=str, help="Number of employees (e.g., 20)")
    parser.add_argument("methods", nargs='+', help="List of methods to include (e.g., exact std acc pruning)")
    
    parser.add_argument("--start_week", type=int, default=None, help="Start week")
    parser.add_argument("--end_week", type=int, default=None, help="End week")
    
    args = parser.parse_args()
    base_dir = f"results_{args.emp}emp"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    print(f"Target Directory: {base_dir}")
    print(f"Methods: {', '.join(args.methods)}")
    
    # 1. 指標データ(Metrics)の読み込みとグラフ化
    print("\n--- 1. Generating Performance Metrics Plots ---")
    df = load_all_metrics_data(base_dir, args.methods, args.start_week, args.end_week)
    
    if not df.empty:
        for method in args.methods:
            plot_time_breakdown(df, args.emp, method, base_dir)
        plot_comparison(df, args.emp, args.methods, base_dir)
    else:
        print("Warning: No report_wk*.txt files found. Skipping metrics plots.")

    # 2. コスト内訳(Cost Components)の計算とグラフ化
    if ShiftProblemData is None:
        print("\nSkipping Cost Breakdown (ShiftProblemData not found).")
        return

    print("\n--- 2. Generating Cost Breakdown Plots ---")
    config_path = os.path.join(base_dir, "problem_config.json")
    try:
        prob = load_problem_config(config_path)
    except Exception as e:
        print(f"Error loading problem config: {e}")
        return

    cost_data_store = {}
    
    # データ範囲の決定
    if not df.empty:
        s_wk = args.start_week if args.start_week else df['Week'].min()
        e_wk = args.end_week if args.end_week else df['Week'].max()
    else:
        s_wk = args.start_week if args.start_week else 1
        e_wk = args.end_week if args.end_week else 5

    print(f"Processing CSVs for Weeks {s_wk} to {e_wk}...")

    for wk in range(int(s_wk), int(e_wk) + 1):
        for method in args.methods:
            pool_file = os.path.join(base_dir, method, f"pool_wk{wk}.csv")
            if os.path.exists(pool_file):
                res = calculate_cost_breakdown(prob, pool_file, wk)
                if res:
                    cost_data_store[(wk, method)] = res
    
    # ★変更: 個別出力用の関数を呼び出し
    plot_cost_breakdown_individual(cost_data_store, args.emp, base_dir)

    print("-" * 40)
    print(f"All graphs have been saved to: {base_dir}/")

if __name__ == "__main__":
    main()
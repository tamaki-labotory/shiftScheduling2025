import os
import argparse
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ユーティリティ関数
# ==========================================

def find_target_directory(emp):
    """
    カレントディレクトリ内で 'schedule_plots_{emp}emp' で始まるフォルダを探す。
    """
    pattern = f"schedule_plots_{emp}emp_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        return None
    
    # 複数見つかった場合はソートして先頭を使用
    if len(dirs) > 1:
        dirs.sort()
        print(f"Warning: Multiple directories found. Using: {dirs[0]}")
    
    return dirs[0]

def parse_metrics(filepath):
    """
    レポートファイルから全ての指標（時間・コスト）を抽出する。
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'RMP_LP': 0.0,
        'RMP_MIP': 0.0,
        'Pool_Search': 0.0,
        'Graph_Search': 0.0,
        'Total_Time': 0.0
    }

    patterns = {
        'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
        'RMP_LP': r"RMP Time \(LP\)\s*:\s*([\d\.]+)",
        'RMP_MIP': r"RMP Time \(MIP\)\s*:\s*([\d\.]+)",
        'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
        'Graph_Search': r"Graph Search Time\s*:\s*([\d\.]+)",
        'Total_Time': r"Execution Time\s*:\s*([\d\.]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    # Total_Timeが0または取得失敗の場合、内訳の合計で補完
    if metrics['Total_Time'] == 0.0:
        metrics['Total_Time'] = (metrics['RMP_LP'] + metrics['RMP_MIP'] + 
                                 metrics['Pool_Search'] + metrics['Graph_Search'])

    return metrics

def load_all_data(target_dir, methods):
    """
    指定されたディレクトリから、全手法・全週のデータを読み込みDataFrame化する。
    """
    data_list = []

    for method in methods:
        # ファイル名パターン: report_wk{週}_{手法}.txt
        pattern = os.path.join(target_dir, f"report_wk*_{method}.txt")
        files = glob.glob(pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # 週数を抽出
            match = re.search(r"report_wk(\d+)_", filename)
            if match:
                wk = int(match.group(1))
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
# グラフ描画関数
# ==========================================

def plot_time_breakdown(df, emp, method, output_dir):
    """
    【内訳グラフ】特定の手法の処理時間内訳（積み上げ棒グラフ）を作成
    """
    subset = df[df['Method'] == method].sort_values('Week')
    if subset.empty:
        print(f"  [Breakdown] No data for method: {method}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    weeks = subset['Week']
    rmp_lp = subset['RMP_LP']
    rmp_mip = subset['RMP_MIP']
    pool_search = subset['Pool_Search']
    graph_search = subset['Graph_Search']
    total_time = subset['Total_Time']

    # 色設定
    c_lp = '#ff9999'    # Light Red
    c_mip = '#66b3ff'   # Light Blue
    c_pool = '#99ff99'  # Light Green
    c_graph = '#ffcc99' # Light Orange

    # 積み上げ棒グラフ
    ax.bar(weeks, rmp_lp, label='RMP (LP)', color=c_lp)
    ax.bar(weeks, rmp_mip, bottom=rmp_lp, label='RMP (Final MIP)', color=c_mip)
    ax.bar(weeks, pool_search, bottom=rmp_lp + rmp_mip, label='Pool Search', color=c_pool)
    ax.bar(weeks, graph_search, bottom=rmp_lp + rmp_mip + pool_search, label='Graph Search', color=c_graph)

    # 合計時間の折れ線
    ax.plot(weeks, total_time, color='red', marker='o', linewidth=2, label='Total Time')

    ax.set_title(f'Time Breakdown: {method} (N={emp})', fontsize=16)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_xticks(weeks)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 凡例を枠外へ
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"breakdown_n{emp}_{method}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved Breakdown: {output_path}")

def plot_comparison(df, emp, methods, output_dir):
    """
    【比較グラフ】全手法の指標比較（折れ線グラフ）を作成
    """
    if df.empty:
        return

    metrics_keys = ['Objective_Value', 'Total_Time', 'RMP_LP', 'RMP_MIP', 'Pool_Search', 'Graph_Search']
    
    metric_titles = {
        'Objective_Value': 'Objective Value (Cost)',
        'Total_Time': 'Total Execution Time',
        'RMP_LP': 'RMP (LP) Time',
        'RMP_MIP': 'RMP (Final MIP) Time',
        'Pool_Search': 'Pool Search Time',
        'Graph_Search': 'Graph Search Time'
    }

    markers = ['o', 's', '^', 'D', 'x', '*']
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    
    # 全データに含まれる週のユニークなリスト（X軸用）
    all_weeks = sorted(df['Week'].unique())

    for metric in metrics_keys:
        plt.figure(figsize=(10, 6))
        has_data = False

        for i, method in enumerate(methods):
            subset = df[df['Method'] == method].sort_values('Week')
            if not subset.empty:
                plt.plot(subset['Week'], subset[metric],
                         marker=markers[i % len(markers)],
                         color=colors[i % len(colors)],
                         label=method, linewidth=2)
                has_data = True
        
        if has_data:
            plt.title(f"Comparison: {metric_titles[metric]} (N={emp})", fontsize=14)
            plt.xlabel("Week", fontsize=12)
            if metric == 'Objective_Value':
                plt.ylabel("Cost", fontsize=12)
            else:
                plt.ylabel("Time (s)", fontsize=12)
            
            plt.xticks(all_weeks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            output_path = os.path.join(output_dir, f"compare_{metric}_n{emp}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"  Saved Comparison: {output_path}")

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate ALL graphs (Breakdown & Comparison) in one go.")
    parser.add_argument("emp", type=str, help="Number of employees (e.g., 15)")
    parser.add_argument("methods", nargs='+', help="List of methods (e.g., exact std pool aging)")
    
    args = parser.parse_args()

    # ディレクトリ特定
    target_dir = find_target_directory(args.emp)
    if not target_dir:
        print(f"Error: Directory starting with 'schedule_plots_{args.emp}emp_' not found.")
        return

    # 出力ディレクトリ作成
    output_dir = f"summary_graphs_n{args.emp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Target Directory: {target_dir}")
    print(f"Processing Methods: {', '.join(args.methods)}")
    print("-" * 40)

    # 1. データのロード
    print("Loading data...")
    df = load_all_data(target_dir, args.methods)
    
    if df.empty:
        print("Error: No data found for any of the specified methods.")
        return

    # 2. 内訳グラフの作成 (Breakdown Plots)
    print("\nGenerating Breakdown Plots (Stacked Bars)...")
    for method in args.methods:
        plot_time_breakdown(df, args.emp, method, output_dir)

    # 3. 比較グラフの作成 (Comparison Plots)
    print("\nGenerating Comparison Plots (Line Charts)...")
    plot_comparison(df, args.emp, args.methods, output_dir)

    print("-" * 40)
    print(f"All graphs have been saved to: {output_dir}")

if __name__ == "__main__":
    main()
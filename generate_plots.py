import os
import argparse
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ユーティリティ関数
# ==========================================

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
        'Solving_Shortest_Path_Problem': 0.0,
        'Total_Time': 0.0
    }

    patterns = {
        'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
        'RMP_LP': r"RMP Time \(LP\)\s*:\s*([\d\.]+)",
        'RMP_MIP': r"RMP Time \(MIP\)\s*:\s*([\d\.]+)",
        'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
        'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
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
                                 metrics['Pool_Search'] + metrics['Solving_Shortest_Path_Problem'])

    return metrics

def load_all_data(base_dir, methods):
    """
    指定されたディレクトリ構造からデータを読み込む。
    構造: base_dir/{method}/report_wk{week}.txt
    """
    data_list = []

    for method in methods:
        # ディレクトリパス: results_{emp}emp/{method}
        method_dir = os.path.join(base_dir, method)
        
        if not os.path.exists(method_dir):
            print(f"Warning: Directory not found for method '{method}': {method_dir}")
            continue

        # ファイル探索: report_wk*.txt
        pattern = os.path.join(method_dir, "report_wk*.txt")
        files = glob.glob(pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # 週数を抽出 (例: report_wk1.txt -> 1)
            match = re.search(r"report_wk(\d+)\.txt", filename)
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
    shortest_path_calculation = subset['Solving_Shortest_Path_Problem']
    total_time = subset['Total_Time']

    # 色設定
    c_lp = '#ff9999'    # Light Red
    c_mip = '#66b3ff'   # Light Blue
    c_pool = '#99ff99'  # Light Green
    c_graph = '#ffcc99' # Light Orange

    # 積み上げ棒グラフ
    ax.bar(weeks, rmp_lp, label='RMP (LP)', color=c_lp)
    ax.bar(weeks, rmp_mip, bottom=rmp_lp, label='RMP (MIP)', color=c_mip)
    ax.bar(weeks, pool_search, bottom=rmp_lp + rmp_mip, label='Pool Search', color=c_pool)
    ax.bar(weeks, shortest_path_calculation, bottom=rmp_lp + rmp_mip + pool_search, label='Shortest Path Problem', color=c_graph)

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
    
    # 出力先: results_{emp}emp/breakdown_n{emp}_{method}.png
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

    metrics_keys = ['Objective_Value', 'Total_Time', 'RMP_LP', 'RMP_MIP', 'Pool_Search', 'Solving_Shortest_Path_Problem']
    
    metric_titles = {
        'Objective_Value': 'Objective Value (Cost)',
        'Total_Time': 'Total Computaion Time',
        'RMP_LP': 'RMP (LP) Time',
        'RMP_MIP': 'RMP (MIP) Time',
        'Pool_Search': 'Pool Search Time',
        'Solving_Shortest_Path_Problem': 'Shortest Path Calculation Time'
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
            plt.title(f"{metric_titles[metric]} (N={emp})", fontsize=14)
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
    parser = argparse.ArgumentParser(description="Generate graphs from results_{emp}emp directory.")
    parser.add_argument("emp", type=str, help="Number of employees (e.g., 20)")
    parser.add_argument("methods", nargs='+', help="List of methods to include (e.g., exact std acc pruning smart)")
    
    args = parser.parse_args()

    # ディレクトリパスの構築
    # 入力も出力も results_{emp}emp
    base_dir = f"results_{args.emp}emp"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    print(f"Target Directory: {base_dir}")
    print(f"Processing Methods: {', '.join(args.methods)}")
    print("-" * 40)

    # 1. データのロード
    print("Loading data...")
    df = load_all_data(base_dir, args.methods)
    
    if df.empty:
        print("Error: No data found. Check directory structure or method names.")
        return

    # 2. 内訳グラフの作成
    print("\nGenerating Breakdown Plots (Stacked Bars)...")
    for method in args.methods:
        plot_time_breakdown(df, args.emp, method, base_dir)

    # 3. 比較グラフの作成
    print("\nGenerating Comparison Plots (Line Charts)...")
    plot_comparison(df, args.emp, args.methods, base_dir)

    print("-" * 40)
    print(f"All graphs have been saved to: {base_dir}/")

if __name__ == "__main__":
    main()
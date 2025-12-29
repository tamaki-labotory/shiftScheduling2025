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
    レポートファイルから全ての指標（時間・コスト・ギャップ）を抽出する。
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
        'Integrality_Gap': None  # ギャップ値（初期値はNone）
    }

    # 正規表現パターンの定義
    # レポートのフォーマットに合わせて調整してください
    patterns = {
            'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
            'RMP': r"RMP Time\s*:\s*([\d\.]+)",
            'MIP': r"MIP Time\s*:\s*([\d\.]+)",
            'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
            'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
            'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
            # "Final MIP Gap : 0.0405 %" のような行を想定
            'Integrality_Gap': r"(?:Final MIP Gap|Integrality Gap)\s*:\s*([\d\.]+)"
        }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    # Total_Timeが0または取得失敗の場合、内訳の合計で補完
    if metrics['Total_Time'] == 0.0:
        metrics['Total_Time'] = (metrics['RMP'] + metrics['MIP'] + 
                                 metrics['Pool_Search'] + metrics['Solving_Shortest_Path_Problem'])

    return metrics

def load_all_data(base_dir, methods, start_week=None, end_week=None):
    """
    指定されたディレクトリ構造からデータを読み込む。
    構造: base_dir/{method}/report_wk{week}.txt
    """
    data_list = []

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        
        if not os.path.exists(method_dir):
            print(f"Warning: Directory not found for method '{method}': {method_dir}")
            continue

        pattern = os.path.join(method_dir, "report_wk*.txt")
        files = glob.glob(pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            match = re.search(r"report_wk(\d+)\.txt", filename)
            if match:
                wk = int(match.group(1))
                
                if start_week is not None and wk < start_week:
                    continue
                if end_week is not None and wk > end_week:
                    continue

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
    rmp = subset['RMP']
    mip = subset['MIP']
    pool_search = subset['Pool_Search']
    shortest_path_calculation = subset['Solving_Shortest_Path_Problem']
    total_time = subset['Total_Time']

    c_rmp = '#ff9999'    # Light Red
    c_mip = '#66b3ff'    # Light Blue
    c_pool = '#99ff99'   # Light Green
    c_graph = '#ffcc99'  # Light Orange

    ax.bar(weeks, rmp, label='RMP', color=c_rmp)
    ax.bar(weeks, mip, bottom=rmp, label='MIP', color=c_mip)
    ax.bar(weeks, pool_search, bottom=rmp + mip, label='Pool Search', color=c_pool)
    ax.bar(weeks, shortest_path_calculation, bottom=rmp + mip + pool_search, label='Shortest Path Problem', color=c_graph)

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
    print(f"  Saved Breakdown: {output_path}")

def plot_comparison(df, emp, methods, output_dir):
    """
    【比較グラフ】全手法の指標比較（折れ線グラフ）を作成
    MIP Time と Integrality Gap を個別のグラフとして出力するように修正
    """
    if df.empty:
        return

    # Integrality_Gap をリストに追加
    metrics_keys = [
        'Objective_Value', 
        'Total_Time', 
        'RMP', 
        'MIP', 
        'Pool_Search', 
        'Solving_Shortest_Path_Problem',
        'Integrality_Gap'  # 追加
    ]
    
    metric_titles = {
        'Objective_Value': 'Objective Value (Cost)',
        'Total_Time': 'Total Computaion Time',
        'RMP': 'RMP Time',
        'MIP': 'MIP Time',
        'Pool_Search': 'Pool Search Time',
        'Solving_Shortest_Path_Problem': 'Shortest Path Calculation Time',
        'Integrality_Gap': 'Integrality Gap (Final)' # 追加
    }

    markers = ['o', 's', '^', 'D', 'x', '*']
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    
    all_weeks = sorted(df['Week'].unique())

    for metric in metrics_keys:
        # データフレームにカラムが存在しない場合はスキップ
        if metric not in df.columns:
            continue

        fig, ax1 = plt.subplots(figsize=(10, 6))
        has_data = False
        
        for i, method in enumerate(methods):
            subset = df[df['Method'] == method].sort_values('Week')
            
            # Gapの場合、NoneやNaNを除外してプロットできるデータがあるか確認
            if metric == 'Integrality_Gap':
                subset = subset.dropna(subset=[metric])

            if not subset.empty:
                ax1.plot(subset['Week'], subset[metric],
                         marker=markers[i % len(markers)],
                         color=colors[i % len(colors)],
                         label=method, linewidth=2)
                has_data = True
        
        if has_data:
            ax1.set_title(f"{metric_titles.get(metric, metric)} (N={emp})", fontsize=14)
            ax1.set_xlabel("Week", fontsize=12)
            
            # Y軸ラベルの切り替え
            if metric == 'Objective_Value':
                ax1.set_ylabel("Cost", fontsize=12)
            elif metric == 'Integrality_Gap':
                ax1.set_ylabel("Gap (%)", fontsize=12)
            else:
                ax1.set_ylabel("Time (s)", fontsize=12)
            
            ax1.set_xticks(all_weeks)
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"compare_{metric}_n{emp}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"  Saved Comparison: {output_path}")

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate graphs from results_{emp}emp directory.")
    parser.add_argument("emp", type=str, help="Number of employees (e.g., 20)")
    parser.add_argument("methods", nargs='+', help="List of methods to include (e.g., exact std acc pruning smart)")
    
    parser.add_argument("--start_week", type=int, default=None, help="Start week for plotting (inclusive)")
    parser.add_argument("--end_week", type=int, default=None, help="End week for plotting (inclusive)")
    
    args = parser.parse_args()

    base_dir = f"results_{args.emp}emp"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    print(f"Target Directory: {base_dir}")
    print(f"Processing Methods: {', '.join(args.methods)}")
    if args.start_week or args.end_week:
        print(f"Week Range: {args.start_week if args.start_week else 'Min'} to {args.end_week if args.end_week else 'Max'}")
    print("-" * 40)

    print("Loading data...")
    df = load_all_data(base_dir, args.methods, args.start_week, args.end_week)
    
    if df.empty:
        print("Error: No data found. Check directory structure or method names (and week range).")
        return

    print("\nGenerating Breakdown Plots (Stacked Bars)...")
    for method in args.methods:
        plot_time_breakdown(df, args.emp, method, base_dir)

    print("\nGenerating Comparison Plots (Line Charts)...")
    plot_comparison(df, args.emp, args.methods, base_dir)

    print("-" * 40)
    print(f"All graphs have been saved to: {base_dir}/")

if __name__ == "__main__":
    main()
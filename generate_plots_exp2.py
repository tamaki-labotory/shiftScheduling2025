import os
import argparse
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# matplotlibの設定 (LaTeX風フォントを使用)
# ==========================================
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

# ==========================================
# パス生成・データ読み込み関数
# ==========================================
def build_path_from_params(base_prefix, params):
    """
    指定されたディレクトリ構造に従ってパスを生成する
    JSON内のパラメータ(max_threshold, method)を使用
    """
    t_val = params.get('max_threshold')
    method = params.get('method')
    
    # ディレクトリ名の構築ルール
    if method == 'std':
        dir_name = f"{base_prefix}_rc_{t_val}"
    else:
        dir_name = f"{base_prefix}_rc_{t_val}_{method}"
    
    path = os.path.join(dir_name, method)
    return path

def parse_metrics(filepath):
    """
    レポートファイルから指標を抽出する
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'MIP_Time': 0.0,
        'MIP_Columns': 0.0,
        'Filtered_Columns': 0.0,
        'Total_Time': 0.0
    }

    # 正規表現パターン
    patterns = {
        'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
        'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
        'MIP_Time': r"MIP Time\s*:\s*([\d\.]+)",
        'MIP_Columns': r"MIP Decision Variables\s*:\s*([\d]+)",
        'Filtered_Columns': r"MIP Filtered Columns\s*:\s*([\d]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    return metrics

def load_metrics_from_json(base_prefix, config_path, start_week=None, end_week=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_list = json.load(f)
        
    data_list = []
    ordered_labels = [] # グラフ描画順序をJSON通りに保つため
    seen_labels = set()
    
    for entry in config_list:
        raw_label = entry['label']
        
        # 数式表記の自動変換 (例: 10^3 -> $10^3$)
        if '$' not in raw_label:
            label = re.sub(r'10\^(\d+)', r'$10^{\1}$', raw_label)
        else:
            label = raw_label
            
        if label not in seen_labels:
            ordered_labels.append(label)
            seen_labels.add(label)
            
        params = entry['params']
        
        # パス生成
        target_dir = build_path_from_params(base_prefix, params)
        
        if not os.path.exists(target_dir):
            print(f"[Warning] Directory not found: {target_dir}")
            continue
            
        # ファイル探索
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
                    metrics['Method'] = label
                    
                    # --- 追加計算 1: MIP Time / MIP Columns ---
                    if metrics['MIP_Columns'] > 0:
                        metrics['Time_Per_Column'] = metrics['MIP_Time'] / metrics['MIP_Columns']
                    else:
                        metrics['Time_Per_Column'] = 0.0

                    # --- 追加計算 2: Objective Value / MIP Columns ---
                    if metrics['MIP_Columns'] > 0:
                        metrics['Obj_Per_Column'] = metrics['Objective_Value'] / metrics['MIP_Columns']
                    else:
                        metrics['Obj_Per_Column'] = 0.0

                    # --- 追加計算 3: Objective Value * MIP Time (積) ---
                    # 変更点: MIP_Columns ではなく MIP_Time を掛ける
                    metrics['Obj_Times_Time'] = metrics['Objective_Value'] * metrics['MIP_Time']
                    
                    data_list.append(metrics)
                    
    return pd.DataFrame(data_list), ordered_labels

# ==========================================
# グラフ描画関数
# ==========================================

def plot_grouped_bar_chart(df, metric, title, ylabel, output_path, ordered_labels):
    if df.empty: return
    
    # ピボットテーブル作成 (行: Week, 列: Method)
    # 該当メトリクスが全てNaNなどの場合はスキップされる可能性があるため確認
    if metric not in df.columns:
        print(f"[Warning] Metric '{metric}' not found in dataframe.")
        return

    pivot_df = df.pivot(index='Week', columns='Method', values=metric)
    
    # カラムの並び順をJSONでの定義順に強制
    existing_labels = [l for l in ordered_labels if l in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=existing_labels)
    
    # カラーパレット
    colors = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
    ]
    
    # プロット設定
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df.plot(kind='bar', width=0.8, ax=ax,
                  color=colors[:len(existing_labels)],
                  edgecolor='white', linewidth=0.5)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Week', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # 凡例をグラフの外側に配置
    ax.legend(title='Threshold', title_fontsize=12, fontsize=11, 
              loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0) 
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate plots for Experiment 2 (JSON config mode)")
    parser.add_argument("prefix", type=str, help="Prefix of the results directory (e.g. results_exp2)")
    parser.add_argument("config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--start_week", type=int, default=None, help="Start week")
    parser.add_argument("--end_week", type=int, default=None, help="End week")

    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    print(f"Loading metrics based on config: {args.config}")
    df, ordered_labels = load_metrics_from_json(args.prefix, args.config, args.start_week, args.end_week)
    
    if df.empty:
        print("No data found. Check your prefix and config file.")
        return

    # 保存先ディレクトリ
    output_dir = "plots_grouped_custom"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. 列数の比較 (MIP Columns) ---
    plot_grouped_bar_chart(
        df, 'MIP_Columns', 
        'MIP Columns by Threshold', 
        'Number of Columns', 
        os.path.join(output_dir, "compare_MIP_Columns.png"),
        ordered_labels
    )

    # --- 2. MIP計算時間の比較 (MIP Time) ---
    plot_grouped_bar_chart(
        df, 'MIP_Time', 
        'MIP Time by Threshold', 
        'Time (sec)', 
        os.path.join(output_dir, "compare_MIP_Time.png"),
        ordered_labels
    )
    
    # --- 3. 目的関数値の比較 (Objective Value) ---
    plot_grouped_bar_chart(
        df, 'Objective_Value', 
        'Objective Value by Threshold', 
        'Cost', 
        os.path.join(output_dir, "compare_Objective_Value.png"),
        ordered_labels
    )

    # --- 4. フィルタされた列数の比較 (Filtered Columns) ---
    plot_grouped_bar_chart(
        df, 'Filtered_Columns', 
        'Filtered Columns by Threshold', 
        'Number of Filtered Columns', 
        os.path.join(output_dir, "compare_Filtered_Columns.png"),
        ordered_labels
    )

    # --- 5. 全体計算時間の比較 (Total Time) ---
    plot_grouped_bar_chart(
        df, 'Total_Time', 
        'Total Execution Time by Threshold', 
        'Time (sec)', 
        os.path.join(output_dir, "compare_Total_Time.png"),
        ordered_labels
    )

    # --- 6. 列単価の比較 (MIP Time / MIP Columns) ---
    plot_grouped_bar_chart(
        df, 'Time_Per_Column', 
        'MIP Time per Column by Threshold', 
        'Time per Column (sec)', 
        os.path.join(output_dir, "compare_Time_Per_Column.png"),
        ordered_labels
    )

    # --- 7. Objective Value / MIP Column (割り算) ---
    plot_grouped_bar_chart(
        df, 'Obj_Per_Column', 
        'Objective Value per MIP Column by Threshold', 
        'Obj / Column', 
        os.path.join(output_dir, "compare_Obj_Per_Column.png"),
        ordered_labels
    )

    # --- 8. Objective Value * MIP Time (掛け算・積) ---
    # 変更点: タイトルとファイル名を変更
    plot_grouped_bar_chart(
        df, 'Obj_Times_Time', 
        r'Objective Value $\times$ MIP Time by Threshold', 
        r'Obj $\times$ Time', 
        os.path.join(output_dir, "compare_Obj_Times_Time.png"),
        ordered_labels
    )

    print(f"Processing complete. Plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
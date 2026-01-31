import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Data Parsing Functions (Optimized)
# ==========================================

def parse_metrics(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {}
    # 正規表現で必要な数値を抽出
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
            metrics[key] = float(match.group(1).replace(',', ''))
        else:
            metrics[key] = 0.0 # Default

    return metrics

def parse_folder_name(dirname):
    parts = dirname.split('_')
    if len(parts) < 4: return None
    if 'rc' not in parts: return None
    try:
        rc_idx = parts.index('rc')
    except ValueError: return None
    
    if rc_idx + 1 >= len(parts): return None
    rc_str = parts[rc_idx + 1]
    
    # Method判定
    if rc_idx + 2 < len(parts):
        method = parts[rc_idx + 2]
    else:
        method = 'std'
        
    # RC Value変換
    if rc_str == 'inf':
        rc_val = float('inf')
        rc_label = 'Inf'
    else:
        try:
            rc_val = float(rc_str)
            rc_label = str(int(rc_val)) if rc_val.is_integer() else str(rc_val)
        except: return None
            
    return rc_val, rc_label, method

def load_data(target_method):
    dirs = glob.glob("results_exp2_rc_*")
    data = []
    print(f"Loading data for method='{target_method}'...")
    
    for d in dirs:
        if not os.path.isdir(d): continue
        parsed = parse_folder_name(d)
        if not parsed: continue
        rc_val, rc_label, method = parsed
        
        if method != target_method: continue
        
        # Report search
        report_files = glob.glob(os.path.join(d, method, "report_wk*.txt"))
        if not report_files:
             report_files = glob.glob(os.path.join(d, "report_wk*.txt"))
             
        for f in report_files:
            match = re.search(r"report_wk(\d+)\.txt", os.path.basename(f))
            if match:
                wk = int(match.group(1))
                m = parse_metrics(f)
                if m:
                    m['Week'] = wk
                    m['RC_Value'] = rc_val
                    m['RC_Label'] = rc_label
                    data.append(m)
                    
    df = pd.DataFrame(data)
    if not df.empty:
        df.sort_values(by=['RC_Value', 'Week'], inplace=True)
    return df

# ==========================================
# Plotting Function (Grouped Bar Chart)
# ==========================================

def plot_grouped_bar_chart(df, metric, title, ylabel, output_path):
    if df.empty: return
    
    # ピボットテーブル作成 (行: Week, 列: RC_Label)
    # Infが最後に来るようにRC_Valueでソートしてからラベルを使う
    unique_rcs = sorted(df['RC_Value'].unique())
    rc_labels = []
    for rc in unique_rcs:
        label = df[df['RC_Value'] == rc]['RC_Label'].iloc[0]
        rc_labels.append(label)
        
    pivot_df = df.pivot(index='Week', columns='RC_Label', values=metric)
    
    # カラムの並び順を数値順(Inf最後)に強制
    pivot_df = pivot_df.reindex(columns=rc_labels)
    
    # ★変更点: generate_plots.py と同じカラーパレットを使用
    colors = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'
    ]
    
    # プロット設定
    # color引数にリストを渡すと、カラム（閾値）ごとに色が割り当てられます
    ax = pivot_df.plot(kind='bar', figsize=(15, 7), width=0.8, 
                       color=colors,
                       edgecolor='white', linewidth=0.5)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Week', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title='Threshold', title_fontsize=12, fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0) # X軸ラベル（週）を水平に
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method name (std, acc, pruning)")
    args = parser.parse_args()
    
    df = load_data(args.method)
    if df.empty:
        print("No data found.")
        return

    out_dir = f"plots_grouped_{args.method}"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # 1. 列数の比較 (MIP Columns)
    plot_grouped_bar_chart(
        df, 'MIP_Columns', 
        f'MIP Columns by RC Threshold ({args.method})', 
        'Number of Columns', 
        os.path.join(out_dir, f"compare_MIP_Columns_rc_{args.method}.png")
    )

    # 2. MIP計算時間の比較
    plot_grouped_bar_chart(
        df, 'MIP_Time', 
        f'MIP Time by RC Threshold ({args.method})', 
        'Time (sec)', 
        os.path.join(out_dir, f"compare_MIP_Time_rc_{args.method}.png")
    )
    
    # 3. 目的関数値の比較 (コストへの影響確認)
    plot_grouped_bar_chart(
        df, 'Objective_Value', 
        f'Objective Value by RC Threshold ({args.method})', 
        'Cost', 
        os.path.join(out_dir, f"compare_Objective_Value_rc_{args.method}.png")
    )

    print(f"All grouped plots saved to {out_dir}/")

if __name__ == "__main__":
    main()
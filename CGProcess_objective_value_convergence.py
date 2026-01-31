import matplotlib
# サーバー環境等でGUIがない場合のエラー回避
matplotlib.use('Agg') 
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import re
import numpy as np

def build_path(base_prefix, patience, method):
    """
    ディレクトリ構造のルールに従ってパスを生成する
    
    ルール:
    1. method == 'std' の場合:
       {base_prefix}_pat_{patience} / std
       
    2. method != 'std' (例: acc) の場合:
       {base_prefix}_pat_{patience}_{method} / {method}
    """
    # フォルダ名が浮動小数点表記か整数表記かで揺れる場合を考慮し文字列化
    p_val = str(patience)
    
    if method == 'std':
        dir_name = f"{base_prefix}_pat_{p_val}"
    else:
        dir_name = f"{base_prefix}_pat_{p_val}_{method}"
    
    path = os.path.join(dir_name, method)
    return path

def parse_and_plot(base_prefix, patience, method, week):
    # --- 1. ディレクトリパスとファイル名の構築 ---
    target_dir = build_path(base_prefix, patience, method)
    filename = os.path.join(target_dir, f"report_wk{week}.txt")
    
    if not os.path.exists(filename):
        print(f"[Skip] File not found: {filename}")
        return

    print(f"Processing: {method} (Week {week}) ...")

    # --- 2. データの読み込み ---
    iterations = []
    obj_values = []
    pool_hits = []   # プールヒット数
    graph_gen = []   # 新規生成数
    is_reading_history = False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if "Iteration History" in line:
                    is_reading_history = True
                    continue
                if not is_reading_history or not stripped_line:
                    continue
                if "Iter" in line or "-----" in line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 4:
                    try:
                        iter_val = int(parts[0].strip())
                        obj_str = parts[1].strip().replace(',', '')
                        obj_val = float(obj_str)
                        p_hits = int(parts[2].strip())
                        g_gen = int(parts[3].strip())
                        
                        iterations.append(iter_val)
                        obj_values.append(obj_val)
                        pool_hits.append(p_hits)
                        graph_gen.append(g_gen)
                    except ValueError:
                        break
    except Exception as e:
        print(f"  Error reading file: {e}")
        return
                    
    if not iterations:
        print("  Error: No iteration data found.")
        return

    # Numpy配列に変換
    iterations = np.array(iterations)
    obj_values = np.array(obj_values)
    pool_hits = np.array(pool_hits)
    graph_gen = np.array(graph_gen)
    
    total_len = len(iterations)

    # --- 3. プロット作成用関数 ---
    def create_plot(x_start, x_end, suffix):
        """
        指定されたX軸範囲(x_start, x_end)でグラフを作成し保存する関数
        """
        if x_start >= x_end:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # タイトル
        title_str = (f'Convergence Statistics ({suffix})\n'
                     f'Method: {method} / Week: {week} / Patience: {patience}')
        fig.suptitle(title_str, fontsize=16, fontweight='bold', y=0.95)

        # 表示範囲内のデータをマスク抽出
        mask = (iterations >= x_start) & (iterations <= x_end)
        if not np.any(mask): mask = slice(None)
        sliced_obj = obj_values[mask]
        
        # === 上段: 目的関数 ===
        ax1.plot(iterations, obj_values, marker='o', markersize=3, linestyle='-', 
                 color='#1f77b4', label='Objective Value', alpha=0.9)
        
        # 停滞区間の描画
        stagnation_found = False
        for i in range(len(obj_values) - 1):
            if not (iterations[i+1] < x_start or iterations[i] > x_end):
                if abs(obj_values[i] - obj_values[i+1]) < 1e-9:
                    ax1.plot([iterations[i], iterations[i+1]], 
                             [obj_values[i], obj_values[i+1]], 
                             color='red', linewidth=2.5, marker='o', markersize=3)
                    stagnation_found = True

        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        ax1.set_xlim(x_start, x_end)
        
        # Y軸自動調整
        if len(sliced_obj) > 0:
            y_min = sliced_obj.min()
            y_max = sliced_obj.max()
            margin = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.01
            if margin == 0: margin = 1.0
            ax1.set_ylim(y_min - margin, y_max + margin)

        handles1, labels1 = ax1.get_legend_handles_labels()
        if stagnation_found and 'Stagnation' not in labels1:
            handles1.append(mpatches.Patch(color='red', label='Stagnation'))
            labels1.append('Stagnation')
        ax1.legend(handles1, labels1, loc='upper right')

        # 最終値
        last_iter_in_range = iterations[mask][-1] if len(iterations[mask]) > 0 else None
        if last_iter_in_range is not None and last_iter_in_range == iterations[-1]:
             ax1.annotate(f'Final: {obj_values[-1]:,.2f}', 
                         xy=(iterations[-1], obj_values[-1]), 
                         xytext=(iterations[-1], obj_values[-1] + margin),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        # === 下段: 列追加数 ===
        ax2.bar(iterations, pool_hits, color='#3cb371', label='Pool Hits', alpha=0.8, width=1.0)
        ax2.bar(iterations, graph_gen, bottom=pool_hits, color='#ff7f0e', label='Graph Gen', alpha=0.9, width=1.0)

        ax2.set_ylabel('Columns Added', fontsize=12)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_xlim(x_start, x_end)
        ax2.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存
        outname = os.path.join(target_dir, f'convergence_{suffix}_wk{week}.png')
        plt.savefig(outname)
        plt.close(fig)
        print(f"  Saved: {outname}")

    # --- 画像生成の実行 ---
    first_iter = iterations[0]
    last_iter = iterations[-1]
    
    # 1. Overall
    create_plot(first_iter, last_iter, "Overall")
    
    # 2. Start (Iteration 15-50)
    start_offset = 15
    end_offset = 50
    if total_len > start_offset:
        s_val = iterations[min(start_offset, total_len - 1)]
        e_val = iterations[min(end_offset, total_len - 1)]
        if s_val < e_val:
            create_plot(s_val, e_val, "Start")
    
    # 3. End (Last 30% or 20 iter)
    num_display = max(20, int(total_len * 0.30))
    start_idx_end = max(0, total_len - num_display)
    s_val_end = iterations[start_idx_end]
    create_plot(s_val_end, last_iter, "End")


def scan_and_process(base_prefix, patience, week):
    print(f"Scanning for methods in {base_prefix} with patience={patience}...")
    
    found_methods = set()
    
    # 1. 'std' の確認 (results_exp1_pat_100/std)
    std_dir = f"{base_prefix}_pat_{patience}"
    std_path = os.path.join(std_dir, "std")
    if os.path.isdir(std_path):
        found_methods.add("std")
        
    # 2. その他の手法の確認 (results_exp1_pat_100_*)
    # パターン: results_exp1_pat_100_{method}
    search_pattern = f"{base_prefix}_pat_{patience}_*"
    candidates = glob.glob(search_pattern)
    
    for c_dir in candidates:
        if not os.path.isdir(c_dir): continue
        
        # ディレクトリ名から手法名を抽出
        # 例: results_exp1_pat_100_acc -> acc
        dirname = os.path.basename(c_dir)
        # プレフィックス部分を除去
        prefix_part = f"{base_prefix}_pat_{patience}_"
        
        if dirname.startswith(prefix_part):
            method_name = dirname[len(prefix_part):]
            if method_name:
                # 念のため内部にその手法名のフォルダがあるか確認
                if os.path.isdir(os.path.join(c_dir, method_name)):
                    found_methods.add(method_name)

    # 見つかった手法に対して実行
    sorted_methods = sorted(list(found_methods))
    if not sorted_methods:
        print("No methods found. Please check the directory names.")
        return

    print(f"Found methods: {sorted_methods}")
    print("-" * 40)
    
    for method in sorted_methods:
        parse_and_plot(base_prefix, patience, method, week)
    
    print("-" * 40)
    print("All processing complete.")

# --- メイン処理 ---
if __name__ == "__main__":
    # 設定値 (固定)
    EXP_PREFIX = "results_exp1"
    PATIENCE = 100
    WEEK = 15
    
    scan_and_process(EXP_PREFIX, PATIENCE, WEEK)
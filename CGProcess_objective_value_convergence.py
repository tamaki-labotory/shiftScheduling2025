from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import sys
import numpy as np

def parse_and_plot(emp, method, week):
    # --- 1. ファイル名の構築 ---
    if method == "default":
        filename = f"report_wk{week}.txt"
    else:
        filename = f"results_{emp}emp/{method}/report_wk{week}.txt"
    
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return

    print(f"読み込み中: {filename} ...")

    # --- 2. データの読み込み ---
    iterations = []
    obj_values = []
    pool_hits = []   # プールヒット数
    graph_gen = []   # 新規生成数
    is_reading_history = False
    
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
                    
    if not iterations:
        print("エラー: データが見つかりませんでした。")
        return

    # Numpy配列に変換
    iterations = np.array(iterations)
    obj_values = np.array(obj_values)
    pool_hits = np.array(pool_hits)
    graph_gen = np.array(graph_gen)

    # --- 3. プロット作成用関数 ---
    def create_plot(x_start, x_end, suffix):
        """
        指定されたX軸範囲(x_start, x_end)でグラフを作成し保存する関数
        """
        # グラフ領域の設定（上下2段）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        fig.suptitle(f'Convergence & Column Generation Statistics ({suffix})\nMethod: {method} / Week: {week}', 
                     fontsize=16, fontweight='bold', y=0.95)

        # 表示範囲内のデータをマスク抽出（Y軸スケール調整用）
        mask = (iterations >= x_start) & (iterations <= x_end)
        if not np.any(mask): mask = slice(None)
        sliced_obj = obj_values[mask]
        
        # === 上段: 目的関数 ===
        ax1.plot(iterations, obj_values, marker='o', markersize=3, linestyle='-', 
                 color='#1f77b4', label='Objective Value', alpha=0.9)
        
        # 停滞区間の描画
        stagnation_found = False
        for i in range(len(obj_values) - 1):
            # 線分が表示範囲に少しかかっていれば描画
            if not (iterations[i+1] < x_start or iterations[i] > x_end):
                if abs(obj_values[i] - obj_values[i+1]) < 1e-9:
                    ax1.plot([iterations[i], iterations[i+1]], 
                             [obj_values[i], obj_values[i+1]], 
                             color='red', linewidth=2.5, marker='o', markersize=3)
                    stagnation_found = True

        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # 軸範囲の設定
        ax1.set_xlim(x_start, x_end)
        
        # Y軸の自動調整（表示範囲内の最大・最小値に基づく）
        if len(sliced_obj) > 0:
            y_min = sliced_obj.min()
            y_max = sliced_obj.max()
            margin = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.01
            if margin == 0: margin = 1.0
            ax1.set_ylim(y_min - margin, y_max + margin)

        # 凡例
        handles1, labels1 = ax1.get_legend_handles_labels()
        if stagnation_found and 'Stagnation' not in labels1:
            handles1.append(mpatches.Patch(color='red', label='Stagnation'))
            labels1.append('Stagnation')
        ax1.legend(handles1, labels1, loc='upper right')

        # 最終値（範囲内なら表示）
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
        
        outname = f'convergence_{suffix}_{method}_wk{week}.png'
        plt.savefig(outname)
        plt.close(fig)
        print(f"画像を保存しました: {outname}")

    # --- 画像生成の実行 ---
    total_iters = iterations[-1]
    
    # 1. Overall（全体）
    create_plot(iterations[0], total_iters, "Overall")
    
    # 2. Start（序盤）: 全体の10%（5~15%の範囲のご要望に対し、中間の10%を採用）
    # ※最低でも20イテレーションは確保
    end_idx_start = max(20, int(len(iterations) * 0.15))
    # 範囲外参照を防ぐ
    end_idx_start = min(end_idx_start, len(iterations)-1)
    
    create_plot(iterations[15], iterations[50], "Start")
    
    # 3. End（終盤）: 最後の15%
    start_idx_end = min(len(iterations) - 20, int(len(iterations) * 0.70))
    start_idx_end = max(0, start_idx_end)
    
    create_plot(iterations[start_idx_end], total_iters, "End")

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RMP Convergence Plotter (3 Views)')
    parser.add_argument('emp', type=str, help='Emploee Number')
    parser.add_argument('method', type=str, help='Method name')
    parser.add_argument('week', type=str, help='Week number')
    
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    parse_and_plot(args.emp, args.method, args.week)
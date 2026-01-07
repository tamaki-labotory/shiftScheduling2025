from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import sys

def parse_and_plot(emp,method, week):
    # --- 1. ファイル名の構築 ---
    # ここでファイル名の命名規則を定義します
    # 例: method="new_algo", week="20" -> "report_new_algo_wk20.txt"
    # ※もし手法名がファイル名に含まれない場合はここを調整してください
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
            if len(parts) >= 2:
                try:
                    iter_val = int(parts[0].strip())
                    obj_str = parts[1].strip().replace(',', '')
                    obj_val = float(obj_str)
                    iterations.append(iter_val)
                    obj_values.append(obj_val)
                except ValueError:
                    break
                    
    if not iterations:
        print("エラー: データが見つかりませんでした。")
        return

    # --- 3. グラフの描画 ---
    fig, ax = plt.subplots(figsize=(12, 8)) # axオブジェクトを使用するのが一般的
    
    # ベースの線（青色）
    ax.plot(iterations, obj_values, marker='o', markersize=3, linestyle='-', 
                color='#1f77b4', label='Improvement', alpha=0.6)
    
    # 停滞区間（赤色）
    for i in range(len(obj_values) - 1):
        if abs(obj_values[i] - obj_values[i+1]) < 1e-9:
            ax.plot([iterations[i], iterations[i+1]], 
                        [obj_values[i], obj_values[i+1]], 
                        color='red', linewidth=2.5, marker='o', markersize=3)

    # --- 情報の表示 ---
    ax.set_title(f'The Convergence of The RMP Objective Value\nMethod: {method} / Week: {week}', fontsize=16, fontweight='bold')
    

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # --- 【修正点】軸のフォーマット設定 ---
    # Y軸: カンマ区切りにするフォーマッターを適用
    # これにより、ズームしても自動的に再計算されて正しい値が表示されます
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # X軸: 整数のみを表示するように強制（ズーム時に0.5などの小数が出るのを防ぐ場合）
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 凡例
    blue_patch = mpatches.Patch(color='#1f77b4', label='Improvement')
    red_patch = mpatches.Patch(color='red', label='Stagnation')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right')

    # 最終値の注釈
    last_iter = iterations[-1]
    last_obj = obj_values[-1]
    ax.annotate(f'Final: {last_obj:,.2f}', 
                    xy=(last_iter, last_obj), 
                    xytext=(last_iter - (last_iter*0.2), last_obj + (obj_values[0]*0.05)),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()
        
# --- コマンドライン引数の処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RMP Convergence Plotter')
    
    # 引数の定義
    parser.add_argument('emp', type=str, help='Emploee Number (e.g., 10)')
    parser.add_argument('method', type=str, help='Name of the method (e.g., proposed, baseline)')
    parser.add_argument('week', type=str, help='Week number (e.g., 20)')
    
    # 引数がない場合にヘルプを表示する処理
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    # 実行
    parse_and_plot(args.emp,args.method, args.week)
import pandas as pd
import matplotlib.pyplot as plt

# ファイルの読み込み
file_path = 'results_5emp/acc/report_wk20.txt'

data = []
is_parsing = False

with open(file_path, 'r') as f:
    for line in f:
        # セクションの開始を検知
        if "Iteration History" in line:
            is_parsing = True
            continue
        
        if is_parsing:
            # 区切り線やヘッダーをスキップ
            if "----" in line or "Iter" in line or "RMP Obj Value" in line:
                continue
            
            # パイプが含まれない行、または空行になったら終了とみなす
            if "|" not in line or not line.strip():
                if len(data) > 0: # データが既に取れていれば終了
                    break
                continue 

            # 行の解析 (例: "1     | 90,016,000.00   | 0          | 5")
            parts = line.strip().split('|')
            
            if len(parts) >= 4:
                try:
                    iteration = int(parts[0].strip())
                    pool_hits = int(parts[2].strip())
                    graph_gen = int(parts[3].strip())
                    
                    data.append({
                        "Iteration": iteration,
                        "Pool Hits": pool_hits,
                        "Graph Gen": graph_gen
                    })
                except ValueError:
                    continue

# DataFrame化
df = pd.DataFrame(data)

# グラフの作成
plt.figure(figsize=(12, 6))

# 線グラフで描画（データ点が多い場合は線が見やすいです）
plt.plot(df['Iteration'], df['Pool Hits'], label='Pool Hits (Reused)', color='blue', linewidth=1.5)
plt.plot(df['Iteration'], df['Graph Gen'], label='Graph Gen (Newly Created)', color='orange', linewidth=1.5)

# グラフの装飾
plt.title('Column Generation History: Pool Hits vs Graph Gen')
plt.xlabel('Iteration')
plt.ylabel('Number of Columns')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 表示
plt.tight_layout()
plt.savefig('column_generation_history.png')
plt.show()
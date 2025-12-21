import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    # ---------------------------------------------------------
    # 1. コマンドライン引数の設定
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Visualize column similarity with fixed directory structure."
    )
    # 必須引数: 従業員数
    parser.add_argument("--emp", type=int, required=True, 
                        help="Number of employees (e.g., 20)")
    # 必須引数: 手法名 (ファイル名のサフィックスに使用)
    parser.add_argument("--method", type=str, required=True, 
                        help="Method name suffix for files (e.g., std, aging)")
    # オプション: ルートディレクトリ
    parser.add_argument("--root_dir", type=str, default=".", 
                        help="Root directory to search (default: .)")
    # オプション: サンプリング数
    parser.add_argument("--samples", type=int, default=3000, 
                        help="Max samples for visualization (default: 3000)")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. パスとファイル名の構築 (修正箇所)
    # ---------------------------------------------------------
    # ディレクトリ名は 'std' で固定
    # 例: schedule_plots_20emp_exact_std_pool_aging
    dir_name = f"schedule_plots_{args.emp}emp_exact_std_pool_aging"
    
    # ファイル名は手法名によって可変
    # 例: pool_wk*_std.csv や pool_wk*_aging.csv
    file_pattern = f"pool_wk*_{args.method}.csv"
    
    # 検索パスの結合
    search_path = os.path.join(args.root_dir, dir_name, file_pattern)
    
    print(f"==================================================")
    print(f" [設定]")
    print(f"  従業員数 : {args.emp}")
    print(f"  手法名   : {args.method}")
    print(f"  探索パス : {search_path}")
    print(f"==================================================")

    files = glob.glob(search_path)
    
    if not files:
        print(f"エラー: ファイルが見つかりませんでした。")
        print(f"ディレクトリ '{dir_name}' が存在するか、")
        print(f"その中に 'pool_wk..._{args.method}.csv' があるか確認してください。")
        return

    print(f"-> {len(files)} 個のファイルが見つかりました。解析を開始します...")

    # ---------------------------------------------------------
    # 3. データの読み込み
    # ---------------------------------------------------------
    data_matrix = []
    labels = []
    
    for f_path in files:
        f_name = os.path.basename(f_path)
        # ラベル作成 (pool_wk1_std.csv -> wk1)
        parts = f_name.split('_')
        if len(parts) > 1:
            label = parts[1] # 'wk1'
        else:
            label = f_name
            
        try:
            df = pd.read_csv(f_path)
            if 'schedule_pattern' in df.columns:
                for pattern in df['schedule_pattern']:
                    # 文字列を数値リストに変換
                    vec = [int(c) for c in pattern]
                    data_matrix.append(vec)
                    labels.append(label)
        except Exception as e:
            print(f"警告: {f_name} の読み込みエラー ({e})")

    if not data_matrix:
        print("エラー: データが抽出できませんでした。")
        return

    X = np.array(data_matrix)
    y = np.array(labels)
    
    # ---------------------------------------------------------
    # 4. サンプリングと可視化
    # ---------------------------------------------------------
    if len(X) > args.samples:
        print(f"-> データ数({len(X)})が多いため、{args.samples}件にサンプリングします。")
        indices = np.random.choice(len(X), args.samples, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y

    print("-> 次元圧縮を実行中 (PCA & t-SNE)...")

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    
    # t-SNE
    perp = min(30, len(X_sub) - 1) if len(X_sub) > 1 else 1
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_sub)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    unique_labels = sorted(list(set(y_sub)))
    
    # PCA Plot
    for label in unique_labels:
        mask = (y_sub == label)
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6, s=20)
    axes[0].set_title(f'PCA Projection ({args.method}, {args.emp}emp)', fontsize=14)
    axes[0].legend(title="Week")
    axes[0].grid(True, alpha=0.3)

    # t-SNE Plot
    for label in unique_labels:
        mask = (y_sub == label)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.6, s=20)
    axes[1].set_title(f't-SNE Projection ({args.method}, {args.emp}emp)', fontsize=14)
    axes[1].legend(title="Week")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Similarity of Generated Columns (Emp: {args.emp}, Method: {args.method})\nDir: {dir_name}', fontsize=16)
    plt.tight_layout()
    
    output_filename = f"similarity_{args.emp}emp_{args.method}.png"
    plt.savefig(output_filename)
    print(f"-> 完了! グラフを保存しました: {output_filename}")

if __name__ == "__main__":
    main()
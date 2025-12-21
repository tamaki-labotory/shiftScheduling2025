import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

def analyze_directory_overlap(emp_num, method_name, root_dir='.'):
    """
    コマンドライン引数で指定された構成に基づき、シフトパターンの重複状況を解析します。
    
    想定ディレクトリ構成:
      [root_dir]/schedule_plots_{emp_num}emp_{method_name}/pool_wk*.csv
    """
    
    # 1. パスの構築
    # ディレクトリ名の組み立て (例: schedule_plots_20emp_exact_std_pool_aging)
    target_subdir = f"schedule_plots_{emp_num}emp_exact_std_pool_aging"
    
    # ファイル検索パターン (末尾が _std.csv だけでなく汎用的に pool_wk*.csv を拾うように緩和)
    file_pattern = f"pool_wk*_{method_name}.csv"
    
    search_path = os.path.join(root_dir, target_subdir, file_pattern)
    print(f"--- 解析開始 ---")
    print(f"対象ディレクトリ: {target_subdir}")
    print(f"検索パターン: {search_path}")
    
    # ファイルの取得
    target_files = glob.glob(search_path)
    
    if not target_files:
        print(f"エラー: 指定されたパスにファイルが見つかりません。")
        print(f"  -> {search_path}")
        print("ディレクトリ名や従業員数、手法名が正しいか確認してください。")
        return

    print(f"発見ファイル数: {len(target_files)}")
    # ファイル名順にソートして表示（wk1, wk2, ... の順序を整えるため簡易的にソート）
    target_files.sort()
    
    # 2. パターン集計
    pattern_file_counts = Counter()
    
    for f in target_files:
        try:
            df = pd.read_csv(f)
            if 'schedule_pattern' in df.columns:
                unique_patterns = set(df['schedule_pattern'])
                pattern_file_counts.update(unique_patterns)
            else:
                print(f"警告: {os.path.basename(f)} に 'schedule_pattern' 列がありません。")
        except Exception as e:
            print(f"読み込みエラー ({os.path.basename(f)}): {e}")

    if not pattern_file_counts:
        print("集計可能なデータがありませんでした。")
        return

    # 出力ファイル名のプレフィックス作成
    output_prefix = f"{emp_num}emp_{method_name}"

    # --- グラフ1: 重複分布 (Distribution) ---
    distribution = Counter(pattern_file_counts.values())
    x_vals = sorted(distribution.keys())
    y_vals = [distribution[x] for x in x_vals]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_vals, y_vals, color='skyblue', edgecolor='black', width=0.6)
    
    plt.xlabel('Number of Files (Overlap Count)', fontsize=12)
    plt.ylabel('Number of Unique Patterns', fontsize=12)
    plt.title(f'Distribution of Schedule Pattern Overlaps\n({output_prefix})', fontsize=14)
    plt.xticks(x_vals)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_dist = f'{output_prefix}_overlap_distribution.png'
    plt.savefig(output_dist)
    plt.close()
    print(f"分布グラフを保存しました: {output_dist}")
    
    # --- グラフ2: ヒートマップ (Heatmap) ---
    # 重複が2回以上のものだけ抽出
    duplicate_info = [(p, count) for p, count in pattern_file_counts.items() if count > 1]
    
    if duplicate_info:
        # ソート: 重複回数(降順) -> 勤務時間量(降順)
        duplicate_info.sort(key=lambda x: (-x[1], -x[0].count('1')))
        
        patterns = [x[0] for x in duplicate_info]
        counts = [x[1] for x in duplicate_info]
        
        # 文字列パターンを数値リストに変換
        matrix = []
        for p in patterns:
            # 文字列中の数字以外の文字が含まれる場合のエラーハンドリングが必要ならここに追加
            row = [int(c) for c in p]
            matrix.append(row)
        matrix = np.array(matrix)
        
        # 行数に応じて高さを調整
        plt.figure(figsize=(15, max(5, len(patterns) * 0.5))) 
        plt.imshow(matrix, aspect='auto', cmap='Blues', interpolation='nearest')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Pattern Index', fontsize=12)
        plt.title(f'Duplicate Schedule Patterns (Count > 1)\n{output_prefix} / Total Duplicates: {len(patterns)}', fontsize=14)
        
        cbar = plt.colorbar(pad=0.02)
        cbar.set_label('Shift Status (0=Off, 1=On)', rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])
        
        # Y軸ラベル設定
        y_labels = [f"#{i+1} (Count: {c})" for i, c in enumerate(counts)]
        plt.yticks(range(len(patterns)), labels=y_labels, fontsize=8)
        
        plt.tight_layout()
        output_heat = f'{output_prefix}_duplicate_heatmap.png'
        plt.savefig(output_heat)
        plt.close()
        print(f"ヒートマップを保存しました: {output_heat}")
    else:
        print("重複回数が1回より多いパターンは存在しませんでした（ヒートマップ作成スキップ）。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze schedule pattern overlaps based on directory parameters.')
    
    # 引数の定義
    parser.add_argument('--employees', '-e', type=int, required=True, help='Number of employees (e.g., 20)')
    parser.add_argument('--method', '-m', type=str, required=True, help='Method suffix name (e.g., exact_std_pool_aging)')
    parser.add_argument('--root', '-r', type=str, default='.', help='Root directory path (default: current directory)')

    args = parser.parse_args()
    
    # 実行
    analyze_directory_overlap(args.employees, args.method, args.root)
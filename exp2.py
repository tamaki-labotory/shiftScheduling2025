import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import shutil

# 既存モジュールのインポート
from problem import ShiftProblemData
from solver_cg import ColumnGenerationSolver
from visualization import BenchmarkReporter, MIPConvergencePlotter

def run_sequential_threshold_experiment():
    # === パス設定 ===
    base_dir = "results_5emp"
    config_path = os.path.join(base_dir, "problem_config.json")
    
    # 実験結果のルートディレクトリ
    exp_root_dir = os.path.join(base_dir, "experiment_sequential")
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)

    print(f"=== Sequential Threshold Experiment based on {config_path} ===")

    # === 1. 設定ファイルから対象となる週（期間）を特定 ===
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # demand_history に保存されているキー（週番号-1 の値）を取得
    # キーは文字列として保存されているため int に変換してソート
    history_keys = sorted([int(k) for k in config_data.get('demand_history', {}).keys()])
    
    if not history_keys:
        print("No demand history found in config file.")
        return

    print(f"Found historical data for {len(history_keys)} weeks: {history_keys}")

    # 問題クラスの初期化
    prob = ShiftProblemData(config_path=config_path)
    
    # Standard CG なので use_pool=False (週をまたいで列を引き継がない)
    solver = ColumnGenerationSolver(prob, use_pool=False)
    
    # 全実験の結果を保持するリスト
    all_results = []

    # 検証する閾値のリスト
    thresholds = [0, 1, 100, 10000, 1e10]

    # === 2. 週ごとのループ (逐次実行) ===
    for period_idx in history_keys:
        week_num = period_idx + 1
        print(f"\n\n{'='*60}")
        print(f" Processing Week {week_num} (Period {period_idx})")
        print(f"{'='*60}")

        # この週の需要データをロード（履歴から復元）
        prob.generate_new_demand(period=period_idx)
        
        # 週ごとの出力フォルダ作成
        week_dir = os.path.join(exp_root_dir, f"Week_{week_num}")
        if not os.path.exists(week_dir):
            os.makedirs(week_dir)

        week_results = []

        # === 3. 閾値ごとのループ ===
        for th in thresholds:
            th_label = "ALL" if th >= 1e9 else str(th)
            print(f"\n  > Threshold: {th_label}")
            
            # ソルバーのリセット（重要：Standard CGなので前の計算の影響を消す）
            solver.reset_stats()
            solver.reset_for_new_period()
            
            # ソルバー実行
            obj_val, total_time, stats, final_schedule = solver.solve(
                max_iter=5000,
                mip_rc_threshold=th,
                mip_gap=0.0,
                time_limit=3000
            )
            
            # --- レポート保存 ---
            base_filename = f"report_wk{week_num}_th{th_label}"
            report_path = os.path.join(week_dir, f"{base_filename}.txt")
            BenchmarkReporter.save_analysis_report(
                report_path, week_num, solver, prob, obj_val, total_time, final_schedule
            )

            # --- MIP収束グラフ ---
            if 'mip_trajectory' in stats and stats['mip_trajectory']:
                mip_plot_path = os.path.join(week_dir, f"mip_wk{week_num}_th{th_label}.png")
                MIPConvergencePlotter.plot_convergence(
                    stats['mip_trajectory'], 
                    f"MIP Convergence Wk{week_num} (Th: {th_label})", 
                    mip_plot_path
                )

            # --- 結果収集 ---
            ip_time = stats.get('time_mip', 0)
            mip_cols = stats.get('mip_total_columns', 0)
            
            res = {
                'Week': week_num,
                'Threshold': th,
                'Label': th_label,
                'ObjVal': obj_val,
                'Time_Total': total_time,
                'Time_IP': ip_time,
                'Cols_MIP': mip_cols
            }
            week_results.append(res)
            all_results.append(res)
            
            print(f"    -> IP Time: {ip_time:.2f}s | MIP Cols: {mip_cols} | Obj: {obj_val:.2f}")

        # === 週ごとのグラフ作成 ===
        df_week = pd.DataFrame(week_results)
        create_week_summary_plots(df_week, week_dir, week_num)
        
        # 週ごとのCSV保存
        df_week.to_csv(os.path.join(week_dir, f"summary_wk{week_num}.csv"), index=False)

    # === 4. 全体の集計保存 ===
    df_all = pd.DataFrame(all_results)
    master_csv = os.path.join(exp_root_dir, "summary_all_weeks.csv")
    df_all.to_csv(master_csv, index=False)
    print(f"\nAll sequential experiments completed. Master summary saved to: {master_csv}")

def create_week_summary_plots(df, output_dir, week_num):
    """特定の週における閾値の影響をグラフ化"""
    x = df['Label'].astype(str)
    
    # IP計算時間 vs 列数
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Reduced Cost Threshold')
    ax1.set_ylabel('IP Execution Time (s)', color=color1)
    ax1.plot(x, df['Time_IP'], marker='o', color=color1, label='IP Time', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Columns in MIP', color=color2)
    ax2.plot(x, df['Cols_MIP'], marker='s', linestyle='--', color=color2, label='MIP Columns')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'Week {week_num}: IP Time vs Column Count by Threshold')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_wk{week_num}_time_cols.png"))
    plt.close()

    # 目的関数値
    plt.figure(figsize=(10, 6))
    plt.plot(x, df['ObjVal'], marker='D', color='tab:green', linewidth=2)
    plt.xlabel('Reduced Cost Threshold')
    plt.ylabel('Objective Value')
    plt.title(f'Week {week_num}: Solution Quality vs Threshold')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_wk{week_num}_obj.png"))
    plt.close()

if __name__ == "__main__":
    run_sequential_threshold_experiment()
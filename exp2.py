import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import shutil
import numpy as np

# 既存モジュールのインポート
from problem import ShiftProblemData
from solver_cg import ColumnGenerationSolver
from solver_cg_pruning import ColumnGenerationSolverWithPruning
from visualization import BenchmarkReporter, MIPConvergencePlotter

def run_comparative_threshold_experiment():
    # === パス設定 ===
    base_dir = "results_5emp"
    config_path = os.path.join(base_dir, "problem_config.json")
    
    # 実験結果のルートディレクトリ
    exp_root_dir = os.path.join(base_dir, "experiment_comparison_methods")
    if not os.path.exists(exp_root_dir):
        os.makedirs(exp_root_dir)

    print(f"=== Comparative Threshold Experiment (Std, Acc, Pruning) ===")

    # === 1. 設定ファイルから対象となる週（期間）を特定 ===
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # 履歴データのキー取得
    history_keys = sorted([int(k) for k in config_data.get('demand_history', {}).keys()])
    if not history_keys:
        print("No demand history found.")
        return
    
    # --- 【変更点1】実験期間を15週までに拡大 ---
    target_periods = [p for p in history_keys if (p + 1) <= 15]
    print(f"Target Periods (Week 1=Period 0): {target_periods}")

    # === 2. 実験条件の設定 ===
    
    # 比較する手法の定義
    methods_config = {
        'Standard': {
            'class': ColumnGenerationSolver,
            'kwargs': {'use_pool': False},
            'color': 'tab:red',
            'marker': 'o'
        },
        'Accumulation': {
            'class': ColumnGenerationSolver,
            'kwargs': {'use_pool': True},
            'color': 'tab:green',
            'marker': 's'
        },
        'Pruning': {
            'class': ColumnGenerationSolverWithPruning,
            'kwargs': {'use_pool': True, 'pool_cleanup_threshold': 2000},
            'color': 'tab:blue',
            'marker': '^'
        }
    }

    # 検証する閾値のリスト
    thresholds = [1e100]

    # 全結果保持用
    all_results = []

    # 問題インスタンスの初期化
    prob = ShiftProblemData(config_path=config_path)

    # === 3. 実験実行ループ ===
    for method_name, m_conf in methods_config.items():
        for th in thresholds:
            th_label = "ALL" if th >= 1e9 else str(int(th))
            run_id = f"{method_name}_Th{th_label}"
            
            print(f"\n{'#'*60}")
            print(f" Starting Sequence: {run_id}")
            print(f"{'#'*60}")

            # ソルバーの初期化
            SolverClass = m_conf['class']
            solver = SolverClass(prob, **m_conf['kwargs'])

            # 出力ディレクトリ
            run_dir = os.path.join(exp_root_dir, method_name, f"Th_{th_label}")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            
            # ヒストグラム保存用フォルダ（散らからないようにまとめる場合）
            hist_dir = os.path.join(run_dir, "histograms")
            if not os.path.exists(hist_dir):
                os.makedirs(hist_dir)

            # --- 時系列ループ ---
            for period_idx in target_periods:
                week_num = period_idx + 1
                
                # 需要の更新
                prob.generate_new_demand(period=period_idx)
                
                # ソルバーの状態更新
                solver.reset_stats()
                solver.reset_for_new_period()

                print(f"  -> Week {week_num}...", end=" ", flush=True)

                # 計算実行
                obj_val, total_time, stats, final_schedule = solver.solve(
                    max_iter=3000,
                    mip_rc_threshold=th,
                    mip_gap=0.001,
                    time_limit=1200
                )

                # 結果データ記録
                ip_time = stats.get('time_mip', 0)
                rmp_time = stats.get('time_rmp', 0)
                pool_size = stats.get('pool_size', len(solver.pool))
                mip_cols = stats.get('mip_total_columns', 0)
                
                res = {
                    'Method': method_name,
                    'Threshold': th,
                    'Threshold_Label': th_label,
                    'Week': week_num,
                    'ObjVal': obj_val,
                    'Time_Total': total_time,
                    'Time_IP': ip_time,
                    'Time_RMP': rmp_time,
                    'Cols_MIP': mip_cols,
                    'Pool_Size': pool_size
                }
                all_results.append(res)
                print(f"Done. (Total: {total_time:.1f}s, IP: {ip_time:.1f}s, Cols: {mip_cols})")

                # --- 【変更点2】被約費用のヒストグラム作成 ---
                # プール内の列が存在する場合のみ作成
                if solver.pool:
                    try:
                        # reduced_cost属性を持つ列のみ対象（念のため）
                        rc_values = [col.reduced_cost for col in solver.pool if hasattr(col, 'reduced_cost')]
                        
                        if rc_values:
                            plt.figure(figsize=(8, 5))
                            # 視認性向上のためbinsを多めにし、Y軸をログスケールにする
                            plt.hist(rc_values, bins=50, color='skyblue', edgecolor='black', log=True)
                            
                            plt.title(f"Reduced Cost Distribution (Wk{week_num}, {method_name})")
                            plt.xlabel("Reduced Cost")
                            plt.ylabel("Frequency (Log Scale)")
                            plt.grid(True, linestyle='--', alpha=0.5)
                            
                            # 閾値のラインを赤線で引く（視覚的な目安）
                            plt.axvline(x=th, color='red', linestyle='dashed', linewidth=1, label=f'Threshold: {th}')
                            plt.legend()

                            # PDFとして保存
                            hist_filename = f"hist_rc_wk{week_num}.pdf"
                            plt.savefig(os.path.join(hist_dir, hist_filename))
                            plt.close() # メモリ解放
                    except Exception as e:
                        print(f"   [Warning] Histogram generation failed: {e}")

                # レポート保存
                report_path = os.path.join(run_dir, f"report_wk{week_num}.txt")
                BenchmarkReporter.save_analysis_report(
                    report_path, week_num, solver, prob, obj_val, total_time, final_schedule
                )
                
                # 収束グラフ
                if 'mip_trajectory' in stats and stats['mip_trajectory']:
                    mip_plot_path = os.path.join(run_dir, f"mip_conv_wk{week_num}.png")
                    MIPConvergencePlotter.plot_convergence(
                        stats['mip_trajectory'], 
                        f"{method_name} Wk{week_num} (Th:{th_label})", 
                        mip_plot_path
                    )

    # === 4. 集計と可視化 ===
    print("\nProcessing results and generating plots...")
    df_all = pd.DataFrame(all_results)
    master_csv = os.path.join(exp_root_dir, "summary_master.csv")
    df_all.to_csv(master_csv, index=False)
    print(f"Master CSV saved to: {master_csv}")

    create_comparison_plots(df_all, exp_root_dir, methods_config)

def create_comparison_plots(df, output_dir, methods_config):
    """
    集計データからグラフを作成する（変更なし）
    """
    unique_weeks = sorted(df['Week'].unique())
    plots_dir = os.path.join(output_dir, "plots_summary")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for wk in unique_weeks:
        df_wk = df[df['Week'] == wk].copy()
        
        # Plot 1: IP Time vs Threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        for method_name, m_conf in methods_config.items():
            df_m = df_wk[df_wk['Method'] == method_name].sort_values('Threshold')
            if df_m.empty: continue
            x_labels = df_m['Threshold_Label'].astype(str)
            ax.plot(x_labels, df_m['Time_IP'], marker=m_conf['marker'], color=m_conf['color'], label=method_name, linewidth=2)

        ax.set_xlabel('Reduced Cost Threshold')
        ax.set_ylabel('IP Execution Time (s)')
        ax.set_title(f'Week {wk}: IP Time vs Threshold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"wk{wk}_time_comparison.png"))
        plt.close()

        # Plot 2: MIP Columns vs Threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        for method_name, m_conf in methods_config.items():
            df_m = df_wk[df_wk['Method'] == method_name].sort_values('Threshold')
            if df_m.empty: continue
            x_labels = df_m['Threshold_Label'].astype(str)
            ax.plot(x_labels, df_m['Cols_MIP'], marker=m_conf['marker'], color=m_conf['color'], label=method_name, linewidth=2, linestyle='--')

        ax.set_xlabel('Reduced Cost Threshold')
        ax.set_ylabel('Number of Columns in MIP')
        ax.set_title(f'Week {wk}: MIP Columns vs Threshold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"wk{wk}_cols_comparison.png"))
        plt.close()
        
        # Plot 3: Total Time vs Threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        for method_name, m_conf in methods_config.items():
            df_m = df_wk[df_wk['Method'] == method_name].sort_values('Threshold')
            if df_m.empty: continue
            x_labels = df_m['Threshold_Label'].astype(str)
            ax.plot(x_labels, df_m['Time_Total'], marker=m_conf['marker'], color=m_conf['color'], label=method_name, linewidth=2)

        ax.set_xlabel('Reduced Cost Threshold')
        ax.set_ylabel('Total Execution Time (s)')
        ax.set_title(f'Week {wk}: Total Time vs Threshold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"wk{wk}_total_time_comparison.png"))
        plt.close()

    print(f"All summary plots saved to: {plots_dir}")

if __name__ == "__main__":
    run_comparative_threshold_experiment()
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# モジュール群のインポート
from problem import ShiftProblemData
from solver_exact import ExactMIPSolver
from solver_cg import ColumnGenerationSolver
from solver_cg_aging import ColumnGenerationSolverWithAging
from solver_cg_smart import ColumnGenerationSolverSmart
from visualization import ScheduleVisualizer, BenchmarkReporter, ComparisonPlotter

# === ソルバー設定レジストリ ===
# ここに新しい手法を追加すれば、自動的に引数として選択可能になります
SOLVER_CONFIG = {
    'exact': {
        'class': ExactMIPSolver,
        'label': 'Exact MIP',
        'color': 'black',
        'marker': 'x',
        'needs_history': False,
        'kwargs': {}
    },
    'std': {
        'class': ColumnGenerationSolver,
        'label': 'CG Std',
        'color': 'red',
        'marker': 'o',
        'needs_history': False,
        'kwargs': {'use_pool': False}
    },
    'pool': {
        'class': ColumnGenerationSolver,
        'label': 'CG Pool',
        'color': 'green',
        'marker': 's',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'aging': {
        'class': ColumnGenerationSolverWithAging,
        'label': 'CG Aging',
        'color': 'blue',
        'marker': '^',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'smart': {
        'class': ColumnGenerationSolverSmart,
        'label': 'CG Smart',
        'color': 'purple',
        'marker': '*',
        'needs_history': True, # 履歴データが必要なフラグ
        'kwargs': {'use_pool': True}
    }
}

def run_benchmark_comparison(n_weeks=5, n_employees=10, selected_methods=None):
    """
    動的に選択された手法を比較するベンチマーク
    """
    # デフォルトは全手法
    if selected_methods is None:
        selected_methods = list(SOLVER_CONFIG.keys())

    # 設定の検証
    active_methods = []
    for m in selected_methods:
        if m in SOLVER_CONFIG:
            active_methods.append(m)
        else:
            print(f"Warning: Unknown method '{m}', skipping.")
    
    if not active_methods:
        raise ValueError("No valid methods selected.")

    # 出力ディレクトリ設定
    method_str = "_".join(active_methods)
    output_dir = f"schedule_plots_{n_employees}emp_{method_str}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Initializing Problem: {n_employees} Employees, {n_weeks} Weeks")
    print(f"Selected Methods: {', '.join([SOLVER_CONFIG[m]['label'] for m in active_methods])}")
    
    prob = ShiftProblemData(n_employees=n_employees)
    
    # ソルバーインスタンスの保持（状態を持つもの用）
    solvers = {}
    
    # Smart手法用の学習データ（履歴）保持
    # key: method_name, value: {time_index: freq}
    histories = defaultdict(dict)

    # ソルバーの初期化
    for name in active_methods:
        cfg = SOLVER_CONFIG[name]
        # Smartなど履歴が必要なものはループ内で都度更新・再設定するため、ここでは初期化のみ
        if name == 'exact':
            solvers[name] = cfg['class'](prob)
        elif cfg['needs_history']:
            # 初期は空の履歴で作成
            solvers[name] = cfg['class'](prob, historical_patterns={}, **cfg['kwargs'])
        else:
            solvers[name] = cfg['class'](prob, **cfg['kwargs'])

    results = []
    
    # === テーブルヘッダー作成 ===
    print(f"\nStarting Comparison Benchmark...")
    print("=" * (10 + 14 * len(active_methods) + 10))
    
    # ヘッダー行1: 時間
    header_time = f"{'Wk':<3} |"
    for m in active_methods:
        header_time += f" {SOLVER_CONFIG[m]['label'] + ' (s)':<12} |"
    
    # ヘッダー行2: Gap (Exactがある場合のみ)
    header_gap = ""
    has_exact = 'exact' in active_methods
    if has_exact:
        for m in active_methods:
            if m == 'exact': continue
            header_gap += f" Gap_{m}% |"

    print(header_time + header_gap)
    print("-" * (len(header_time) + len(header_gap)))
    
    # === 週次ループ ===
    for w in range(n_weeks):
        prob.generate_new_demand(period=w)
        
        week_result = {'Week': w+1}
        week_objs = {}
        
        # 各手法の実行
        row_str = f"{w+1:<3} |"
        
        for name in active_methods:
            cfg = SOLVER_CONFIG[name]
            solver = solvers[name]
            
            # --- 前処理 (Reset & History Injection) ---
            if name != 'exact':
                if cfg['needs_history']:
                    # Smart: 最新の履歴を注入して重み再計算
                    solver.historical_freq = histories[name]
                    if hasattr(solver, '_apply_batch_graph_adjustments'):
                        solver._apply_batch_graph_adjustments()
                
                solver.reset_for_new_period()

            # --- 実行 ---
            obj_val = 0.0
            elapsed = 0.0
            final_sched = None
            stats = {}

            if name == 'exact':
                if n_employees > 20:
                    obj_val, elapsed = 0.0, 0.0
                    final_sched = np.zeros((prob.K, prob.T))
                else:
                    obj_val, elapsed, final_sched = solver.solve(time_limit=1200)
            else:
                # CG系
                max_iter = 400 if name == 'std' else 200
                obj_val, elapsed, stats, final_sched = solver.solve(max_iter=max_iter)
                
                # プール保存
                solver.save_pool_to_csv(f"{output_dir}/pool_wk{w+1}_{name}.csv")
                
                # 履歴データ更新 (Smart用)
                if cfg['needs_history']:
                    for k in range(prob.K):
                        for t in range(prob.T):
                            if final_sched[k, t] == 1:
                                histories[name][t] = histories[name].get(t, 0) + 1

            # 結果記録
            week_objs[name] = obj_val
            week_result[f'Time_{name}'] = elapsed
            
            # 統計情報の保存（CG系のみ）
            if name != 'exact':
                prefix = SOLVER_CONFIG[name]['label'] # グラフ用ラベル等に使う
                # キー名は簡単のため name ('std', 'smart' 等) を使う
                week_result[f'{name}_RMP_LP'] = stats.get('time_rmp_lp', 0)
                week_result[f'{name}_RMP_MIP'] = stats.get('time_rmp_mip', 0)
                week_result[f'{name}_Pool'] = stats.get('time_pool', 0)
                week_result[f'{name}_Graph'] = stats.get('time_graph', 0)

            # レポートと画像保存
            ScheduleVisualizer.save_schedule_heatmap(
                final_sched, prob, f"Week {w+1} {cfg['label']}", 
                f"{output_dir}/wk{w+1}_{name}.png"
            )
            BenchmarkReporter.save_analysis_report(
                f"{output_dir}/report_wk{w+1}_{name}.txt", w+1, solver, prob, obj_val, elapsed, final_sched
            )

            # ログ表示用文字列作成
            row_str += f" {elapsed:<12.2f} |"

        # Gap計算と表示
        if has_exact:
            base_obj = week_objs['exact']
            for name in active_methods:
                if name == 'exact': continue
                gap = (week_objs[name] - base_obj)/base_obj * 100 if base_obj > 1e-5 else 0.0
                row_str += f" {gap:<6.2f} |"

        print(row_str)
        results.append(week_result)

    return pd.DataFrame(results), output_dir, active_methods



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Shift Scheduling Benchmark')
    parser.add_argument('--weeks', type=int, default=5, help='Number of weeks')
    parser.add_argument('--employees', type=int, default=10, help='Number of employees')
    
    # 手法選択用の引数（デフォルトは全手法）
    # usage: python main.py --methods exact std smart
    all_keys = list(SOLVER_CONFIG.keys())
    parser.add_argument('--methods', nargs='+', default=all_keys, 
                        choices=all_keys,
                        help=f'Methods to run. Options: {", ".join(all_keys)}')
    
    args = parser.parse_args()
    
    df, out_dir, active_methods = run_benchmark_comparison(
        n_weeks=args.weeks, 
        n_employees=args.employees,
        selected_methods=args.methods
    )
    
    # 1. 各手法の詳細内訳グラフ
    ComparisonPlotter.plot_dynamic_breakdown(df, active_methods, SOLVER_CONFIG, out_dir)
    
    # 2. 全体比較グラフ
    ComparisonPlotter.plot_overall_comparison(df, active_methods, SOLVER_CONFIG, out_dir)
    
    
    print(f"\nAll plots and reports saved to: {out_dir}/")
import os
import re
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
        'needs_history': True, 
        'kwargs': {'use_pool': True}
    }
}

def get_last_week_number(output_dir):
    """
    出力ディレクトリをスキャンして、すでに保存されている最大の週番号(wkX)を取得する。
    ファイルがない場合は 0 を返す。
    """
    max_week = 0
    if not os.path.exists(output_dir):
        return 0
    
    # pool_wk(\d+)_*.csv というパターンを探す
    pattern = re.compile(r'pool_wk(\d+)_.*\.csv')
    
    for filename in os.listdir(output_dir):
        match = pattern.search(filename)
        if match:
            week_num = int(match.group(1))
            if week_num > max_week:
                max_week = week_num
    return max_week

def run_benchmark_comparison(n_weeks=5, n_employees=10, selected_methods=None):
    if selected_methods is None:
        selected_methods = list(SOLVER_CONFIG.keys())

    active_methods = []
    for m in selected_methods:
        if m in SOLVER_CONFIG:
            active_methods.append(m)
        else:
            print(f"Warning: Unknown method '{m}', skipping.")
    
    if not active_methods:
        raise ValueError("No valid methods selected.")

    method_str = "_".join(active_methods)
    output_dir = f"schedule_plots_{n_employees}emp_{method_str}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === ★変更点1: 既存の最大週番号を取得して開始地点を決める ===
    last_week = get_last_week_number(output_dir)
    start_week_num = last_week + 1
    end_week_num = start_week_num + n_weeks
    
    print(f"Initializing Problem: {n_employees} Employees")
    if last_week > 0:
        print(f"Found existing data up to Week {last_week}. Resuming from Week {start_week_num}...")
    else:
        print(f"No existing data found. Starting from Week 1...")

    print(f"Running for {n_weeks} weeks (Week {start_week_num} to {end_week_num - 1})")
    print(f"Selected Methods: {', '.join([SOLVER_CONFIG[m]['label'] for m in active_methods])}")
    
    prob = ShiftProblemData(n_employees=n_employees)
    solvers = {}
    histories = defaultdict(dict)

    # ソルバー初期化
    for name in active_methods:
        cfg = SOLVER_CONFIG[name]
        if name == 'exact':
            solvers[name] = cfg['class'](prob)
        elif cfg['needs_history']:
            solvers[name] = cfg['class'](prob, historical_patterns={}, **cfg['kwargs'])
        else:
            solvers[name] = cfg['class'](prob, **cfg['kwargs'])

    results = []
    
    print(f"\nStarting Comparison Benchmark...")
    print("=" * (10 + 14 * len(active_methods) + 10))
    
    header_time = f"{'Wk':<3} |"
    for m in active_methods:
        header_time += f" {SOLVER_CONFIG[m]['label'] + ' (s)':<12} |"
    
    header_gap = ""
    has_exact = 'exact' in active_methods
    if has_exact:
        for m in active_methods:
            if m == 'exact': continue
            header_gap += f" Gap_{m}% |"

    print(header_time + header_gap)
    print("-" * (len(header_time) + len(header_gap)))
    
    # === ★変更点2: ループ範囲を実際の週番号 (start_week_num から) に合わせる ===
    for current_week in range(start_week_num, end_week_num):
        # generate_new_demandは0始まりのインデックス(period)を想定しているため -1 する
        prob.generate_new_demand(period=current_week - 1)
        
        week_result = {'Week': current_week}
        week_objs = {}
        
        row_str = f"{current_week:<3} |"
        
        for name in active_methods:
            cfg = SOLVER_CONFIG[name]
            solver = solvers[name]
            
            # --- 前処理 ---
            if name != 'exact':
                if cfg['needs_history']:
                    solver.historical_freq = histories[name]
                    if hasattr(solver, '_apply_batch_graph_adjustments'):
                        solver._apply_batch_graph_adjustments()
                
                solver.reset_for_new_period()

                # === ★変更点3: 常に「1つ前の週」のCSVを探して読み込む ===
                # current_weekが1より大きい場合、prev_week = current_week - 1 のファイルが存在するはず
                if current_week > 1 and cfg['kwargs'].get('use_pool', False):
                    prev_week_num = current_week - 1
                    prev_pool_file = f"{output_dir}/pool_wk{prev_week_num}_{name}.csv"
                    
                    if os.path.exists(prev_pool_file):
                        if hasattr(solver, 'load_pool_from_csv'):
                            # ログがうるさくなる場合は print を抑制しても良い
                            # print(f"DEBUG: Loading {prev_pool_file} for {name}") 
                            solver.load_pool_from_csv(prev_pool_file)
                    else:
                        # 途中再開だがファイルが見つからない場合の警告（初回実行時はweek=1なのでここには来ない）
                        print(f"Warning: Previous pool file not found: {prev_pool_file}")

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
                max_iter = 400 if name == 'std' else 200
                obj_val, elapsed, stats, final_sched = solver.solve(max_iter=max_iter)
                
                # --- 保存 (現在の週番号 current_week を使う) ---
                solver.save_pool_to_csv(f"{output_dir}/pool_wk{current_week}_{name}.csv")
                
                if cfg['needs_history']:
                    for k in range(prob.K):
                        for t in range(prob.T):
                            if final_sched[k, t] == 1:
                                histories[name][t] = histories[name].get(t, 0) + 1

            week_objs[name] = obj_val
            week_result[f'Time_{name}'] = elapsed
            
            if name != 'exact':
                week_result[f'{name}_RMP_LP'] = stats.get('time_rmp_lp', 0)
                week_result[f'{name}_RMP_MIP'] = stats.get('time_rmp_mip', 0)
                week_result[f'{name}_Pool'] = stats.get('time_pool', 0)
                week_result[f'{name}_Graph'] = stats.get('time_graph', 0)

            ScheduleVisualizer.save_schedule_heatmap(
                final_sched, prob, f"Week {current_week} {cfg['label']}", 
                f"{output_dir}/schedule_wk{current_week}_{name}.png"
            )
            BenchmarkReporter.save_analysis_report(
                f"{output_dir}/report_wk{current_week}_{name}.txt", current_week, solver, prob, obj_val, elapsed, final_sched
            )

            row_str += f" {elapsed:<12.2f} |"

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
    parser = argparse.ArgumentParser(description='Run Shift Scheduling Benchmark (Resume capability)')
    parser.add_argument('--weeks', type=int, default=5, help='Number of NEW weeks to run')
    parser.add_argument('--employees', type=int, default=10, help='Number of employees')
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
    
    # 注意: ここでプロットされるのは「今回実行した分」の統計のみです。
    # 過去分も含めて統合してプロットしたい場合は、別途全csvを読み込む処理が必要ですが、
    # まずは今回の実行結果を出力します。
    ComparisonPlotter.plot_dynamic_breakdown(df, active_methods, SOLVER_CONFIG, out_dir)
    ComparisonPlotter.plot_overall_comparison(df, active_methods, SOLVER_CONFIG, out_dir)
    
    print(f"\nAll plots and reports saved to: {out_dir}/")
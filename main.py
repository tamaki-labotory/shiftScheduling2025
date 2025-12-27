import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

# モジュール群のインポート
from problem import ShiftProblemData
from solver_exact import ExactMIPSolver
from solver_cg import ColumnGenerationSolver
from solver_cg_pruning import ColumnGenerationSolverWithAging
from solver_cg_lru import ColumnGenerationSolverLRU
from visualization import ScheduleVisualizer, BenchmarkReporter, ComparisonPlotter, MIPConvergencePlotter  

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
    'acc': {
        'class': ColumnGenerationSolver,
        'label': 'CG Acc',
        'color': 'green',
        'marker': 's',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'pruning': {
        'class': ColumnGenerationSolverWithAging,
        'label': 'CG Aging',
        'color': 'blue',
        'marker': '^',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'lru': {
        'class': ColumnGenerationSolverLRU,
        'label': 'CG LRU',
        'color': 'blue',
        'marker': '^',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    }
}

def get_last_week_number_from_config(config_file):
    """
    configファイルの履歴情報から、保存済みの最大週番号(wkX)を取得する。
    """
    if not os.path.exists(config_file):
        return 0
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            history = data.get('demand_history', {})
            if not history:
                return 0
            # キーを整数に変換して最大値を探す
            max_period_idx = max(int(k) for k in history.keys())
            return max_period_idx + 1 # period 0 -> Week 1
    except Exception as e:
        print(f"Warning: Could not read config file to determine last week: {e}")
        return 0

def parse_report_stats(filepath):
    """
    既存のレポートファイルから目的関数値と実行時間を抽出する
    """
    obj_val = 0.0
    elapsed = 0.0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Objective Value : 12,345.67
            m_obj = re.search(r"Objective Value\s*:\s*([\d,]+\.?\d*)", content)
            if m_obj:
                obj_val = float(m_obj.group(1).replace(',', ''))
            
            # Execution Time  : 12.3456 sec
            m_time = re.search(r"Execution Time\s*:\s*([\d\.]+)", content)
            if m_time:
                elapsed = float(m_time.group(1))
    except Exception as e:
        print(f"Warning: Failed to parse stats from {filepath}: {e}")
    return obj_val, elapsed

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

    output_dir = f"results_{n_employees}emp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 各手法ごとのサブディレクトリを作成
    for m in active_methods:
        method_dir = os.path.join(output_dir, m)
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

    config_file = os.path.join(output_dir, "problem_config.json")

    # === スタート地点と終了地点の計算 ===
    config_last_week = get_last_week_number_from_config(config_file)
    target_week_num = config_last_week + n_weeks
    
    print(f"Initializing Problem: {n_employees} Employees")
    if config_last_week > 0:
        print(f"Found existing configuration up to Week {config_last_week}.")
        print(f"Goal: Ensure results up to Week {target_week_num} (adding {n_weeks} new weeks).")
    else:
        print(f"No existing data found. Goal: Run from Week 1 to Week {target_week_num}.")

    print(f"Selected Methods: {', '.join([SOLVER_CONFIG[m]['label'] for m in active_methods])}")
    print(f"Output Directory: {output_dir}")
    
    # 問題設定のロードまたは作成
    if os.path.exists(config_file):
        prob = ShiftProblemData(config_path=config_file)
    else:
        prob = ShiftProblemData(n_employees=n_employees)
        prob.save_config(config_file)

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
    
    # === 常に Week 1 からターゲット週までループ ===
    for current_week in range(1, target_week_num + 1):
        
        # 需要生成 (履歴にあればロード)
        prob.generate_new_demand(period=current_week - 1)
        prob.save_config(config_file)
        
        week_result = {'Week': current_week}
        week_objs = {}
        
        row_str = f"{current_week:<3} |"
        
        for name in active_methods:
            cfg = SOLVER_CONFIG[name]
            solver = solvers[name]
            method_dir = os.path.join(output_dir, name)
            
            # --- パス定義 (手法ディレクトリ内にシンプル名で保存) ---
            pool_file = os.path.join(method_dir, f"pool_wk{current_week}.csv")
            report_file = os.path.join(method_dir, f"report_wk{current_week}.txt")
            
            # --- 実行要否判定 ---
            # プールとレポート両方があればスキップ可能とみなす
            skip_execution = False
            if name != 'exact':
                if os.path.exists(pool_file) and os.path.exists(report_file):
                    skip_execution = True
            else:
                if os.path.exists(report_file): # exactはpoolがない
                    skip_execution = True

            # --- 前処理 (共通) ---
            if name != 'exact':
                if cfg['needs_history']:
                    solver.historical_freq = histories[name]
                    if hasattr(solver, '_apply_batch_graph_adjustments'):
                        solver._apply_batch_graph_adjustments()
                
                # 新しい週のためにリセット
                solver.reset_for_new_period()

            # --- 実行またはスキップ ---
            obj_val = 0.0
            elapsed = 0.0
            stats = {}
            final_sched = None

            if skip_execution:
                # === SKIP ===
                obj_val, elapsed = parse_report_stats(report_file)
                
                # 次の週のために状態（Pool）を復元
                if name != 'exact' and cfg['kwargs'].get('use_pool', False):
                    if hasattr(solver, 'load_pool_from_csv'):
                        solver.load_pool_from_csv(pool_file)
                        
                row_str += f" {'(Skip)':<12} |"
            
            else:
                # === RUN ===
                # 計算実行
                if name == 'exact':
                    if n_employees > 20:
                        obj_val, elapsed = 0.0, 0.0
                        final_sched = np.zeros((prob.K, prob.T))
                    else:
                        obj_val, elapsed, final_sched = solver.solve(time_limit=3600)
                else:
                    max_iter = 400 if name == 'std' else 200
                    obj_val, elapsed, stats, final_sched = solver.solve(max_iter=max_iter)
                    
                    # 保存 (CGのみ)
                    solver.save_pool_to_csv(pool_file)
                    
                    if cfg['needs_history']:
                        for k in range(prob.K):
                            for t in range(prob.T):
                                if final_sched[k, t] == 1:
                                    histories[name][t] = histories[name].get(t, 0) + 1
                                    
                # === 共通保存処理（ここが修正箇所） ===
                # ExactでもCGでも実行される位置に配置
                schedule_img = os.path.join(method_dir, f"schedule_wk{current_week}.png")
                ScheduleVisualizer.save_schedule_heatmap(
                    final_sched, prob, f"Week {current_week} {cfg['label']}", 
                    schedule_img
                )
                BenchmarkReporter.save_analysis_report(
                    report_file, current_week, solver, prob, obj_val, elapsed, final_sched
                )

                # stats内にmip_trajectoryがあり、かつデータが存在する場合のみ描画
                if 'mip_trajectory' in stats and stats['mip_trajectory']:
                    mip_plot_file = os.path.join(method_dir, f"mip_convergence_wk{current_week}.png")
                    MIPConvergencePlotter.plot_convergence(
                        stats['mip_trajectory'], 
                        f"MIP Convergence: Week {current_week} ({cfg['label']})", 
                        mip_plot_file
                    )
                
                row_str += f" {elapsed:<12.2f} |"

            # 結果格納
            week_objs[name] = obj_val
            week_result[f'Time_{name}'] = elapsed
            
            if name != 'exact' and not skip_execution:
                week_result[f'{name}_RMP_LP'] = stats.get('time_rmp_lp', 0)
                week_result[f'{name}_RMP_MIP'] = stats.get('time_rmp_mip', 0)
                week_result[f'{name}_Pool'] = stats.get('time_pool', 0)
                week_result[f'{name}_Graph'] = stats.get('time_graph', 0)

        # Gap計算
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
    parser.add_argument('--weeks', type=int, default=5, help='Number of NEW weeks to add')
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
    
    # 比較グラフ生成
    ComparisonPlotter.plot_dynamic_breakdown(df, active_methods, SOLVER_CONFIG, out_dir)
    ComparisonPlotter.plot_overall_comparison(df, active_methods, SOLVER_CONFIG, out_dir)
    
    print(f"\nAll plots and reports saved to: {out_dir}/")
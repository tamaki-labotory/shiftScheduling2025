import os
import re
import argparse
import pandas as pd
import numpy as np
import json
import time
from collections import defaultdict

# モジュール群のインポート
# ※ ユーザー環境に合わせてクラス名などが正しいか確認してください
try:
    from problem import ShiftProblemData
    from solver_exact import ExactMIPSolver
    from solver_cg import ColumnGenerationSolver
    from solver_cg_pruning import ColumnGenerationSolverWithPruning
    from solver_cg_bnb import ColumnGenerationBnBSolver
    from solver_cg_priority import ColumnGenerationSolverPriority
    from solver_cg_neighbor import ColumnGenerationSolverNeighbor
    from visualization import ScheduleVisualizer, BenchmarkReporter, ComparisonPlotter, MIPConvergencePlotter
except ImportError as e:
    print(f"Warning: Module import failed ({e}). Please ensure all files are in the same directory.")

# ==========================================
# 設定: ソルバーの定義
# ==========================================
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
        'class': ColumnGenerationSolverWithPruning,
        'label': 'CG Pruning',
        'color': 'blue',
        'marker': '^',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'bnb': {
        'class': ColumnGenerationBnBSolver,
        'label': 'CG BnB',
        'color': 'purple',
        'marker': '*',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'priority': {
        'class': ColumnGenerationSolverPriority,
        'label': 'CG Priority',
        'color': 'orange',
        'marker': 'D',
        'needs_history': False,
        'kwargs': {'use_pool': True}
    },
    'neighbor': {
        'class': ColumnGenerationSolverNeighbor,
        'label': 'CG Neighbor (IP1B)', 
        'color': 'magenta',
        'marker': 'v',
        'needs_history': False,
        'kwargs': {'use_pool': False}
    }
}

# ==========================================
# ユーティリティ関数
# ==========================================
def ensure_dir(path):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_report_stats(filepath):
    obj_val = 0.0
    elapsed = 0.0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            m_obj = re.search(r"Objective Value\s*:\s*([\d,]+\.?\d*)", content)
            if m_obj:
                obj_val = float(m_obj.group(1).replace(',', ''))
            m_time = re.search(r"Execution Time\s*:\s*([\d\.]+)", content)
            if m_time:
                elapsed = float(m_time.group(1))
    except Exception as e:
        print(f"Warning: Failed to parse stats from {filepath}: {e}")
    return obj_val, elapsed

# ==========================================
# コア機能: ベンチマーク実行エンジン
# ==========================================
def run_benchmark_engine(n_weeks, n_employees, selected_methods, solver_params, output_dir_root):
    """
    指定されたパラメータと手法でベンチマークを実行する中核関数
    """
    # 1. メソッドのフィルタリング
    active_methods = []
    if selected_methods is None: selected_methods = list(SOLVER_CONFIG.keys())
    for m in selected_methods:
        if m in SOLVER_CONFIG: active_methods.append(m)
    
    if not active_methods:
        raise ValueError("No valid methods selected.")

    # 2. ディレクトリ準備
    ensure_dir(output_dir_root)
    for m in active_methods:
        ensure_dir(os.path.join(output_dir_root, m))
    
    # 3. Exact解のキャッシュディレクトリ準備
    cache_dir = "cache_exact"
    ensure_dir(cache_dir)

    # 4. 問題設定の読み込み/作成
    config_file = os.path.join(output_dir_root, "problem_config.json")
    
    # 【修正】指定された n_weeks で終了する（追加ではない）
    target_week_num = n_weeks 
    
    if os.path.exists(config_file):
        prob = ShiftProblemData(config_path=config_file)
    else:
        prob = ShiftProblemData(n_employees=n_employees)
        prob.save_config(config_file)

    # 5. ソルバーのインスタンス化
    solvers = {}
    histories = defaultdict(dict)
    
    for name in active_methods:
        cfg = SOLVER_CONFIG[name]
        if 'class' in cfg:
            if name == 'exact':
                solvers[name] = cfg['class'](prob)
            elif cfg['needs_history']:
                solvers[name] = cfg['class'](prob, historical_patterns={}, **cfg['kwargs'])
            else:
                solvers[name] = cfg['class'](prob, **cfg['kwargs'])

    results = []
    print(f"\nStarting Comparison Benchmark -> {output_dir_root}")
    print("=" * 80)
    
    # ヘッダー表示
    header_time = f"{'Wk':<3} |"
    for m in active_methods:
        header_time += f" {SOLVER_CONFIG[m]['label'] + ' (s)':<18} |"
    print(header_time)
    print("-" * len(header_time))

    # --- 週ごとのループ ---
    for current_week in range(1, target_week_num + 1):
        # 需要生成と保存
        prob.generate_new_demand(period=current_week - 1)
        prob.save_config(config_file)
        
        week_result = {'Week': current_week}
        row_str = f"{current_week:<3} |"
        
        for name in active_methods:
            cfg = SOLVER_CONFIG[name]
            solver = solvers[name]
            method_dir = os.path.join(output_dir_root, name)
            
            pool_file = os.path.join(method_dir, f"pool_wk{current_week}.csv")
            report_file = os.path.join(method_dir, f"report_wk{current_week}.txt")
            
            # --- 実行スキップ判定 ---
            skip_execution = False
            if os.path.exists(pool_file) and os.path.exists(report_file):
                skip_execution = True

            # --- 変数初期化 ---
            obj_val = 0.0
            elapsed = 0.0
            stats = {}
            final_sched = None
            
            exact_cache_json = os.path.join(cache_dir, f"exact_emp{n_employees}_wk{current_week}.json")
            exact_cache_sched = os.path.join(cache_dir, f"exact_emp{n_employees}_wk{current_week}.npy")

            # --- 実行ロジック ---
            if skip_execution:
                obj_val, elapsed = parse_report_stats(report_file)
                if name != 'exact' and cfg['kwargs'].get('use_pool', False):
                    if hasattr(solver, 'load_pool_from_csv'):
                        solver.load_pool_from_csv(pool_file)
                row_str += f" {'(Skip)':<12} |"
            
            elif name == 'exact':
                # === Exactのキャッシュ処理 ===
                if os.path.exists(exact_cache_json):
                    try:
                        with open(exact_cache_json, 'r') as f:
                            cached_data = json.load(f)
                            obj_val = cached_data['obj_val']
                            elapsed = cached_data['elapsed']
                            stats = cached_data.get('stats', {})
                        if os.path.exists(exact_cache_sched):
                            final_sched = np.load(exact_cache_sched)
                        else:
                            final_sched = np.zeros((prob.K, prob.T))
                        row_str += f" {elapsed:<12.2f} (Cache)|"
                    except Exception as e:
                        print(f"Error loading cache: {e}. Recalculating.")
                        skip_execution = False 
                
                if final_sched is None:
                    if n_employees > 20:
                        print(" [Info] Skipping Exact solver for large instance > 20")
                        obj_val, elapsed, stats, final_sched = 0.0, 0.0, {}, np.zeros((prob.K, prob.T))
                    else:
                        obj_val, elapsed, stats, final_sched = solver.solve(
                            time_limit=solver_params.get('time_limit', 3600),
                            gapRel=solver_params.get('mip_gap', 0.01)
                        )
                        with open(exact_cache_json, 'w') as f:
                            json.dump({'obj_val': obj_val, 'elapsed': elapsed, 'stats': stats}, f)
                        np.save(exact_cache_sched, final_sched)
                        row_str += f" {elapsed:<12.2f} |"

            else:
                # === 通常の手法 (CG等) ===
                if cfg['needs_history']:
                    solver.historical_freq = histories[name]
                    if hasattr(solver, '_apply_batch_graph_adjustments'):
                        solver._apply_batch_graph_adjustments()
                solver.reset_for_new_period()

                obj_val, elapsed, stats, final_sched = solver.solve(
                    max_iter=solver_params.get('max_iter', 1000),
                    time_limit=solver_params.get('time_limit', 3600),
                    tol=solver_params.get('tol', 1e-8),
                    patience=solver_params.get('patience', 10),
                    mip_rc_threshold=solver_params.get('mip_rc_threshold', 1e10),
                    mip_gap=solver_params.get('mip_gap', 0.0001)
                )
                
                if cfg['needs_history'] and final_sched is not None:
                    for k in range(prob.K):
                        for t in range(prob.T):
                            if final_sched[k, t] == 1:
                                histories[name][t] = histories[name].get(t, 0) + 1
                
                row_str += f" {elapsed:<12.2f} |"

            # --- 結果の保存 ---
            if final_sched is not None:
                schedule_img = os.path.join(method_dir, f"schedule_wk{current_week}.png")
                ScheduleVisualizer.save_schedule_heatmap(
                    final_sched, prob, f"Week {current_week} {cfg['label']}", 
                    schedule_img
                )
            
            if not skip_execution:
                BenchmarkReporter.save_analysis_report(
                    report_file, current_week, solver, prob, obj_val, elapsed, final_sched, solver_params=solver_params
                )
                if hasattr(solver, 'save_pool_to_csv'):
                     solver.save_pool_to_csv(pool_file)
                if 'mip_trajectory' in stats and stats['mip_trajectory']:
                    mip_plot_file = os.path.join(method_dir, f"mip_convergence_wk{current_week}.png")
                    MIPConvergencePlotter.plot_convergence(
                        stats['mip_trajectory'], 
                        f"MIP Convergence: Week {current_week} ({cfg['label']})", 
                        mip_plot_file
                    )

            # データフレーム用結果格納
            week_result[f'Time_{name}'] = elapsed
            if not skip_execution:
                week_result[f'{name}_FirstSolTime'] = stats.get('time_first_sol', None)
                week_result[f'{name}_BestSolTime'] = stats.get('time_best_sol', None)
                if name != 'exact':
                    week_result[f'{name}_RMP'] = stats.get('time_rmp', 0)
                    week_result[f'{name}_MIP'] = stats.get('time_mip', 0)
                    week_result[f'{name}_Pool'] = stats.get('time_pool', 0)
                    week_result[f'{name}_Graph'] = stats.get('time_graph', 0)
        
        print(row_str)
        results.append(week_result)

    return pd.DataFrame(results), output_dir_root, active_methods

# ==========================================
# 実験シナリオ
# ==========================================
def run_experiment_1(n_weeks, n_employees,target_methods=None):
    """実験1: Patienceの影響調査"""
    print(f"\n{'='*20} Experiment 1: Patience {'='*20}")
    
    patience_values = [5, 10, 15, 100]
    if target_methods is None:
        methods = ['std', 'acc', 'pruning']
    else:
        methods = target_methods

    base_params = {
        'max_iter': 1000, 'time_limit': 3600, 'tol': 1e-8,
        'mip_rc_threshold': 1e10, 'mip_gap': 0.0001
    }
    
    for p in patience_values:
        out_dir = f"results_exp1_pat_{p}"
        final_report = os.path.join(out_dir, "std", f"report_wk{n_weeks}.txt")
        if os.path.exists(final_report):
            print(f"[SKIP] Patience={p} is already done.")
            continue
            
        print(f"Running with patience={p}...")
        current_params = base_params.copy()
        current_params['patience'] = p
        
        df, _, _ = run_benchmark_engine(
            n_weeks=n_weeks,
            n_employees=n_employees,
            selected_methods=methods + ['exact'], 
            solver_params=current_params,
            output_dir_root=out_dir
        )
        df.to_csv(os.path.join(out_dir, "summary_results.csv"), index=False)
        print(f"Finished patience={p}")

def run_experiment_2(n_weeks, n_employees,target_methods=None):
    """実験2: RC Thresholdの影響調査"""
    print(f"\n{'='*20} Experiment 2: RC Threshold {'='*20}")
    
    thresholds = [1, 1000, 1000000, np.inf] 
    if target_methods is None:
        methods = ['std', 'acc', 'pruning']
    else:
        methods = target_methods
    
    base_params = {
        'max_iter': 1000, 'time_limit': 3600, 'tol': 1e-8,
        'patience': 10, 'mip_gap': 0.0001
    }
    
    for t in thresholds:
        t_label = "inf" if t >= 1e9 else str(t)
        out_dir = f"results_exp2_rc_{t_label}"
        final_report = os.path.join(out_dir, "std", f"report_wk{n_weeks}.txt")
        if os.path.exists(final_report):
            print(f"[SKIP] Threshold={t_label} is already done.")
            continue
        
        print(f"Running with RC Threshold={t_label}...")
        current_params = base_params.copy()
        current_params['mip_rc_threshold'] = t
        
        df, _, _ = run_benchmark_engine(
            n_weeks=n_weeks,
            n_employees=n_employees,
            selected_methods=methods + ['exact'],
            solver_params=current_params,
            output_dir_root=out_dir
        )
        df.to_csv(os.path.join(out_dir, "summary_results.csv"), index=False)
        print(f"Finished threshold={t_label}")

# ==========================================
# メインエントリ
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Shift Scheduling Benchmark & Experiments')
    parser.add_argument('mode', type=str, choices=['simple', 'exp1', 'exp2', 'all'], 
                        help='Mode: simple, exp1, exp2, or all (run exp1 then exp2)')
    
    parser.add_argument('--weeks', type=int, default=15, help='Number of weeks')
    parser.add_argument('--employees', type=int, default=5, help='Number of employees')
    
    all_keys = list(SOLVER_CONFIG.keys())
    parser.add_argument('--methods', nargs='+', default=all_keys, choices=all_keys, help='Methods for simple mode')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--rc_threshold', type=float, default=1e10)
    
    # ★修正箇所: time_limit 引数を追加
    parser.add_argument('--time_limit', type=float, default=3600, help='Time limit in seconds')

    args = parser.parse_args()
    
    if args.mode == 'simple':
        params = {
            'max_iter': 1000, 
            'time_limit': args.time_limit, # ★修正箇所: 引数の値を使用
            'tol': 1e-8,
            'patience': args.patience,
            'mip_rc_threshold': args.rc_threshold, 
            'mip_gap': 0.0001
        }
        out_dir = f"results_{args.employees}emp"
        
        df, out_dir, active_methods = run_benchmark_engine(
            n_weeks=args.weeks,
            n_employees=args.employees,
            selected_methods=args.methods,
            solver_params=params,
            output_dir_root=out_dir
        )
        df.to_csv(os.path.join(out_dir, "summary_results.csv"), index=False)
        ComparisonPlotter.plot_dynamic_breakdown(df, active_methods, SOLVER_CONFIG, out_dir)
        ComparisonPlotter.plot_overall_comparison(df, active_methods, SOLVER_CONFIG, out_dir)
        print(f"\nAll plots and reports saved to: {out_dir}/")

    elif args.mode == 'exp1':
        run_experiment_1(args.weeks, args.employees, args.methods)
        
    elif args.mode == 'exp2':
        run_experiment_2(args.weeks, args.employees, args.methods)
        
    elif args.mode == 'all':
        print("\n=== Running ALL Experiments (Exp1 -> Exp2) ===")
        run_experiment_1(args.weeks, args.employees)
        run_experiment_2(args.weeks, args.employees)
        print("\nAll experiments completed.")
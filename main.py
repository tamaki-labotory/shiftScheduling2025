import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# モジュール群のインポート
from problem import ShiftProblemData
from solver_exact import ExactMIPSolver
from solver_cg import ColumnGenerationSolver
from solver_cg_aging import ColumnGenerationSolverWithAging
from visualization import ScheduleVisualizer, BenchmarkReporter

output_dir = "schedule_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_benchmark(solver_type='default'):
    """
    solver_type: 'default' (solver_cg) or 'aging' (solver_cg_aging)
    """
    n_weeks = 5
    prob = ShiftProblemData(n_employees=5)
    
    # ソルバーの初期化
    solver_exact = ExactMIPSolver(prob)
    
    # Standard (比較基準) は常に基本のCGソルバー(プールなし)を使用
    solver_std = ColumnGenerationSolver(prob, use_pool=False)
    
    # Proposed (提案手法) を引数によって切り替え
    if solver_type == 'aging':
        print(">>> Using Solver: ColumnGenerationSolverWithAging")
        SolverClass = ColumnGenerationSolverWithAging
        prop_label = "Aging"
    else:
        print(">>> Using Solver: ColumnGenerationSolver (Default)")
        SolverClass = ColumnGenerationSolver
        prop_label = "Prop"

    solver_prop = SolverClass(prob, use_pool=True)
    
    results = []
    
    print(f"Starting Benchmark for {n_weeks} weeks with Visualization & Analysis...")
    print("-" * 120)
    # 出力ヘッダーも動的に変更
    print(f"{'Week':<4} | {'Exact(s)':<8} | {'Std(s)':<8} | {f'{prop_label}(s)':<8} | {'Gap_S%':<7} | {'Gap_P%':<7}")
    print("-" * 120)
    
    for w in range(n_weeks):
        prob.generate_new_demand(period=w)
        
        # 1. Exact Solver
        obj_ex, time_ex, sched_ex = solver_exact.solve(time_limit=300)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_ex, prob, f"Week {w+1} Exact", f"{output_dir}/schedule_week{w+1}_exact.png"
        )
        
        # 2. CG Standard (No Pool)
        solver_std.reset_for_new_period()
        obj_std, time_std, _, sched_std = solver_std.solve(max_iter=50)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_std, prob, f"Week {w+1} CG_Standard", f"{output_dir}/schedule_week{w+1}_cg_std.png"
        )
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/analysis_week{w+1}_std.txt", w+1, solver_std, prob, obj_std, time_std, sched_std
        )
        
        # 3. CG Proposed (With Pool / Aging)
        solver_prop.reset_for_new_period()
        obj_prop, time_prop, stats, sched_prop = solver_prop.solve(max_iter=50)
        
        # ファイル名に識別子を入れる
        filename_suffix = prop_label.lower() # 'prop' or 'aging'
        
        ScheduleVisualizer.save_schedule_heatmap(
            sched_prop, prob, f"Week {w+1} CG_{prop_label}", f"{output_dir}/schedule_week{w+1}_cg_{filename_suffix}.png"
        )
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/analysis_week{w+1}_{filename_suffix}.txt", w+1, solver_prop, prob, obj_prop, time_prop, sched_prop
        )
        
        # 集計
        gap_std = (obj_std - obj_ex)/obj_ex * 100 if obj_ex else 0
        gap_prop = (obj_prop - obj_ex)/obj_ex * 100 if obj_ex else 0
        
        print(f"{w+1:<4} | {time_ex:<8.2f} | {time_std:<8.2f} | {time_prop:<8.2f} | {gap_std:<7.2f} | {gap_prop:<7.2f}")
        
        results.append({
            'Week': w+1,
            'Solver_Type': prop_label,
            'Time_Prop': time_prop, 
            'RMP_LP_Time': stats['time_rmp_lp'],
            'RMP_MIP_Time': stats['time_rmp_mip'],
            'Pool_Time': stats['time_pool'], 
            'Graph_Time': stats['time_graph']
        })
        
    return pd.DataFrame(results), prop_label

def plot_breakdown(df, label):
    plt.figure(figsize=(10, 6))
    weeks = df['Week']
    
    # 積み上げグラフ
    p1 = plt.bar(weeks, df['RMP_LP_Time'], label='RMP (LP/Duals)')
    p2 = plt.bar(weeks, df['RMP_MIP_Time'], bottom=df['RMP_LP_Time'], label='RMP (Final MIP)')
    
    bottom_pool = df['RMP_LP_Time'] + df['RMP_MIP_Time']
    p3 = plt.bar(weeks, df['Pool_Time'], bottom=bottom_pool, label='Pool Search')
    
    bottom_graph = bottom_pool + df['Pool_Time']
    p4 = plt.bar(weeks, df['Graph_Time'], bottom=bottom_graph, label='Graph Search')
    
    plt.plot(weeks, df['Time_Prop'], color='black', marker='o', linestyle='-', linewidth=2, label=f'Total Time ({label})')
    
    plt.title(f"Breakdown of {label} Method Execution Time", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Time (seconds)")
    plt.xticks(weeks)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_breakdown_{label.lower()}.png")
    # plt.show() # 自動実行時はコメントアウト推奨

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Shift Scheduling Benchmark')
    parser.add_argument('--solver', type=str, choices=['default', 'aging'], default='default',
                        help='Choose the solver implementation: "default" (solver_cg) or "aging" (solver_cg_aging)')
    
    args = parser.parse_args()
    
    df, label_used = run_benchmark(solver_type=args.solver)
    plot_breakdown(df, label_used)
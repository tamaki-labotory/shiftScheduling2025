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

def run_benchmark_comparison(n_weeks=5, n_employees=10):
    """
    4つの手法（厳密、CG標準、CGプール蓄積、CGプールAging）を比較するベンチマーク
    """
    
    # 出力ディレクトリ設定
    output_dir = f"schedule_plots_{n_employees}emp_comparison"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Initializing Problem: {n_employees} Employees, {n_weeks} Weeks")
    prob = ShiftProblemData(n_employees=n_employees)
    
    # === 4つのソルバーを準備 ===
    # 1. 厳密解法
    solver_exact = ExactMIPSolver(prob)
    
    # 2. 通常のCG (プールなし)
    solver_std = ColumnGenerationSolver(prob, use_pool=False)
    
    # 3. 列プール方式ありCG (蓄積型: use_pool=True, 親クラスそのまま)
    solver_pool = ColumnGenerationSolver(prob, use_pool=True)
    
    # 4. 列プール方式ありCG (Aging型: 継承クラスを使用)
    solver_aging = ColumnGenerationSolverWithAging(prob, use_pool=True)
    
    results = []
    
    # ヘッダー出力
    print(f"\nStarting 4-Way Comparison Benchmark...")
    print("=" * 140)
    header = (f"{'Wk':<3} | {'Exact':<7} | {'Std':<7} | {'Pool':<7} | {'Aging':<7} | "
              f"{'Gap_S%':<6} | {'Gap_P%':<6} | {'Gap_A%':<6}")
    print(header)
    print("-" * 140)
    
    for w in range(n_weeks):
        prob.generate_new_demand(period=w)
        
        # --- 1. Exact Solver ---
        # 人数が多い場合はスキップする処理を入れても良い
        if n_employees > 20:
             obj_ex, time_ex = 0.0, 0.0
             sched_ex = np.zeros((prob.K, prob.T))
        else:
             obj_ex, time_ex, sched_ex = solver_exact.solve(time_limit=300)
             ScheduleVisualizer.save_schedule_heatmap(
                 sched_ex, prob, f"Week {w+1} Exact", f"{output_dir}/wk{w+1}_exact.png"
             )
        
        # --- 2. CG Standard (No Pool) ---
        solver_std.reset_for_new_period()
        obj_std, time_std, stats_std, sched_std = solver_std.solve(max_iter=200)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_std, prob, f"Week {w+1} CG_Std", f"{output_dir}/wk{w+1}_cg_std.png"
        )
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/report_wk{w+1}_std.txt", w+1, solver_std, prob, obj_std, time_std, sched_std
        )
        
        # --- 3. CG Pool (Accumulation) ---
        solver_pool.reset_for_new_period()
        obj_pool, time_pool, stats_pool, sched_pool = solver_pool.solve(max_iter=200)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_pool, prob, f"Week {w+1} CG_Pool", f"{output_dir}/wk{w+1}_cg_pool.png"
        )
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/report_wk{w+1}_pool.txt", w+1, solver_pool, prob, obj_pool, time_pool, sched_pool
        )

        # --- 4. CG Aging (With Removal) ---
        solver_aging.reset_for_new_period()
        obj_aging, time_aging, stats_aging, sched_aging = solver_aging.solve(max_iter=200)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_aging, prob, f"Week {w+1} CG_Aging", f"{output_dir}/wk{w+1}_cg_aging.png"
        )
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/report_wk{w+1}_aging.txt", w+1, solver_aging, prob, obj_aging, time_aging, sched_aging
        )
        
        # --- 集計と表示 ---
        def calc_gap(obj, base):
            return (obj - base)/base * 100 if base > 1e-5 else 0.0
            
        gap_std = calc_gap(obj_std, obj_ex)
        gap_pool = calc_gap(obj_pool, obj_ex)
        gap_aging = calc_gap(obj_aging, obj_ex)
        
        print(f"{w+1:<3} | {time_ex:<7.2f} | {time_std:<7.2f} | {time_pool:<7.2f} | {time_aging:<7.2f} | "
              f"{gap_std:<6.2f} | {gap_pool:<6.2f} | {gap_aging:<6.2f}")
        
        # 結果保存用辞書
        results.append({
            'Week': w+1,
            # Times
            'Time_Exact': time_ex,
            'Time_Std': time_std,
            'Time_Pool': time_pool,
            'Time_Aging': time_aging,
            # Stats (Std)
            'Std_RMP_LP': stats_std['time_rmp_lp'], 'Std_RMP_MIP': stats_std['time_rmp_mip'],
            'Std_Pool': stats_std['time_pool'], 'Std_Graph': stats_std['time_graph'],
            # Stats (Pool)
            'Pool_RMP_LP': stats_pool['time_rmp_lp'], 'Pool_RMP_MIP': stats_pool['time_rmp_mip'],
            'Pool_Pool': stats_pool['time_pool'], 'Pool_Graph': stats_pool['time_graph'],
            # Stats (Aging)
            'Aging_RMP_LP': stats_aging['time_rmp_lp'], 'Aging_RMP_MIP': stats_aging['time_rmp_mip'],
            'Aging_Pool': stats_aging['time_pool'], 'Aging_Graph': stats_aging['time_graph'],
        })
        
    return pd.DataFrame(results), output_dir

def plot_method_breakdown(df, method_prefix, label, output_dir, color_code):
    """
    指定された手法の時間内訳グラフを作成する
    method_prefix: 'Std', 'Pool', 'Aging'
    """
    plt.figure(figsize=(8, 5))
    weeks = df['Week']
    
    # データ取得
    rmp_lp = df[f'{method_prefix}_RMP_LP']
    rmp_mip = df[f'{method_prefix}_RMP_MIP']
    pool_t = df[f'{method_prefix}_Pool']
    graph_t = df[f'{method_prefix}_Graph']
    total_t = df[f'Time_{method_prefix}']
    
    # 積み上げ
    p1 = plt.bar(weeks, rmp_lp, label='RMP (LP)', color='#ff9999', alpha=0.8)
    p2 = plt.bar(weeks, rmp_mip, bottom=rmp_lp, label='RMP (Final MIP)', color='#66b3ff', alpha=0.8)
    
    bot_pool = rmp_lp + rmp_mip
    p3 = plt.bar(weeks, pool_t, bottom=bot_pool, label='Pool Search', color='#99ff99', alpha=0.8)
    
    bot_graph = bot_pool + pool_t
    p4 = plt.bar(weeks, graph_t, bottom=bot_graph, label='Graph Search', color='#ffcc99', alpha=0.8)
    
    # トータル時間線
    plt.plot(weeks, total_t, color='black', marker='o', linestyle='-', linewidth=1.5, label='Total Time')
    
    plt.title(f"Time Breakdown: {label}", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Time (s)")
    plt.xticks(weeks)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/breakdown_{method_prefix.lower()}.png")
    plt.close()

def plot_overall_comparison(df, output_dir):
    """
    4手法のトータル時間を比較するグラフ
    """
    plt.figure(figsize=(10, 6))
    weeks = df['Week']
    
    plt.plot(weeks, df['Time_Exact'], 'k--', marker='x', label='Exact MIP', alpha=0.5)
    plt.plot(weeks, df['Time_Std'], 'r-o', label='CG Standard (No Pool)')
    plt.plot(weeks, df['Time_Pool'], 'g-s', label='CG Pool (Accumulate)')
    plt.plot(weeks, df['Time_Aging'], 'b-^', label='CG Pool (Aging)')
    
    plt.title("Execution Time Comparison", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Time (seconds)")
    plt.xticks(weeks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_total_time.png")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 4-Way Shift Scheduling Comparison')
    parser.add_argument('--weeks', type=int, default=5, help='Number of weeks')
    parser.add_argument('--employees', type=int, default=10, help='Number of employees')
    
    args = parser.parse_args()
    
    df, out_dir = run_benchmark_comparison(n_weeks=args.weeks, n_employees=args.employees)
    
    # 各手法ごとの詳細内訳グラフを出力
    plot_method_breakdown(df, 'Std', 'CG Standard (No Pool)', out_dir, 'red')
    plot_method_breakdown(df, 'Pool', 'CG Pool (Accumulation)', out_dir, 'green')
    plot_method_breakdown(df, 'Aging', 'CG Pool (Aging)', out_dir, 'blue')
    
    # 全体比較グラフを出力
    plot_overall_comparison(df, out_dir)
    
    print(f"\nAll plots and reports saved to: {out_dir}/")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict

class ScheduleVisualizer:
    @staticmethod
    def save_schedule_heatmap(schedule, problem, title, filename):
        K, T = schedule.shape
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[50, 1], wspace=0.02, hspace=0.1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        cax = fig.add_subplot(gs[0, 1])
        
        cost_matrix = np.zeros((K, T))
        for k in range(K):
            emp = problem.employees[k]
            cost_matrix[k, :] = emp['hourly_wage'] + emp['rho']
            
        im = ax1.imshow(cost_matrix, aspect='auto', cmap='Reds', interpolation='nearest', alpha=0.5,
                        extent=[0, T, K-0.5, -0.5])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Cost (Wage + Penalty)', rotation=270, labelpad=15)

        bar_height = 0.6
        for k in range(K):
            ranges = []
            start_t = None
            for t in range(T):
                if schedule[k, t] == 1:
                    if start_t is None: start_t = t
                else:
                    if start_t is not None:
                        ranges.append((start_t, t - start_t))
                        start_t = None
            if start_t is not None:
                ranges.append((start_t, T - start_t))
            if ranges:
                ax1.broken_barh(ranges, (k - bar_height/2, bar_height), facecolors='tab:blue', edgecolors='black', linewidth=0.5)

        ax1.set_xlim(0, T)
        ax1.set_xticks(np.arange(0, T+1, 24))
        ax1.set_xticks(np.arange(0, T+1, 6), minor=True)
        ax1.set_yticks(np.arange(K))
        ax1.set_yticklabels([f"Emp {k} ({problem.employees[k]['type']})" for k in range(K)], fontsize=9)
        ax1.set_ylabel("Employee ID")
        ax1.set_title(f"{title}", fontsize=14)
        ax1.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(days):
            ax1.text(i * 24 + 12, K - 0.5, day, ha='center', va='top', fontsize=10, weight='bold', color='black')

        supplied = np.sum(schedule, axis=0)
        time_axis = np.arange(T)
        demand = problem.demand
        ax2.plot(time_axis, demand, 'r--', label='Demand', linewidth=2)
        ax2.fill_between(time_axis, supplied, step="mid", alpha=0.4, color='blue', label='Supplied')
        ax2.step(time_axis, supplied, 'b-', where="mid", linewidth=1.5)
        ax2.set_xlabel("Time (Hours)")
        ax2.set_ylabel("Headcount")
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, max(np.max(demand), np.max(supplied)) + 2)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

class BenchmarkReporter:
    @staticmethod
    def save_analysis_report(filename, week, solver, problem, final_obj, elapsed_time, final_schedule):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"==========================================================\n")
            f.write(f" ANALYSIS REPORT: Week {week}\n")
            f.write(f"==========================================================\n\n")

            f.write(f"1. Performance Metrics\n")
            f.write(f"----------------------\n")
            f.write(f"  Objective Value : {final_obj:,.2f}\n")
            f.write(f"  Execution Time  : {elapsed_time:.4f} sec\n")
            if hasattr(solver, 'stats'):
                f.write(f"  Iterations      : {solver.stats['iterations']}\n")
                f.write(f"  RMP Time (LP)   : {solver.stats['time_rmp_lp']:.4f} s\n")
                f.write(f"  RMP Time (MIP)  : {solver.stats['time_rmp_mip']:.4f} s\n")
                f.write(f"  Pool Search Time: {solver.stats['time_pool']:.4f} s\n")
                f.write(f"  Graph Search Time: {solver.stats['time_graph']:.4f} s\n")
                
                # ★修正: MIPで使用された決定変数の総数を表示
                if 'mip_total_columns' in solver.stats:
                    f.write(f"  MIP Decision Variables: {solver.stats['mip_total_columns']} (Columns used in Final MIP)\n")
                    
            f.write(f"\n")

            f.write(f"2. Column Generation & Pool Statistics\n")
            f.write(f"--------------------------------------\n")
            if hasattr(solver, 'pool'):
                total_pool_size = len(solver.pool)
                f.write(f"  Total Columns Generated (History) : {total_pool_size}\n")
                
                if hasattr(solver, 'stats'):
                    f.write(f"  Pool Hits (Reused from History)   : {solver.stats['count_pool_hit']}\n")
                    f.write(f"  Graph Gen (Newly Created)         : {solver.stats['count_graph_new']}\n")
                    if solver.stats['count_graph_new'] + solver.stats['count_pool_hit'] > 0:
                        hit_rate = solver.stats['count_pool_hit'] / (solver.stats['count_pool_hit'] + solver.stats['count_graph_new']) * 100
                        f.write(f"  Pool Hit Rate                     : {hit_rate:.1f}%\n")
            f.write(f"\n")

            f.write(f"3. Shift Pattern Diversity (Unique Patterns)\n")
            f.write(f"------------------------------------------\n")
            
            if hasattr(solver, 'pool') and solver.pool:
                type_patterns = defaultdict(set)
                type_total_cols = defaultdict(int)
                
                for col in solver.pool:
                    emp_id = col['group_id']
                    emp_type = problem.employees[emp_id]['type']
                    pat = tuple(col['schedule'])
                    if sum(pat) > 0: 
                        type_patterns[emp_type].add(pat)
                        type_total_cols[emp_type] += 1
                
                for t_name in sorted(type_patterns.keys()):
                    unique_count = len(type_patterns[t_name])
                    total_count = type_total_cols[t_name]
                    f.write(f"  Type: {t_name:<6} | Unique Patterns: {unique_count:>4} / Total Gen: {total_count:>4}\n")
                    work_hours = [sum(p) for p in type_patterns[t_name]]
                    avg_hours = np.mean(work_hours) if work_hours else 0
                    f.write(f"       -> Avg Length of Unique Patterns: {avg_hours:.1f} hours\n")

            f.write(f"\n4. Final Schedule Assignment Breakdown\n")
            f.write(f"--------------------------------------\n")
            for k in range(problem.K):
                total_work = np.sum(final_schedule[k])
                emp_type = problem.employees[k]['type']
                f.write(f"  Emp {k:<2} ({emp_type}): {int(total_work)} hours worked\n")

            f.write(f"\n5. Iteration History (Convergence Log)\n")
            f.write(f"--------------------------------------\n")
            if hasattr(solver, 'history') and solver.history:
                header = f"{'Iter':<5} | {'RMP Obj Value':<15} | {'Pool Hits':<10} | {'Graph Gen':<10}\n"
                f.write(header)
                f.write("-" * len(header) + "\n")
                for log in solver.history:
                    f.write(f"{log['iter']:<5} | {log['obj']:<15,.2f} | {log['pool_hits']:<10} | {log['graph_gen']:<10}\n")
            else:
                f.write("No iteration history available.\n")
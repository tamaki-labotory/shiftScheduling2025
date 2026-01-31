import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from collections import defaultdict

class MIPConvergencePlotter:
    @staticmethod
    def plot_convergence(trajectory, title, filename):
        if not trajectory:
            print(f"Warning: No trajectory data to plot for {filename}")
            return
        times = [x[0] for x in trajectory]
        objs = [x[1] for x in trajectory]
        plt.figure(figsize=(8, 5))
        plt.step(times, objs, where='post', color='b', linestyle='-', label='Incumbent Obj')
        plt.scatter([times[0]], [objs[0]], color='green', s=100, label='First Sol', zorder=5)
        plt.scatter([times[-1]], [objs[-1]], color='red', s=100, marker='*', label='Best Sol', zorder=5)
        plt.xlabel("Time (s)")
        plt.ylabel("Objective Value")
        plt.title(title)
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

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
            if start_t is not None: ranges.append((start_t, T - start_t))
            if ranges: ax1.broken_barh(ranges, (k - bar_height/2, bar_height), facecolors='tab:blue', edgecolors='black', linewidth=0.5)
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
    def save_analysis_report(filename, week, solver, problem, final_obj, elapsed_time, final_schedule, solver_params=None):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"==========================================================\n")
            f.write(f" ANALYSIS REPORT: Week {week}\n")
            f.write(f"==========================================================\n\n")
            if solver_params:
                f.write(f"0. Solver Configuration\n")
                f.write(f"-----------------------\n")
                priority_keys = ['max_iter', 'time_limit', 'tol', 'patience', 'mip_rc_threshold', 'mip_gap']
                for k in priority_keys:
                    if k in solver_params:
                        val = solver_params[k]
                        if isinstance(val, (int, float)): f.write(f"  {k:<20} : {val:g}\n")
                        else: f.write(f"  {k:<20} : {val}\n")
                for k, v in solver_params.items():
                    if k not in priority_keys: f.write(f"  {k:<20} : {v}\n")
                f.write(f"\n")
            f.write(f"1. Performance Metrics\n")
            f.write(f"----------------------\n")
            if abs(final_obj) > 1e15: f.write(f"  Objective Value : {final_obj:.4e}\n")
            else: f.write(f"  Objective Value : {final_obj:,.2f}\n")
            if hasattr(solver, 'stats') and 'rmp_obj_lp' in solver.stats:
                rmp = solver.stats['rmp_obj_lp']
                f.write(f"  RMP Relaxed Value (Root): {rmp:,.2f}\n")
                if abs(rmp) > 1e-5:
                    gap = (final_obj - rmp) / abs(rmp) * 100
                    f.write(f"  Integrality Gap       : {gap:.4f} %\n")
            if hasattr(solver, 'stats') and 'mip_lower_bound' in solver.stats:
                lb = solver.stats['mip_lower_bound']
                if lb is not None:
                    f.write(f"  MIP Best Bound (Final)  : {lb:,.2f}\n")
                    if abs(lb) > 1e-5:
                        final_gap = (final_obj - lb) / abs(lb) * 100
                        f.write(f"  Final MIP Gap           : {final_gap:.4f} %\n")
            f.write(f"  Execution Time  : {elapsed_time:.4f} sec\n")
            
            # ★追加: MIP初期解の時間をここに表示
            if hasattr(solver, 'stats') and 'time_mip_start' in solver.stats:
                f.write(f"  MIP Start Time  : {solver.stats['time_mip_start']:.4f} s\n")

            if hasattr(solver, 'stats'):
                f.write(f"  Iterations      : {solver.stats.get('iterations', 0)}\n")
                f.write(f"  RMP Time  : {solver.stats.get('time_rmp', 0):.4f} s\n")
                f.write(f"  MIP Time  : {solver.stats.get('time_mip', 0):.4f} s\n")
                f.write(f"  Pool Search Time: {solver.stats.get('time_pool', 0):.4f} s\n")
                f.write(f"  Graph Search Time: {solver.stats.get('time_graph', 0):.4f} s\n")
                if 'mip_total_columns' in solver.stats:
                    f.write(f"  MIP Decision Variables: {solver.stats['mip_total_columns']} (Columns used in Final MIP)\n")
                if 'mip_filtered_columns' in solver.stats:
                    f.write(f"  MIP Filtered Columns  : {solver.stats['mip_filtered_columns']} (Removed before MIP)\n")
            f.write(f"\n")
            
            f.write(f"2. Column Generation & Pool Statistics\n")
            f.write(f"--------------------------------------\n")
            if hasattr(solver, 'pool'):
                total_pool_size = len(solver.pool)
                f.write(f"  Total Columns Generated (History) : {total_pool_size}\n")
                if hasattr(solver, 'stats'):
                    if 'count_backlog_push' in solver.stats:
                        rmp_adds = solver.stats.get('count_pool_hit', 0)
                        q_adds = solver.stats.get('count_backlog_push', 0)
                        f.write(f"  Columns Added to RMP              : {rmp_adds}\n")
                        f.write(f"  Columns Added to Queue (Backlog)  : {q_adds}\n")
                    else:
                        hits = solver.stats.get('count_pool_hit', 0)
                        news = solver.stats.get('count_graph_new', 0)
                        f.write(f"  Pool Hits (Reused from History)   : {hits}\n")
                        f.write(f"  Graph Gen (Newly Created)         : {news}\n")
                        if news + hits > 0:
                            hit_rate = hits / (hits + news) * 100
                            f.write(f"  Pool Hit Rate                     : {hit_rate:.1f}%\n")
            f.write(f"\n")

            f.write(f"3. Shift Pattern Diversity (Unique Patterns)\n")
            f.write(f"------------------------------------------\n")
            if hasattr(solver, 'pool') and solver.pool:
                type_patterns = defaultdict(set)
                type_total_cols = defaultdict(int)
                for col in solver.pool:
                    emp_id = col['group_id']
                    if 0 <= emp_id < len(problem.employees):
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
            if final_schedule is not None:
                for k in range(min(problem.K, 20)):
                    total_work = np.sum(final_schedule[k])
                    emp_type = problem.employees[k]['type']
                    f.write(f"  Emp {k:<2} ({emp_type}): {int(total_work)} hours worked\n")

            f.write(f"\n5. Iteration History (Convergence Log)\n")
            f.write(f"--------------------------------------\n")
            if hasattr(solver, 'history') and solver.history:
                if hasattr(solver, 'stats') and 'count_backlog_push' in solver.stats:
                    header = f"{'Iter':<5} | {'RMP Obj Value':<15} | {'RMP Add':<10} | {'Queue Add':<10}\n"
                else:
                    header = f"{'Iter':<5} | {'RMP Obj Value':<15} | {'Pool Hits':<10} | {'Graph Gen':<10}\n"
                
                f.write(header)
                f.write("-" * len(header) + "\n")
                for log in solver.history:
                    f.write(f"{log['iter']:<5} | {log['obj']:<15,.2f} | {log['pool_hits']:<10} | {log['graph_gen']:<10}\n")
            else:
                f.write("No iteration history available.\n")

class ComparisonPlotter:
    @staticmethod
    def plot_dynamic_breakdown(df, active_methods, solver_config, output_dir):
        for name in active_methods:
            if name == 'exact': continue 
            cfg = solver_config.get(name, {})
            label = cfg.get('label', name)
            color = cfg.get('color', 'blue')
            if f'{name}_RMP' not in df.columns: continue
            plt.figure(figsize=(8, 5))
            weeks = df['Week']
            rmp = df[f'{name}_RMP']
            mip = df[f'{name}_MIP']
            pool_t = df[f'{name}_Pool']
            graph_t = df[f'{name}_Graph']
            total_t = df[f'Time_{name}']
            p1 = plt.bar(weeks, rmp, label='RMP', color='#ff9999', alpha=0.8)
            p2 = plt.bar(weeks, mip, bottom=rmp, label='MIP', color='#66b3ff', alpha=0.8)
            bot_pool = rmp + mip
            p3 = plt.bar(weeks, pool_t, bottom=bot_pool, label='Pool Search', color='#99ff99', alpha=0.8)
            bot_graph = bot_pool + pool_t
            p4 = plt.bar(weeks, graph_t, bottom=bot_graph, label='Graph Search', color='#ffcc99', alpha=0.8)
            plt.plot(weeks, total_t, color=color, marker='o', linestyle='-', linewidth=1.5, label='Total Time')
            plt.title(f"Time Breakdown: {label}", fontsize=14)
            plt.xlabel("Week")
            plt.ylabel("Time (s)")
            plt.xticks(weeks)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            method_dir = os.path.join(output_dir, name)
            if not os.path.exists(method_dir): os.makedirs(method_dir)
            plt.savefig(os.path.join(method_dir, "breakdown.png"))
            plt.close()

    @staticmethod
    def plot_overall_comparison(df, active_methods, solver_config, output_dir):
        plt.figure(figsize=(10, 6))
        weeks = df['Week']
        for name in active_methods:
            cfg = solver_config.get(name, {})
            label = cfg.get('label', name)
            color = cfg.get('color', None)
            marker = cfg.get('marker', 'o')
            time_col = f'Time_{name}'
            if time_col in df.columns:
                plt.plot(weeks, df[time_col], color=color, marker=marker, linestyle='--' if name == 'exact' else '-', label=label, linewidth=2, markersize=8)
        plt.title("Execution Time Comparison", fontsize=16)
        plt.xlabel("Week", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.xticks(weeks)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_total_time.png")
        plt.close()
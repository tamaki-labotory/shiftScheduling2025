import pulp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import defaultdict, Counter
import os
import matplotlib.gridspec as gridspec

# 保存先ディレクトリの作成
output_dir = "schedule_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 1. 問題データ設定
# ==========================================
class ShiftProblemData:
    def __init__(self, n_employees=10, seed=42):
        self.T = 168
        self.n_employees = n_employees
        self.seed = seed
        self.employees = []
        self.demand = None
        self.big_m = 1e6
        
        # 人数を動的に計算
        type_weights = {'TypeA': 1, 'TypeB': 2, 'TypeC': 2}
        total_weight = sum(type_weights.values())
        
        counts = {}
        current_total = 0
        for name, weight in type_weights.items():
            c = int(n_employees * (weight / total_weight))
            counts[name] = c
            current_total += c
            
        remainder = n_employees - current_total
        type_names = list(type_weights.keys())
        for i in range(remainder):
            counts[type_names[i % len(type_names)]] += 1
            
        self.types = [
            {'name': 'TypeA', 'L_min': 8, 'L_max': 10, 'R_int': 11, 'K_max': 5, 'base_w': 1000, 'count': counts['TypeA']},
            {'name': 'TypeB', 'L_min': 4, 'L_max': 8,  'R_int': 9,  'K_max': 5, 'base_w': 1000, 'count': counts['TypeB']},
            {'name': 'TypeC', 'L_min': 3, 'L_max': 6,  'R_int': 10, 'K_max': 5, 'base_w': 1000, 'count': counts['TypeC']}
        ]
        
        self._generate_employees()
        self.generate_new_demand(period=0)

    def _generate_employees(self):
        np.random.seed(self.seed)
        emp_id = 0
        for emp_type in self.types:
            for _ in range(emp_type['count']):
                rho = np.zeros(self.T, dtype=int)
                if emp_type['name'] == 'TypeA': 
                    for t in range(self.T):
                        if 9 > (t % 24) or (t % 24) > 18: rho[t] += 300
                elif emp_type['name'] == 'TypeB': 
                    for t in range(self.T):
                        if (t // 24) >= 5: rho[t] += 300
                elif emp_type['name'] == 'TypeC': 
                    for t in range(self.T):
                        if 9 <= (t % 24) <= 16: rho[t] += 300
                
                self.employees.append({
                    'id': emp_id,
                    'type': emp_type['name'],
                    'L_min': emp_type['L_min'], 'L_max': emp_type['L_max'],
                    'R_int': emp_type['R_int'], 'K_max': emp_type['K_max'],
                    'hourly_wage': emp_type['base_w'], 'rho': rho
                })
                emp_id += 1
        self.K = len(self.employees)

    def generate_new_demand(self, period):
        np.random.seed(self.seed + 100 + period)
        base_pattern = np.array([1,1,1,1,1, 2,3,4,5,5, 5,4,5,6,5, 4,3,2,2, 2,2,1,1,1])
        scale = (self.n_employees * 0.7) / 10.0 
        weekly_demand = np.tile(base_pattern * scale, 7)
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_demand + noise, 0, None).astype(int)
        for t in range(self.T):
            day = t // 24
            hour = t % 24
            if day >= 5: self.demand[t] = int(self.demand[t] * 1.2)
            if day == 4 and hour >= 18: self.demand[t] = int(self.demand[t] * 1.3)

# ==========================================
# 2. グラフ構築ヘルパー
# ==========================================
class GraphBuilder:
    @staticmethod
    def build_graph(problem, k):
        emp = problem.employees[k]
        G = nx.DiGraph()
        source, sink = 'S', 'T_node'
        T_end = problem.T
        L_min, L_max = emp['L_min'], emp['L_max']
        R_int = emp['R_int']
        K_max = emp['K_max']
        
        init_node = (0, R_int, 0, 0) 
        G.add_edge(source, init_node, weight=0, type='start')
        
        current_nodes = {init_node}
        for t in range(T_end):
            next_nodes = set()
            for u in current_nodes:
                _, dur, state, k_count = u
                if state == 0: # REST
                    next_dur = min(dur + 1, R_int)
                    v_rest = (t + 1, next_dur, 0, k_count)
                    G.add_edge(u, v_rest, weight=0, type='rest_cont')
                    next_nodes.add(v_rest)
                    if dur >= R_int and k_count < K_max and (t + 1 + L_min <= T_end):
                        v_work = (t + 1, 1, 1, k_count + 1)
                        cost = emp['hourly_wage'] + emp['rho'][t] 
                        G.add_edge(u, v_work, weight=cost, type='work_start', time=t)
                        next_nodes.add(v_work)
                elif state == 1: # WORK
                    if dur < L_max:
                        v_work = (t + 1, dur + 1, 1, k_count)
                        cost = emp['hourly_wage'] + emp['rho'][t]
                        G.add_edge(u, v_work, weight=cost, type='work_cont', time=t)
                        next_nodes.add(v_work)
                    if dur >= L_min:
                        v_rest = (t + 1, 1, 0, k_count)
                        G.add_edge(u, v_rest, weight=0, type='leave')
                        next_nodes.add(v_rest)
            current_nodes = next_nodes
            
        for u in current_nodes:
            G.add_edge(u, sink, weight=0, type='end')
        return G, source, sink

# ==========================================
# 3. 可視化・分析クラス (機能追加)
# ==========================================
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

# --- 新規追加: レポート作成クラス ---
class BenchmarkReporter:
    @staticmethod
    def save_analysis_report(filename, week, solver, problem, final_obj, elapsed_time, final_schedule):
        """
        ソルバーの状態、プールの内容、生成されたシフトパターンの多様性を分析してtxt出力
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"==========================================================\n")
            f.write(f" ANALYSIS REPORT: Week {week}\n")
            f.write(f"==========================================================\n\n")

            # 1. Basic Performance
            f.write(f"1. Performance Metrics\n")
            f.write(f"----------------------\n")
            f.write(f"  Objective Value : {final_obj:,.2f}\n")
            f.write(f"  Execution Time  : {elapsed_time:.4f} sec\n")
            if hasattr(solver, 'stats'):
                f.write(f"  Iterations      : {solver.stats['iterations']}\n")
                # --- 変更: RMP時間をLPとMIPに分けて出力 ---
                f.write(f"  RMP Time (LP)   : {solver.stats['time_rmp_lp']:.4f} s\n")
                f.write(f"  RMP Time (MIP)  : {solver.stats['time_rmp_mip']:.4f} s\n")
                # ----------------------------------------
                f.write(f"  Pool Search Time: {solver.stats['time_pool']:.4f} s\n")
                f.write(f"  Graph Search Time: {solver.stats['time_graph']:.4f} s\n")
            f.write(f"\n")

            # 2. Pool Statistics (Proposed Method Only usually has interesting stats here)
            f.write(f"2. Column Generation & Pool Statistics\n")
            f.write(f"--------------------------------------\n")
            if hasattr(solver, 'pool'):
                total_pool_size = len(solver.pool)
                # Active columns in final solution (approximation based on indices if available, or recalc)
                # RMP indices are those *considered* in the last LP, not necessarily chosen in IP.
                # Here we check actual usage in final_schedule
                
                f.write(f"  Total Columns Generated (History) : {total_pool_size}\n")
                
                if hasattr(solver, 'stats'):
                    f.write(f"  Pool Hits (Reused from History)   : {solver.stats['count_pool_hit']}\n")
                    f.write(f"  Graph Gen (Newly Created)         : {solver.stats['count_graph_new']}\n")
                    if solver.stats['count_graph_new'] + solver.stats['count_pool_hit'] > 0:
                        hit_rate = solver.stats['count_pool_hit'] / (solver.stats['count_pool_hit'] + solver.stats['count_graph_new']) * 100
                        f.write(f"  Pool Hit Rate                     : {hit_rate:.1f}%\n")
            f.write(f"\n")

            # 3. Pattern Diversity Analysis
            f.write(f"3. Shift Pattern Diversity (Unique Patterns)\n")
            f.write(f"------------------------------------------\n")
            
            # Extract patterns from Pool
            if hasattr(solver, 'pool') and solver.pool:
                # Group by Type
                type_patterns = defaultdict(set)
                type_total_cols = defaultdict(int)
                
                for col in solver.pool:
                    emp_id = col['group_id']
                    emp_type = problem.employees[emp_id]['type']
                    # Convert array to tuple for hashing
                    pat = tuple(col['schedule'])
                    # Ignore empty schedules for diversity count if desired, but keeping them is fine
                    if sum(pat) > 0: 
                        type_patterns[emp_type].add(pat)
                        type_total_cols[emp_type] += 1
                
                for t_name in sorted(type_patterns.keys()):
                    unique_count = len(type_patterns[t_name])
                    total_count = type_total_cols[t_name]
                    f.write(f"  Type: {t_name:<6} | Unique Patterns: {unique_count:>4} / Total Gen: {total_count:>4}\n")
                    
                    # Calculate stats for this type (Avg work hours)
                    work_hours = [sum(p) for p in type_patterns[t_name]]
                    avg_hours = np.mean(work_hours) if work_hours else 0
                    f.write(f"       -> Avg Length of Unique Patterns: {avg_hours:.1f} hours\n")

            # 4. Final Schedule Breakdown
            f.write(f"\n4. Final Schedule Assignment Breakdown\n")
            f.write(f"--------------------------------------\n")
            # Calculate actual working hours per employee in the final solution
            for k in range(problem.K):
                total_work = np.sum(final_schedule[k])
                emp_type = problem.employees[k]['type']
                f.write(f"  Emp {k:<2} ({emp_type}): {int(total_work)} hours worked\n")

            
            # 5. Iteration Log
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


        print(f"Report saved: {filename}")


# ==========================================
# 4. 厳密解法ソルバー
# ==========================================
class ExactMIPSolver:
    def __init__(self, problem):
        self.prob = problem

    def solve(self, time_limit=300):
        start_time = time.time()
        model = pulp.LpProblem("ExactMIP", pulp.LpMinimize)
        flow_vars = {} 
        obj_terms = []
        work_vars_mapping = []
        demand_vars = defaultdict(list)
        
        for k in range(self.prob.K):
            G, src, sink = GraphBuilder.build_graph(self.prob, k)
            for u, v, d in G.edges(data=True):
                var_name = f"x_{k}_{hash((u,v))}"
                x = pulp.LpVariable(var_name, 0, 1, pulp.LpBinary)
                flow_vars[(k, u, v)] = x
                if d['weight'] > 0: obj_terms.append(d['weight'] * x)
                
                if d.get('type') in ['work_start', 'work_cont']:
                    t = d['time']
                    demand_vars[t].append(x)
                    work_vars_mapping.append((k, t, x))
            
            model += pulp.lpSum([flow_vars[(k, src, v)] for v in G.successors(src)]) == 1
            for n in G.nodes():
                if n == src or n == sink: continue
                vin = pulp.lpSum([flow_vars[(k, u, n)] for u in G.predecessors(n)])
                vout = pulp.lpSum([flow_vars[(k, n, v)] for v in G.successors(n)])
                model += vin == vout
            model += pulp.lpSum([flow_vars[(k, u, sink)] for u in G.predecessors(sink)]) == 1

        slacks = [pulp.LpVariable(f"s_{t}", 0) for t in range(self.prob.T)]
        obj_terms.extend([self.prob.big_m * s for s in slacks])
        model += pulp.lpSum(obj_terms)
        
        for t in range(self.prob.T):
            model += pulp.lpSum(demand_vars[t]) + slacks[t] >= self.prob.demand[t]
            
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        model.solve(solver)
        
        elapsed = time.time() - start_time
        schedule = np.zeros((self.prob.K, self.prob.T))
        for k, t, x in work_vars_mapping:
            if x.varValue is not None and x.varValue > 0.5:
                schedule[k, t] = 1
        
        return pulp.value(model.objective), elapsed, schedule

# ==========================================
# 5. 列生成ソルバー
# ==========================================
class ColumnGenerationSolver:
    def __init__(self, problem, use_pool=True):
        self.prob = problem
        self.use_pool = use_pool
        self.pool = []
        self.rmp_indices = []
        self.graphs = {}
        
        # --- 追加: パターン重複チェック用の辞書 ---
        # Key: (group_id, tuple(schedule)), Value: col_id
        self.pattern_to_id = {} 

        # --- 追加: 反復履歴の保存用リスト ---
        self.history = []
        
        self.stats = {
            'time_rmp_lp': 0.0,   # 変更: LP緩和(双対変数用)の合計時間
            'time_rmp_mip': 0.0,  # 変更: 最後の整数解構築の合計時間
            'time_pool': 0.0,
            'time_graph': 0.0,
            'count_pool_hit': 0,
            'count_graph_new': 0,
            'count_graph_skip': 0,
            'iterations': 0,
            'pool_size': 0
        }

    def reset_stats(self):
        self.stats = {k: 0 for k in self.stats}

    def reset_for_new_period(self):
        self.rmp_indices = []
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            # --- 追加: プールリセット時は辞書もクリア ---
            self.pattern_to_id = {} 

    def initialize_rmp(self):
        self.rmp_indices = []
        for k in range(self.prob.K):
            # 初期列の追加（add_columnの変更により、ここでも重複チェックが効きます）
            idx_null = self.add_column(k, [0]*self.prob.T)
            self.rmp_indices.append(idx_null)
            
            emp = self.prob.employees[k]
            sched = [0]*self.prob.T
            start_t = 10
            end_t = min(start_t + emp['L_min'], self.prob.T)
            for t in range(start_t, end_t): 
                sched[t] = 1
            idx_simple = self.add_column(k, sched)
            # 重複していなければRMPに追加
            if idx_simple not in self.rmp_indices:
                self.rmp_indices.append(idx_simple)

    def add_column(self, k, schedule):
        # --- 変更: 重複チェックロジック ---
        
        # リストはハッシュ化できないため、タプルに変換してキーにする
        sched_tuple = tuple(schedule)
        pattern_key = (k, sched_tuple)

        # 既に存在する場合は、既存のIDを返して終了（プールには追加しない）
        if pattern_key in self.pattern_to_id:
            return self.pattern_to_id[pattern_key]

        # 存在しない場合のみ新規作成
        col_id = len(self.pool)
        emp = self.prob.employees[k]
        # コスト計算にはnumpy配列を使用（元のロジック維持）
        cost = np.sum(np.array(schedule) * (emp['hourly_wage'] + emp['rho']))
        
        self.pool.append({'id': col_id, 'group_id': k, 'schedule': schedule, 'cost': cost})
        
        # 辞書に登録
        self.pattern_to_id[pattern_key] = col_id
        
        return col_id

    def solve_rmp(self, integer=False):
        t_start = time.perf_counter()
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        active_cols = [self.pool[i] for i in self.rmp_indices]
        
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=cat) for c in active_cols}
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        model += pulp.lpSum([c['cost']*x[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * d for d in delta])
        
        cons_d = []
        for t in range(self.prob.T):
            expr = pulp.lpSum([c['schedule'][t]*x[c['id']] for c in active_cols]) + delta[t]
            model += expr >= self.prob.demand[t]
            cons_d.append(model.constraints[list(model.constraints.keys())[-1]])
            
        cons_c = []
        for k in range(self.prob.K):
            expr = pulp.lpSum([x[c['id']] for c in active_cols if c['group_id'] == k])
            model += expr == 1
            cons_c.append(model.constraints[list(model.constraints.keys())[-1]])
            
        solver = pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        elapsed = time.perf_counter() - t_start
        if integer:
            self.stats['time_rmp_mip'] += elapsed
        else:
            self.stats['time_rmp_lp'] += elapsed

        if integer:
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            for c in active_cols:
                val = x[c['id']].varValue
                if val is not None and val > 0.5:
                    final_schedule[c['group_id']] = c['schedule']
            return pulp.value(model.objective), final_schedule
        else:
            pi = [c.pi for c in cons_d]
            sigma = [c.pi for c in cons_c]
            return pulp.value(model.objective), pi, sigma

    def pricing(self, pi, sigma):
        # 戻り値を「(プールからの追加数, グラフからの追加数)」のタプルに変更
        pool_added_count = 0
        graph_added_count = 0
        
        t_pool_start = time.perf_counter()
        
        candidates = []
        if self.use_pool:
            for i, col in enumerate(self.pool):
                if i in self.rmp_indices: continue 
                k = col['group_id']
                rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
                if rc < -1e-5:
                    candidates.append((rc, i))
        
        candidates.sort(key=lambda x: x[0])
        limit_add = self.prob.K * 2 
        
        for rc, i in candidates[:limit_add]:
            self.rmp_indices.append(i)
            pool_added_count += 1
            self.stats['count_pool_hit'] += 1
        
        self.stats['time_pool'] += (time.perf_counter() - t_pool_start)

        # プールで十分な列が見つかった場合、グラフ探索をスキップ
        if self.use_pool and pool_added_count > 0:
            self.stats['count_graph_skip'] += self.prob.K 
            return pool_added_count, 0  # <--- タプルで返す

        t_graph_start = time.perf_counter()
        for k in range(self.prob.K):
            if k not in self.graphs:
                self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, src, sink = self.graphs[k]
            
            emp = self.prob.employees[k]
            for u, v, d in G.edges(data=True):
                etype = d.get('type')
                if etype in ['work_start', 'work_cont']:
                    t = d['time']
                    w = (emp['hourly_wage'] + emp['rho'][t]) - pi[t]
                    d['weight'] = w
                
                # --- 重要: 前回のバグ修正 (sigmaの減算場所を変更) ---
                elif etype == 'start': # Sourceから出るエッジで一律引く
                    d['weight'] = -sigma[k]
                elif etype == 'leave':
                    d['weight'] = 0 # ここでは引かない
                # ------------------------------------------------
                else:
                    d['weight'] = 0
            
            try:
                path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                sched = [0]*self.prob.T
                rc_val = 0
                for u, v in zip(path, path[1:]):
                    d = G[u][v]
                    rc_val += d['weight']
                    if d.get('type') in ['work_start', 'work_cont']:
                        sched[d['time']] = 1
                
                if rc_val < -1e-5:
                    idx = self.add_column(k, sched)
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        graph_added_count += 1
                        self.stats['count_graph_new'] += 1
            except nx.NetworkXNoPath:
                pass

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        
        return pool_added_count, graph_added_count # <--- タプルで返す

    def solve(self, max_iter=50):
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] # 履歴リセット
        
        for i in range(max_iter):
            res = self.solve_rmp(integer=False)
            if res is None: break
            obj, pi, sigma = res
            
            # pricingからの戻り値を分解
            pool_add, graph_add = self.pricing(pi, sigma)
            total_added = pool_add + graph_add
            
            self.stats['iterations'] += 1
            
            # --- 履歴の記録 ---
            self.history.append({
                'iter': i + 1,
                'obj': obj,
                'pool_hits': pool_add,
                'graph_gen': graph_add
            })
            
            if total_added == 0: break
            
        final_obj, final_schedule = self.solve_rmp(integer=True)
        self.stats['pool_size'] = len(self.pool)
        
        return final_obj, time.time() - start_total, self.stats, final_schedule

# ==========================================
# 6. ベンチマーク実行 (レポート出力を追加)
# ==========================================
def run_benchmark():
    n_weeks = 5
    prob = ShiftProblemData(n_employees=10)
    
    solver_exact = ExactMIPSolver(prob)
    solver_std = ColumnGenerationSolver(prob, use_pool=False)
    solver_prop = ColumnGenerationSolver(prob, use_pool=True)
    
    results = []
    
    print(f"Starting Benchmark for {n_weeks} weeks with Visualization & Analysis...")
    print("-" * 120)
    print(f"{'Week':<4} | {'Exact(s)':<8} | {'Std(s)':<8} | {'Prop(s)':<8} | {'Gap_S%':<7} | {'Gap_P%':<7}")
    print("-" * 120)
    
    for w in range(n_weeks):
        prob.generate_new_demand(period=w)
        
        # 1. Exact
        obj_ex, time_ex, sched_ex = solver_exact.solve(time_limit=300)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_ex, prob, f"Week {w+1} Exact", f"{output_dir}/schedule_week{w+1}_exact.png"
        )
        # Exactはソルバー構造が違うため今回は詳細分析対象外とする（必要ならラッパーを作って対応可能）
        
        # 2. Standard
        solver_std.reset_for_new_period()
        obj_std, time_std, _, sched_std = solver_std.solve(max_iter=50)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_std, prob, f"Week {w+1} CG_Standard", f"{output_dir}/schedule_week{w+1}_cg_std.png"
        )
        # Standardの分析レポート保存
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/analysis_week{w+1}_std.txt", w+1, solver_std, prob, obj_std, time_std, sched_std
        )
        
        # 3. Proposed
        solver_prop.reset_for_new_period()
        obj_prop, time_prop, stats, sched_prop = solver_prop.solve(max_iter=50)
        ScheduleVisualizer.save_schedule_heatmap(
            sched_prop, prob, f"Week {w+1} CG_Proposed", f"{output_dir}/schedule_week{w+1}_cg_prop.png"
        )
        # Proposedの分析レポート保存
        BenchmarkReporter.save_analysis_report(
            f"{output_dir}/analysis_week{w+1}_prop.txt", w+1, solver_prop, prob, obj_prop, time_prop, sched_prop
        )
        
        gap_std = (obj_std - obj_ex)/obj_ex * 100 if obj_ex else 0
        gap_prop = (obj_prop - obj_ex)/obj_ex * 100 if obj_ex else 0
        
        print(f"{w+1:<4} | {time_ex:<8.2f} | {time_std:<8.2f} | {time_prop:<8.2f} | {gap_std:<7.2f} | {gap_prop:<7.2f}")
        
        results.append({
            'Week': w+1,
            'Time_Prop': time_prop, 
            'RMP_LP_Time': stats['time_rmp_lp'],   # LP時間
            'RMP_MIP_Time': stats['time_rmp_mip'], # MIP時間
            'Pool_Time': stats['time_pool'], 
            'Graph_Time': stats['time_graph']
        })
        
    return pd.DataFrame(results)

def plot_breakdown(df):
    plt.figure(figsize=(10, 6))
    weeks = df['Week']
    
    # 1. RMP (LP) - 一番下
    p1 = plt.bar(weeks, df['RMP_LP_Time'], label='RMP (LP/Duals)')
    
    # 2. RMP (MIP) - その上
    p2 = plt.bar(weeks, df['RMP_MIP_Time'], bottom=df['RMP_LP_Time'], label='RMP (Final MIP)')
    
    # 3. Pool - その上 (LP + MIP の合計がボトムになる)
    bottom_pool = df['RMP_LP_Time'] + df['RMP_MIP_Time']
    p3 = plt.bar(weeks, df['Pool_Time'], bottom=bottom_pool, label='Pool Search')
    
    # 4. Graph - 一番上
    bottom_graph = bottom_pool + df['Pool_Time']
    p4 = plt.bar(weeks, df['Graph_Time'], bottom=bottom_graph, label='Graph Search')
    # --------------------------------
    
    plt.plot(weeks, df['Time_Prop'], color='black', marker='o', linestyle='-', linewidth=2, label='Total Time (Prop)')
    
    plt.title("Breakdown of Proposed Method Execution Time", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Time (seconds)")
    plt.xticks(weeks)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_breakdown.png")
    plt.show()

if __name__ == "__main__":
    df = run_benchmark()
    plot_breakdown(df)
import time
import heapq
import pulp
import numpy as np
from collections import defaultdict
from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverNeighbor(ColumnGenerationSolver):
    """
    【修正済み完全版: Neighbor Search + Correct Timing + Fix K_max】
    
    修正内容:
    1. 時間計測ロジックの適正化 (RMP, MIPの時間を正確に記録)
    2. solve_final_mip: 1人1パターン制約への変更
    3. is_feasible: K_max (勤務開始回数) チェックの実装
    4. PuLPソルバーオプションの記述を標準化
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.label = "CG Neighbor (Final w/ Cleanup)"
        self.backlog_queues = defaultdict(list) 
        self.seen_patterns = set()

    def reset_stats(self):
        super().reset_stats()
        # 追加の統計情報を確実に初期化
        self.stats.update({
            'count_backlog_push': 0,
            'count_feasible_discard': 0,
            'count_search_steps': 0,
            'time_mip_start': 0.0,
            'time_final_mip': 0.0,
            'time_rmp': 0.0,  # 上書きして確実にfloatにする
            'time_mip': 0.0
        })

    def reset_for_new_period(self):
        super().reset_for_new_period()
        self.backlog_queues = defaultdict(list)
        self.seen_patterns = set()

    def initialize_rmp(self):
        """MIPで初期解を構築"""
        super().initialize_rmp()
        print(f"  [MIP Start] Constructing initial feasible solution via MIP...")
        t_start = time.perf_counter()

        model = pulp.LpProblem("Original_Problem_Initialization", pulp.LpMinimize)
        K, T = self.prob.K, self.prob.T
        employees, demand = self.prob.employees, self.prob.demand
        
        x = [[pulp.LpVariable(f"x_{k}_{t}", cat=pulp.LpBinary) for t in range(T)] for k in range(K)]
        s = [[pulp.LpVariable(f"s_{k}_{t}", cat=pulp.LpBinary) for t in range(T)] for k in range(K)]
        delta = [pulp.LpVariable(f"delta_{t}", lowBound=0) for t in range(T)]

        obj_terms = []
        for k in range(K):
            emp = employees[k]
            cv = emp['hourly_wage'] + emp['rho']
            for t in range(T): obj_terms.append(cv[t] * x[k][t])
        for t in range(T): obj_terms.append(self.prob.big_m * delta[t])
        model += pulp.lpSum(obj_terms)

        for t in range(T):
            model += pulp.lpSum([x[k][t] for k in range(K)]) + delta[t] >= demand[t]

        for k in range(K):
            emp = employees[k]
            L_min, L_max = emp.get('L_min', 1), emp.get('L_max', T)
            R_int, K_max = emp.get('R_int', 0), emp.get('K_max', 7)

            for t in range(T):
                if t == 0: model += s[k][t] >= x[k][t]
                else: model += s[k][t] >= x[k][t] - x[k][t-1]

            for t in range(T):
                if t + L_min <= T:
                    model += pulp.lpSum([x[k][t+j] for j in range(L_min)]) >= L_min * s[k][t]

            for t in range(T - L_max):
                model += pulp.lpSum([x[k][t+j] for j in range(L_max + 1)]) <= L_max

            if R_int > 0:
                for t in range(T):
                    start_lookback = max(0, t - R_int)
                    if t > 0:
                        model += pulp.lpSum([x[k][j] for j in range(start_lookback, t)]) <= (t - start_lookback) * (1 - s[k][t])
            
            model += pulp.lpSum([s[k][t] for t in range(T)]) <= K_max

        # ソルバー設定 (修正: optionsでスレッド指定)
        mip_time_limit = 120 
        solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=mip_time_limit, gapRel=0.1, options=['threads 4'])
        model.solve(solver)

        elapsed = time.perf_counter() - t_start
        self.stats['time_mip_start'] = elapsed

        status = pulp.LpStatus[model.status]
        added_count = 0
        if status in ['Optimal', 'Integer']:
            for k in range(K):
                sched = [0] * T
                for t in range(T):
                    if x[k][t].varValue and x[k][t].varValue > 0.5: sched[t] = 1
                if sum(sched) > 0:
                    idx = self.add_column(k, sched)
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        added_count += 1
            print(f"  [MIP Start] Added {added_count} cols. Time: {elapsed:.2f}s")
        else:
            print(f"  [MIP Start] Failed (Status: {status}). Time: {elapsed:.2f}s")

    def pricing(self, pi, sigma):
        t_start = time.perf_counter()
        
        self.backlog_queues = defaultdict(list)
        self.seen_patterns = set() 
        start_queue_count = self.stats['count_backlog_push']
        
        seed_limit = 20
        best_indices = self._select_promising_rmp_indices(pi, sigma, limit=seed_limit)
        
        self._generate_neighbors_from_indices(best_indices, pi, sigma)
        
        total_added_to_rmp = 0
        SEARCH_BUDGET = 2000
        MAX_QUEUE_SIZE = 5000
        
        K = self.prob.K
        for k_idx, k in enumerate(range(K)):
            steps = 0
            emp = self.prob.employees[k]
            wage_vec = emp['hourly_wage'] + emp['rho']
            
            while self.backlog_queues[k]:
                if steps >= SEARCH_BUDGET: break
                
                rc_stored, eid, sched_tuple, cost = heapq.heappop(self.backlog_queues[k])
                sched_arr = np.array(sched_tuple)
                current_rc = cost - np.dot(pi, sched_arr) - sigma[eid]
                
                if current_rc > -1e-4: continue

                steps += 1
                if self.is_feasible(emp, sched_tuple):
                    if (eid, sched_tuple) not in self.pattern_to_id:
                        new_id = len(self.pool)
                        self.pool.append({'id': new_id, 'group_id': eid, 'schedule': list(sched_tuple), 'cost': cost})
                        self.pattern_to_id[(eid, sched_tuple)] = new_id
                    
                    col_id = self.pattern_to_id[(eid, sched_tuple)]
                    if col_id not in self.rmp_indices:
                        self.rmp_indices.append(col_id)
                        total_added_to_rmp += 1
                        
                        new_candidates = []
                        self._generate_neighbors_for_schedule(sched_arr, eid, pi, sigma, wage_vec, new_candidates)
                        for cand in new_candidates: self._push_to_queue(cand)

            self.stats['count_search_steps'] += steps
            if len(self.backlog_queues[k]) > MAX_QUEUE_SIZE:
                self.backlog_queues[k] = heapq.nsmallest(MAX_QUEUE_SIZE // 2, self.backlog_queues[k])
                heapq.heapify(self.backlog_queues[k])

        self.stats['time_pool'] += (time.perf_counter() - t_start)
        queue_added_this_iter = self.stats['count_backlog_push'] - start_queue_count

        if total_added_to_rmp > 0:
            return total_added_to_rmp, queue_added_this_iter
        if queue_added_this_iter > 0:
            return 0, queue_added_this_iter
        return 0, 0

    def _select_promising_rmp_indices(self, pi, sigma, limit=20):
        candidates = []
        for idx in self.rmp_indices:
            col = self.pool[idx]
            emp_id, sched, cost = col['group_id'], np.array(col['schedule']), col['cost']
            rc = cost - np.dot(pi, sched) - sigma[emp_id]
            candidates.append((rc, idx))
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:limit]]

    def _generate_neighbors_from_indices(self, indices, pi, sigma):
        for idx in indices:
            col = self.pool[idx]
            emp_id = col['group_id']
            base_schedule = np.array(col['schedule'])
            emp = self.prob.employees[emp_id]
            wage_vec = emp['hourly_wage'] + emp['rho']
            candidates = []
            self._generate_neighbors_for_schedule(base_schedule, emp_id, pi, sigma, wage_vec, candidates)
            for cand in candidates: self._push_to_queue(cand)

    def _generate_neighbors_for_schedule(self, base_schedule, emp_id, pi, sigma, wage_vec, candidates):
        T = self.prob.T
        emp = self.prob.employees[emp_id]
        L_min = emp.get('L_min', 1)
        L_max = emp.get('L_max', T)
        
        padded = np.concatenate(([0], base_schedule, [0]))
        diffs = np.diff(padded)
        starts, ends = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
        
        for i, (s, e) in enumerate(zip(starts, ends)):
            length = e - s
            if s > 0 and base_schedule[s-1] == 0:
                if length + 1 <= L_max:
                    new = base_schedule.copy(); new[s-1] = 1; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if length - 1 >= L_min:
                new = base_schedule.copy(); new[s] = 0; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if e < T and base_schedule[e] == 0:
                if length + 1 <= L_max:
                    new = base_schedule.copy(); new[e] = 1; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if length - 1 >= L_min:
                new = base_schedule.copy(); new[e-1] = 0; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if s > 0 and base_schedule[s-1] == 0:
                new = base_schedule.copy(); new[s-1:e-1] = 1; new[e-1] = 0; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if e < T and base_schedule[e] == 0:
                new = base_schedule.copy(); new[s] = 0; new[s+1:e+1] = 1; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)
            if i < len(starts) - 1:
                next_s = starts[i+1]
                if (ends[i+1] - s) <= L_max:
                    new = base_schedule.copy(); new[e:next_s] = 1; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)

        gaps = []
        if len(starts) > 0 and starts[0] > 0: gaps.append((0, starts[0]))
        elif len(starts) == 0: gaps.append((0, T))
        for i in range(len(starts) - 1): gaps.append((ends[i], starts[i+1]))
        if len(ends) > 0 and ends[-1] < T: gaps.append((ends[-1], T))
            
        stride = max(1, L_min)
        for g_start, g_end in gaps:
            if (g_end - g_start) >= L_min:
                for s in range(g_start, g_end - L_min + 1, stride):
                    new = base_schedule.copy(); new[s : s + L_min] = 1; self._add_candidate_lazy(new, emp_id, pi, sigma, wage_vec, candidates)

    def _add_candidate_lazy(self, new_sched_arr, emp_id, pi, sigma, wage_vec, candidates):
        sched_tuple = tuple(new_sched_arr.tolist())
        if (emp_id, sched_tuple) in self.seen_patterns: return
        cost_n = np.dot(new_sched_arr, wage_vec)
        rc = cost_n - np.dot(pi, new_sched_arr) - sigma[emp_id]
        if rc < -0.01: 
            candidates.append((rc, emp_id, sched_tuple, cost_n))

    def _push_to_queue(self, candidate):
        rc, eid, sched_tuple, cost = candidate
        if (eid, sched_tuple) in self.seen_patterns: return
        heapq.heappush(self.backlog_queues[eid], (rc, eid, sched_tuple, cost))
        self.seen_patterns.add((eid, sched_tuple))
        self.stats['count_backlog_push'] += 1

    def is_feasible(self, emp, schedule):
        sched_arr = np.array(schedule)
        padded = np.concatenate(([0], sched_arr, [0]))
        diffs = np.diff(padded)
        starts, ends = np.where(diffs == 1)[0], np.where(diffs == -1)[0]
        lengths = ends - starts
        if 'L_min' in emp and np.any(lengths < emp['L_min']): return False
        if 'L_max' in emp and np.any(lengths > emp['L_max']): return False
        
        # K_max制約（最大勤務回数）のチェック
        if 'K_max' in emp and len(starts) > emp['K_max']: return False

        if len(starts) > 1:
            gaps = starts[1:] - ends[:-1]
            if 'R_int' in emp and np.any(gaps < emp['R_int']): return False
        return True
    
    def solve_rmp(self, integer=False, **kwargs):
        """
        RMPを解き、計算時間を確実に stats に反映させる。
        """
        t_start = time.perf_counter()
        
        self.rmp_prob = pulp.LpProblem("RMP", pulp.LpMinimize)
        
        active_cols = [self.pool[i] for i in self.rmp_indices]
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        
        x_dict = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=cat) for c in active_cols}
        
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        # 目的関数
        self.rmp_prob += pulp.lpSum([c['cost']*x_dict[c['id']] for c in active_cols]) + \
                         pulp.lpSum([self.prob.big_m * d for d in delta])
        
        # 制約追加
        for t in range(self.prob.T):
            expr = pulp.lpSum([c['schedule'][t]*x_dict[c['id']] for c in active_cols]) + delta[t]
            self.rmp_prob += expr >= self.prob.demand[t], f"Demand_{t}"
            
        for k in range(self.prob.K):
            expr = pulp.lpSum([x_dict[c['id']] for c in active_cols if c['group_id'] == k])
            self.rmp_prob += expr == 1, f"Convexity_{k}"
            
        # ソルバー設定
        solver_opts = ['threads 4']
        if integer:
            # 整数計画の場合はタイムリミットを設ける
            limit = kwargs.get('mip_time_limit', 3600)
            solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=limit, gapRel=0.01, options=solver_opts)
        else:
            # 線形緩和は通常短時間で終わるが、念のため
            solver = pulp.COIN_CMD(path='cbc', msg=0, options=solver_opts)

        self.rmp_prob.solve(solver)
        
        # 変数値の保存
        self.rmp_vars = [None] * len(self.pool)
        for c in active_cols:
            self.rmp_vars[c['id']] = x_dict[c['id']]

        # ★時間計測の確定
        elapsed = time.perf_counter() - t_start
        
        if integer:
            # 最終MIP用
            self.stats['time_mip'] += elapsed
            self.stats['time_final_mip'] = elapsed
            
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            selected_ids = []
            if self.rmp_prob.status in [pulp.LpStatusOptimal, pulp.LpStatusInteger]:
                for c in active_cols:
                    if x_dict[c['id']].varValue > 0.5:
                        final_schedule[c['group_id']] = c['schedule']
                        selected_ids.append(c['id'])
                return pulp.value(self.rmp_prob.objective), final_schedule, selected_ids
            else:
                return None
        else:
            # RMP用
            self.stats['time_rmp'] += elapsed
            
            if self.rmp_prob.status != pulp.LpStatusOptimal:
                 # 緩和問題が解けないケース（通常ありえないが）
                 return 0.0, np.zeros(self.prob.T), np.zeros(self.prob.K)

            # Dualの取得
            pi = np.zeros(self.prob.T)
            for t in range(self.prob.T):
                if f"Demand_{t}" in self.rmp_prob.constraints:
                    pi[t] = self.rmp_prob.constraints[f"Demand_{t}"].pi
                    
            sigma = np.zeros(self.prob.K)
            for k in range(self.prob.K):
                if f"Convexity_{k}" in self.rmp_prob.constraints:
                    sigma[k] = self.rmp_prob.constraints[f"Convexity_{k}"].pi
                    
            return pulp.value(self.rmp_prob.objective), pi, sigma

    def solve(self, max_iter=50, time_limit=None, tol=1e-4, **kwargs):
        self.reset_stats()
        self.history = [] 

        print(f"=== Starting CG Solver: {self.label} ===")
        t_start_global = time.time()
        
        self.initialize_rmp()
        
        for i in range(max_iter):
            if time_limit and (time.time() - t_start_global > time_limit):
                print(">>> Time limit reached.")
                break
            
            # --- Progress Logging ---
            if (i + 1) % 5 == 0:
                 print(f"--- Iteration {i+1} ---")
            
            res = self.solve_rmp(integer=False)
            current_obj, pi, sigma = res

            added_count, queue_count = self.pricing(pi, sigma)
            
            self.stats['iterations'] += 1
            self.history.append({
                'iter': i + 1,
                'obj': current_obj,
                'pool_hits': added_count,      
                'graph_gen': queue_count       
            })

            if added_count == 0:
                print(">>> Convergence Reached (No new columns added).")
                break
        else:
            print(">>> Max iterations reached.")

        print("\n=== Final Phase: Cleanup & MIP ===")
        pi = self.get_dual_pi()
        sigma = self.get_dual_sigma()
        
        cleaned_indices = self.cleanup_rmp(pi, sigma, keep_threshold=0.05)
        
        final_schedule, obj_val = self.solve_final_mip(cleaned_indices)
        
        elapsed = time.time() - t_start_global
        
        return obj_val, elapsed, self.stats, final_schedule

    def _get_lp_constraints(self):
        """Pulpの制約辞書を安全に取得するヘルパー"""
        if hasattr(self, 'rmp_prob') and self.rmp_prob is not None:
            return self.rmp_prob.constraints
        elif hasattr(self.prob, 'rmp_prob') and self.prob.rmp_prob is not None:
            return self.prob.rmp_prob.constraints
        return {}

    def get_dual_pi(self):
        pi = np.zeros(self.prob.T)
        constraints = self._get_lp_constraints()
        for t in range(self.prob.T):
            keys = [f"Demand_{t}", f"demand_{t}"]
            for key in keys:
                if key in constraints:
                    pi[t] = constraints[key].pi
                    break
        return pi

    def get_dual_sigma(self):
        sigma = np.zeros(self.prob.K)
        constraints = self._get_lp_constraints()
        for k in range(self.prob.K):
            keys = [f"Convexity_{k}", f"convexity_{k}"]
            for key in keys:
                if key in constraints:
                    sigma[k] = constraints[key].pi
                    break
        return sigma

    def cleanup_rmp(self, pi, sigma, keep_threshold=0.05):
        initial_count = len(self.rmp_indices)
        kept_indices = []
        
        for idx in self.rmp_indices:
            col = self.pool[idx]
            emp_id = col['group_id']
            sched = np.array(col['schedule'])
            cost = col['cost']
            
            var_val = 0
            if hasattr(self, 'rmp_vars') and idx < len(self.rmp_vars):
                if self.rmp_vars[idx] is not None:
                    var_val = self.rmp_vars[idx].varValue
            
            is_basis = (var_val is not None and var_val > 1e-6)
            rc = cost - np.dot(pi, sched) - sigma[emp_id]
            is_promising = (rc < keep_threshold)
            
            if is_basis or is_promising:
                kept_indices.append(idx)
        
        print(f"  [Cleanup] Reduced columns from {initial_count} to {len(kept_indices)}.")
        return kept_indices

    def solve_final_mip(self, indices_to_use):
        t_start = time.perf_counter()
        print(f"  [Final MIP] Building MIP with {len(indices_to_use)} columns...")
        
        mip_model = pulp.LpProblem("Final_MIP_Schedule", pulp.LpMinimize)
        
        use_vars = []
        for idx in indices_to_use:
            col = self.pool[idx]
            v_name = f"y_{col['group_id']}_id{col['id']}"
            v = pulp.LpVariable(v_name, cat=pulp.LpBinary)
            use_vars.append((v, idx))
            
        delta = [pulp.LpVariable(f"final_delta_{t}", lowBound=0) for t in range(self.prob.T)]
        
        obj_list = []
        for v, idx in use_vars:
            obj_list.append(self.pool[idx]['cost'] * v)
        for t in range(self.prob.T):
            obj_list.append(self.prob.big_m * delta[t])
        mip_model += pulp.lpSum(obj_list)
        
        vars_at_t = [[] for _ in range(self.prob.T)]
        for v, idx in use_vars:
            sched = self.pool[idx]['schedule']
            for t, val in enumerate(sched):
                if val > 0.5:
                    vars_at_t[t].append(v)
        
        for t in range(self.prob.T):
            mip_model += pulp.lpSum(vars_at_t[t]) + delta[t] >= self.prob.demand[t], f"Demand_{t}"
            
        vars_by_emp = defaultdict(list)
        for v, idx in use_vars:
            emp_id = self.pool[idx]['group_id']
            vars_by_emp[emp_id].append(v)
            
        for k in range(self.prob.K):
            if k in vars_by_emp:
                # 1人1パターンの制約
                mip_model += pulp.lpSum(vars_by_emp[k]) == 1, f"Convexity_{k}"

        print("  [Final MIP] Solving...")
        # 修正: optionsでスレッド指定
        solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=300, gapRel=0.01, options=['threads 4'])
        mip_model.solve(solver)

        elapsed = time.perf_counter() - t_start
        self.stats['time_final_mip'] = elapsed # 確実に記録
        
        status = pulp.LpStatus[mip_model.status]
        print(f"  [Final MIP] Status: {status}, Time: {elapsed:.2f}s")
        
        if status in ['Optimal', 'Integer']:
            final_schedule = np.zeros((self.prob.K, self.prob.T), dtype=int)
            total_cost = pulp.value(mip_model.objective)
            
            for v, idx in use_vars:
                if v.varValue and v.varValue > 0.5:
                    emp_id = self.pool[idx]['group_id']
                    final_schedule[emp_id] = self.pool[idx]['schedule']
            
            print(f"  [Final MIP] Objective: {total_cost:,.2f}")
            return final_schedule, total_cost
        else:
            print("  [Final MIP] Failed to find feasible solution.")
            return None, float('inf')
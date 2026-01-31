import time
import heapq
import pulp
import numpy as np
from collections import defaultdict
from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverNeighbor(ColumnGenerationSolver):
    """
    【高速化版: MIP初期解 + 制限付き近傍探索】
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.label = "CG Neighbor (Fast Heuristic)"
        self.backlog_queues = defaultdict(list) 
        self.seen_patterns = set()

    def reset_stats(self):
        super().reset_stats()
        self.stats['count_backlog_push'] = 0
        self.stats['count_feasible_discard'] = 0
        self.stats['count_search_steps'] = 0
        # ★追加: MIP初期解の時間を記録するキー
        self.stats['time_mip_start'] = 0.0

    def reset_for_new_period(self):
        super().reset_for_new_period()
        self.backlog_queues = defaultdict(list)
        self.seen_patterns = set()

    def initialize_rmp(self):
        """MIPで初期解を構築"""
        super().initialize_rmp()
        
        print(f"  [MIP Start] Constructing initial feasible solution via MIP...")
        t_mip_start = time.time()

        # --- MIPモデル構築 ---
        model = pulp.LpProblem("Original_Problem_Initialization", pulp.LpMinimize)
        
        K = self.prob.K
        T = self.prob.T
        employees = self.prob.employees
        demand = self.prob.demand
        
        x = [[pulp.LpVariable(f"x_{k}_{t}", cat=pulp.LpBinary) for t in range(T)] for k in range(K)]
        s = [[pulp.LpVariable(f"s_{k}_{t}", cat=pulp.LpBinary) for t in range(T)] for k in range(K)]
        delta = [pulp.LpVariable(f"delta_{t}", lowBound=0) for t in range(T)]

        obj_terms = []
        for k in range(K):
            emp = employees[k]
            cost_vec = emp['hourly_wage'] + emp['rho']
            for t in range(T):
                obj_terms.append(cost_vec[t] * x[k][t])
        
        for t in range(T):
            obj_terms.append(self.prob.big_m * delta[t])
            
        model += pulp.lpSum(obj_terms)

        # 制約
        for t in range(T):
            model += pulp.lpSum([x[k][t] for k in range(K)]) + delta[t] >= demand[t]

        for k in range(K):
            emp = employees[k]
            L_min = emp.get('L_min', 1)
            L_max = emp.get('L_max', T)
            R_int = emp.get('R_int', 0)
            K_max = emp.get('K_max', 7)

            for t in range(T):
                if t == 0:
                    model += s[k][t] >= x[k][t]
                else:
                    model += s[k][t] >= x[k][t] - x[k][t-1]

            for t in range(T):
                if t + L_min <= T:
                    model += pulp.lpSum([x[k][t+j] for j in range(L_min)]) >= L_min * s[k][t]

            for t in range(T - L_max):
                model += pulp.lpSum([x[k][t+j] for j in range(L_max + 1)]) <= L_max

            if R_int > 0:
                for t in range(T):
                    start_lookback = max(0, t - R_int)
                    if t > 0:
                        lookback_len = t - start_lookback
                        model += pulp.lpSum([x[k][j] for j in range(start_lookback, t)]) <= lookback_len * (1 - s[k][t])

            model += pulp.lpSum([s[k][t] for t in range(T)]) <= K_max

        # --- 求解 ---
        mip_time_limit = 30
        solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=mip_time_limit, gapRel=0.05, threads=4)
        model.solve(solver)
        
        # --- RMPへの登録 ---
        status = pulp.LpStatus[model.status]
        added_count = 0
        if status in ['Optimal', 'Integer']:
            for k in range(K):
                sched = [0] * T
                for t in range(T):
                    val = x[k][t].varValue
                    if val is not None and val > 0.5:
                        sched[t] = 1
                
                idx = self.add_column(k, sched)
                if idx not in self.rmp_indices:
                    self.rmp_indices.append(idx)
                    added_count += 1
            
            # ★追加: 時間の記録
            elapsed = time.time() - t_mip_start
            self.stats['time_mip_start'] = elapsed
            print(f"  [MIP Start] Added {added_count} cols. Time: {elapsed:.2f}s")
        else:
            # ★追加: 失敗時も記録
            elapsed = time.time() - t_mip_start
            self.stats['time_mip_start'] = elapsed
            print(f"  [MIP Start] Failed (Status: {status}). Time: {elapsed:.2f}s")

    def pricing(self, pi, sigma):
        # 変更なし
        t_start = time.perf_counter()
        start_queue_count = self.stats['count_backlog_push']
        
        self._generate_neighbors_from_rmp(pi, sigma)
        
        total_added_to_rmp = 0
        SEARCH_BUDGET = 2000 
        
        for k in range(self.prob.K):
            steps = 0
            while self.backlog_queues[k]:
                if steps >= SEARCH_BUDGET: break
                rc, eid, sched_tuple, cost = heapq.heappop(self.backlog_queues[k])
                steps += 1
                emp = self.prob.employees[eid]
                is_feas = self.is_feasible(emp, sched_tuple)
                if is_feas and rc < -1e-5:
                    if (eid, sched_tuple) not in self.pattern_to_id:
                        new_id = len(self.pool)
                        self.pool.append({'id': new_id, 'group_id': eid, 'schedule': list(sched_tuple), 'cost': cost})
                        self.pattern_to_id[(eid, sched_tuple)] = new_id
                    col_id = self.pattern_to_id[(eid, sched_tuple)]
                    if col_id not in self.rmp_indices:
                        self.rmp_indices.append(col_id)
                        total_added_to_rmp += 1
                        break 
                new_candidates = []
                base_schedule = np.array(sched_tuple)
                wage_vec = emp['hourly_wage'] + emp['rho']
                self._generate_neighbors_for_schedule(base_schedule, eid, pi, sigma, wage_vec, new_candidates)
                for cand in new_candidates: self._push_to_queue(cand)
            self.stats['count_search_steps'] += steps

        self.stats['time_pool'] += (time.perf_counter() - t_start)
        queue_added_this_iter = self.stats['count_backlog_push'] - start_queue_count

        if total_added_to_rmp > 0:
            self.stats['count_pool_hit'] += total_added_to_rmp
            return total_added_to_rmp, queue_added_this_iter
        if queue_added_this_iter > 0:
            return 0, queue_added_this_iter
        return 0, 0

    # 以下のヘルパーメソッドは変更なし
    def _generate_neighbors_from_rmp(self, pi, sigma):
        for idx in self.rmp_indices:
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
        padded = np.concatenate(([0], base_schedule, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) == 0:
            l_min = emp.get('L_min', 1)
            for s in range(0, T - l_min + 1, 4): 
                new_sched = np.zeros(T, dtype=int)
                new_sched[s : s + l_min] = 1
                self._add_candidate_lazy(new_sched, emp_id, pi, sigma, wage_vec, candidates)
        for s, e in zip(starts, ends):
            length = e - s
            for delta in [-1, 1]:
                new_s = s + delta
                if 0 <= new_s < e:
                    new_sched = base_schedule.copy()
                    if delta == -1: new_sched[new_s] = 1
                    else:           new_sched[s] = 0
                    self._add_candidate_lazy(new_sched, emp_id, pi, sigma, wage_vec, candidates)
                new_e = e + delta
                if s < new_e <= T:
                    new_sched = base_schedule.copy()
                    if delta == 1: new_sched[e] = 1
                    else:          new_sched[e-1] = 0
                    self._add_candidate_lazy(new_sched, emp_id, pi, sigma, wage_vec, candidates)
            for delta in [-2, -1, 1, 2]:
                new_s = s + delta
                new_e = e + delta
                if 0 <= new_s and new_e <= T:
                    new_sched = base_schedule.copy()
                    new_sched[s:e] = 0
                    new_sched[new_s:new_e] = 1
                    self._add_candidate_lazy(new_sched, emp_id, pi, sigma, wage_vec, candidates)
            stride = 6
            for new_s in range(0, T - length + 1, stride):
                if abs(new_s - s) < stride: continue
                new_e = new_s + length
                new_sched = base_schedule.copy()
                new_sched[s:e] = 0
                new_sched[new_s:new_e] = 1
                self._add_candidate_lazy(new_sched, emp_id, pi, sigma, wage_vec, candidates)

    def _add_candidate_lazy(self, new_sched_arr, emp_id, pi, sigma, wage_vec, candidates):
        sched_tuple = tuple(new_sched_arr.tolist())
        if (emp_id, sched_tuple) in self.seen_patterns: return
        cost_n = np.dot(new_sched_arr, wage_vec)
        rc = cost_n - np.dot(pi, new_sched_arr) - sigma[emp_id]
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
        lengths = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
        if 'L_min' in emp and np.any(lengths < emp['L_min']): return False
        if 'L_max' in emp and np.any(lengths > emp['L_max']): return False
        return True
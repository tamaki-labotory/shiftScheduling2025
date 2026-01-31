import pandas as pd
import pulp
import time
import numpy as np
import networkx as nx
import os
import re
from problem import GraphBuilder

class ColumnGenerationSolver:
    def __init__(self, problem, use_pool=True):
        self.prob = problem
        self.use_pool = use_pool
        self.pool = []
        self.rmp_indices = []
        self.graphs = {}
        self.pattern_to_id = {} 
        self.history = []
        self.final_selected_ids = set() 
        
        self.stats = {
            'time_rmp': 0.0,
            'time_mip': 0.0,
            'time_pool': 0.0,
            'time_graph': 0.0,
            'count_pool_hit': 0,
            'count_graph_new': 0,
            'count_graph_skip': 0,
            'iterations': 0,
            'pool_size': 0,
            'mip_total_columns': 0,
            'mip_filtered_columns': 0,
            'time_first_sol': None, # ★追加
            'time_best_sol': None   # ★追加
        }

    def reset_stats(self):
        # 辞書を再初期化（初期値設定）
        self.stats = {
            'time_rmp': 0.0,
            'time_mip': 0.0,
            'time_pool': 0.0,
            'time_graph': 0.0,
            'count_pool_hit': 0,
            'count_graph_new': 0,
            'count_graph_skip': 0,
            'iterations': 0,
            'pool_size': 0,
            'mip_total_columns': 0,
            'mip_filtered_columns': 0,
            'time_first_sol': None,
            'time_best_sol': None
        }

    def reset_for_new_period(self):
        self.rmp_indices = []
        self.final_selected_ids = set()
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 

    def initialize_rmp(self):
        self.rmp_indices = []
        for k in range(self.prob.K):
            idx_null = self.add_column(k, [0]*self.prob.T)
            if idx_null not in self.rmp_indices:
                self.rmp_indices.append(idx_null)
            
            emp = self.prob.employees[k]
            sched = [0]*self.prob.T
            start_t = 10
            end_t = min(start_t + emp['L_min'], self.prob.T)
            for t in range(start_t, end_t): 
                sched[t] = 1
            idx_simple = self.add_column(k, sched)
            if idx_simple not in self.rmp_indices:
                self.rmp_indices.append(idx_simple)

    def add_column(self, k, schedule):
        sched_tuple = tuple(schedule)
        pattern_key = (k, sched_tuple)
        if pattern_key in self.pattern_to_id:
            return self.pattern_to_id[pattern_key]
        col_id = len(self.pool)
        emp = self.prob.employees[k]
        cost = np.sum(np.array(schedule) * (emp['hourly_wage'] + emp['rho']))
        self.pool.append({'id': col_id, 'group_id': k, 'schedule': schedule, 'cost': cost})
        self.pattern_to_id[pattern_key] = col_id
        return col_id

    def load_pool_from_csv(self, filename):
        if not os.path.exists(filename):
            return
        try:
            df = pd.read_csv(filename)
            loaded_count = 0
            for _, row in df.iterrows():
                k = int(row['emp_id'])
                sched_str = str(row['schedule_pattern'])
                schedule = [int(c) for c in sched_str]
                prev_pool_size = len(self.pool)
                self.add_column(k, schedule)
                if len(self.pool) > prev_pool_size:
                    loaded_count += 1
            print(f"  -> Loaded pool from {filename}: Added {loaded_count} new columns")
        except Exception as e:
            print(f"  [Warning] Failed to load pool from {filename}: {e}")

    def solve_rmp(self, integer=False, mip_time_limit=3600, mip_gap=0.01, log_path=None):
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
            
        base_options = ['randomSeed 42', 'randomCbcSeed 42', 'threads 1']
        
        if integer:
            # 最終MIP用: maxSolutions 1 は削除 (複数の解を見つけてログに残すため)
            if log_path:
                solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=mip_time_limit, gapRel=mip_gap, logPath=log_path, options=base_options)
            else:
                solver = pulp.COIN_CMD(path='cbc', msg=0, timeLimit=mip_time_limit, gapRel=mip_gap, options=base_options)
        else:
            solver = pulp.COIN_CMD(path='cbc', msg=0, options=base_options)

        model.solve(solver)
        elapsed = time.perf_counter() - t_start
        if integer: self.stats['time_mip'] += elapsed
        else: self.stats['time_rmp'] += elapsed

        if model.status != pulp.LpStatusOptimal and model.status != pulp.LpStatusInteger: 
            return None

        if integer:
            self.stats['mip_total_columns'] = len(active_cols)
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            selected_ids = []
            
            for c in active_cols:
                val = x[c['id']].varValue
                if val is not None and val > 0.5:
                    final_schedule[c['group_id']] = c['schedule']
                    selected_ids.append(c['id'])
            
            return pulp.value(model.objective), final_schedule, selected_ids
        else:
            pi = [c.pi for c in cons_d]
            sigma = [c.pi for c in cons_c]
            return pulp.value(model.objective), pi, sigma
        
    def parse_cbc_log(self, log_path):
            trajectory = []
            final_lower_bound = None

            if not os.path.exists(log_path):
                return trajectory, final_lower_bound
            
            with open(log_path, 'r') as f:
                content = f.read()
            
            pattern_sol = re.compile(r"Integer solution of\s+([-\d\.]+)\s+found.*?\(([\d\.]+)\s+seconds\)")
            matches = pattern_sol.findall(content)
            for obj_str, time_str in matches:
                try:
                    t = float(time_str)
                    obj = float(obj_str)
                    trajectory.append((t, obj))
                except ValueError:
                    continue
            trajectory.sort(key=lambda x: x[0])

            pattern_lb = re.compile(r"(?:Lower bound|Best possible):?\s*([-\d\.]+)")
            match_lb = pattern_lb.search(content)
            if match_lb:
                final_lower_bound = float(match_lb.group(1))
            elif trajectory:
                final_lower_bound = trajectory[-1][1]  

            return trajectory, final_lower_bound

    def pricing(self, pi, sigma):
        pool_added_count = 0
        graph_added_count = 0
        
        candidates_by_emp = {k: [] for k in range(self.prob.K)}
        
        t_pool_start = time.perf_counter()
        if self.use_pool:
            for i, col in enumerate(self.pool):
                if i in self.rmp_indices: continue 
                k = col['group_id']
                rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
                if rc < -1e-5:
                    candidates_by_emp[k].append((rc, i))
        self.stats['time_pool'] += (time.perf_counter() - t_pool_start)

        t_graph_start = time.perf_counter()
        for k in range(self.prob.K):
            if candidates_by_emp[k]:
                candidates_by_emp[k].sort(key=lambda x: x[0])
                best_rc, best_idx = candidates_by_emp[k][0]
                if best_idx not in self.rmp_indices:
                    self.rmp_indices.append(best_idx)
                    pool_added_count += 1
                    self.stats['count_pool_hit'] += 1
                continue

            if k not in self.graphs: self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, src, sink = self.graphs[k]
            emp = self.prob.employees[k]
            
            for u, v, d in G.edges(data=True):
                etype = d.get('type')
                if etype in ['work_start', 'work_cont']:
                    t = d['time']
                    w = (emp['hourly_wage'] + emp['rho'][t]) - pi[t]
                    d['weight'] = w
                elif etype == 'start': d['weight'] = -sigma[k]
                elif etype == 'leave': d['weight'] = 0
                else: d['weight'] = 0
            
            try:
                path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                sched = [0]*self.prob.T
                rc_val = 0
                for u, v in zip(path, path[1:]):
                    d = G[u][v]
                    rc_val += d['weight']
                    if d.get('type') in ['work_start', 'work_cont']: sched[d['time']] = 1
                
                if rc_val < -1e-5:
                    idx = self.add_column(k, sched)
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        graph_added_count += 1
                        self.stats['count_graph_new'] += 1
            except nx.NetworkXNoPath:
                pass

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        return pool_added_count, graph_added_count

    def solve(self, max_iter=1000, time_limit=36000, tol=1e-8, patience=10, mip_rc_threshold=1e10, mip_gap=0.0001):
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] 
        prev_obj = float('inf')
        no_improve_iter = 0
        last_pi = None
        last_sigma = None
        
        # --- CG Phase ---
        for i in range(max_iter):
            if time.time() - start_total > time_limit: break
            
            res = self.solve_rmp(integer=False)
            if res is None: break
            obj, pi, sigma = res
            last_pi, last_sigma = pi, sigma
            
            if prev_obj != float('inf'):
                improvement = (prev_obj - obj) / abs(prev_obj + 1e-9)
                if improvement < tol: no_improve_iter += 1
                else: no_improve_iter = 0
            prev_obj = obj

            pool_add, graph_add = self.pricing(pi, sigma)
            total_added = pool_add + graph_add
            
            self.stats['iterations'] += 1
            self.history.append({'iter': i + 1, 'obj': obj, 'pool_hits': pool_add, 'graph_gen': graph_add})
            
            if total_added == 0: break
            if no_improve_iter >= patience: break
        
        # --- Pre-MIP Filtering ---
        if last_pi is not None and last_sigma is not None:
            filtered_indices = []
            removed_count = 0
            for idx in self.rmp_indices:
                col = self.pool[idx]
                k = col['group_id']
                rc = col['cost'] - np.dot(last_pi, col['schedule']) - last_sigma[k]
                if rc <= mip_rc_threshold:
                    filtered_indices.append(idx)
                else:
                    removed_count += 1
            self.rmp_indices = filtered_indices
            self.stats['mip_filtered_columns'] = removed_count

        self.stats['rmp_obj_lp'] = prev_obj
        
        # --- Final MIP Phase ---
        # CGフェーズ終了時点の時刻を記録（ここまでの経過時間がオフセットになる）
        time_until_mip = time.time() - start_total
        
        timestamp = int(time.time())
        log_file = f"cbc_mip_log_{timestamp}.txt"
        
        # 残り時間を計算してMIPに渡す
        remaining_time = max(1.0, time_limit - time_until_mip)
        
        res_mip = self.solve_rmp(integer=True, mip_time_limit=remaining_time, log_path=log_file, mip_gap=mip_gap)
        
        # ログ解析
        trajectory, lb = self.parse_cbc_log(log_file)
        self.stats['mip_trajectory'] = trajectory
        self.stats['mip_lower_bound'] = lb
        
        if os.path.exists(log_file):
            os.remove(log_file)
            
        if res_mip: 
            final_obj, final_schedule, selected_ids = res_mip
            self.final_selected_ids = set(selected_ids)
            
            # ★追加: 経過時間の計算 (CG時間 + MIP内時間)
            if trajectory:
                # trajectory = [(time_in_mip, obj), ...]
                first_mip_time = trajectory[0][0]
                best_mip_time = trajectory[-1][0]
                self.stats['time_first_sol'] = time_until_mip + first_mip_time
                self.stats['time_best_sol'] = time_until_mip + best_mip_time
            else:
                # ログから取れなかったが解はある場合 (即座に終わった場合など)
                # 全体のelapsedを使う
                current_total = time.time() - start_total
                self.stats['time_first_sol'] = current_total
                self.stats['time_best_sol'] = current_total
        else:
            final_obj = 0.0
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            self.final_selected_ids = set()
            self.stats['time_first_sol'] = None
            self.stats['time_best_sol'] = None
            
        self.stats['pool_size'] = len(self.pool)
        return final_obj, time.time() - start_total, self.stats, final_schedule

    def save_pool_to_csv(self, filename):
        final_mip_indices_set = set(self.rmp_indices)
        data = []
        for i, col in enumerate(self.pool):
            emp = self.prob.employees[col['group_id']]
            sched_str = "".join(map(str, map(int, col['schedule'])))
            is_in_mip = 1 if i in final_mip_indices_set else 0
            is_selected = 1 if col['id'] in self.final_selected_ids else 0
            
            data.append({
                'col_id': col['id'],
                'emp_id': col['group_id'],
                'emp_type': emp['type'],
                'cost': col['cost'],
                'schedule_pattern': sched_str,
                'in_final_mip': is_in_mip,
                'is_selected': is_selected
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"  -> Pool saved to: {filename}")
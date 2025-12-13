import pulp
import time
import numpy as np
import networkx as nx
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
        
        self.stats = {
            'time_rmp_lp': 0.0,
            'time_rmp_mip': 0.0,
            'time_pool': 0.0,
            'time_graph': 0.0,
            'count_pool_hit': 0,
            'count_graph_new': 0,
            'count_graph_skip': 0,
            'iterations': 0,
            'pool_size': 0,
            'mip_total_columns': 0 # ★追加: 最終MIPで使われた列数
        }

    def reset_stats(self):
        self.stats = {k: 0 for k in self.stats}

    def reset_for_new_period(self):
        """期間更新時の処理（ベースクラスはプールリセット）"""
        self.rmp_indices = []
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        # ベースクラスでは、use_pool=Trueでもプール掃除は行わず維持するだけ

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
        if integer: self.stats['time_rmp_mip'] += elapsed
        else: self.stats['time_rmp_lp'] += elapsed

        if model.status != pulp.LpStatusOptimal: return None

        if integer:
            # ★追加: MIP決定変数の総数を記録
            self.stats['mip_total_columns'] = len(active_cols)
            
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
        pool_added_count = 0
        graph_added_count = 0
        t_pool_start = time.perf_counter()
        
        candidates = []
        if self.use_pool:
            for i, col in enumerate(self.pool):
                if i in self.rmp_indices: continue 
                k = col['group_id']
                rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
                if rc < -1e-5: candidates.append((rc, i))
        
        candidates.sort(key=lambda x: x[0])
        limit_add = self.prob.K * 2 
        for rc, i in candidates[:limit_add]:
            self.rmp_indices.append(i)
            pool_added_count += 1
            self.stats['count_pool_hit'] += 1
        
        self.stats['time_pool'] += (time.perf_counter() - t_pool_start)
        if self.use_pool and pool_added_count > 5:
            self.stats['count_graph_skip'] += self.prob.K 
            return pool_added_count, 0

        t_graph_start = time.perf_counter()
        for k in range(self.prob.K):
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
            except nx.NetworkXNoPath: pass

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        return pool_added_count, graph_added_count

    def solve(self, max_iter=50, time_limit=300, tol=1e-4, patience=3):
        """
        :param max_iter: 最大反復回数
        :param time_limit: 制限時間(秒)
        :param tol: 収束判定の許容誤差率 (例: 1e-4 = 0.01% の改善未満なら停滞とみなす)
        :param patience: 停滞を何回許容するか (例: 3回連続で改善が小さければ終了)
        """
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] 
        
        # 収束判定用の変数
        prev_obj = float('inf')
        no_improve_iter = 0
        
        for i in range(max_iter):
            # 1. 時間切れチェック
            if time.time() - start_total > time_limit:
                print(f"Stopping: Time limit ({time_limit}s) reached.")
                break
                
            # 2. RMP緩和問題を解く
            res = self.solve_rmp(integer=False)
            if res is None: break
            obj, pi, sigma = res
            
            # --- 収束判定ロジック (ここを追加) ---
            # 改善率の計算: (前回 - 今回) / 前回
            # ※初回(prev_obj=inf)はスキップ
            if prev_obj != float('inf'):
                improvement = (prev_obj - obj) / abs(prev_obj + 1e-9)
                
                if improvement < tol:
                    no_improve_iter += 1
                else:
                    no_improve_iter = 0 # 改善したらカウンタリセット
            
            prev_obj = obj
            # -----------------------------------

            # 3. Pricing（列生成）
            pool_add, graph_add = self.pricing(pi, sigma)
            total_added = pool_add + graph_add
            
            self.stats['iterations'] += 1
            
            self.history.append({
                'iter': i + 1,
                'obj': obj,
                'pool_hits': pool_add,
                'graph_gen': graph_add
            })
            
            # 4. 終了条件のチェック
            
            # A. 完全収束: 負のReduced Costを持つ列が見つからない
            if total_added == 0:
                print(f"Converged: Optimal LP solution found at iter {i+1}.")
                break
            
            # B. 早期打ち切り: 改善が停滞している
            if no_improve_iter >= patience:
                print(f"Stopping: Objective improvement stalled for {patience} iters (tol={tol}).")
                break
            
        # 5. 最後に整数計画として解く
        res_mip = self.solve_rmp(integer=True)
        if res_mip:
            final_obj, final_schedule = res_mip
        else:
            final_obj = 0.0
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            
        self.stats['pool_size'] = len(self.pool)
        
        return final_obj, time.time() - start_total, self.stats, final_schedule
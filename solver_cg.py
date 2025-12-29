import pandas as pd
import pulp
import time
import numpy as np
import networkx as nx
import os  # 追加
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
            'mip_filtered_columns': 0 
        }

    # ... (既存の reset_stats, reset_for_new_period, initialize_rmp, add_column はそのまま) ...
    
    def reset_stats(self):
        self.stats = {k: 0 for k in self.stats}

    def reset_for_new_period(self):
        self.rmp_indices = []
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

    # === ★ここが追加・変更箇所です★ ===
    def load_pool_from_csv(self, filename):
        """
        保存されたCSVファイルからプールを復元・追加する。
        既にプールにあるパターンは add_column 内で重複チェックされるため、単純に追加呼び出しでOK。
        """
        if not os.path.exists(filename):
            # ファイルが無い場合は何もしない（初回実行時など）
            return

        try:
            df = pd.read_csv(filename)
            loaded_count = 0
            
            for _, row in df.iterrows():
                k = int(row['emp_id'])
                # schedule_pattern は "001110..." という文字列で保存されている前提
                sched_str = str(row['schedule_pattern'])
                
                # 文字列を整数のリストに変換
                schedule = [int(c) for c in sched_str]
                
                # 現在のプールの長さを確認（新規追加判定用）
                prev_pool_size = len(self.pool)
                
                # 列を追加 (重複していれば既存IDが返る)
                self.add_column(k, schedule)
                
                if len(self.pool) > prev_pool_size:
                    loaded_count += 1
            
            print(f"  -> Loaded pool from {filename}: Added {loaded_count} new columns (Total pool: {len(self.pool)})")
            
        except Exception as e:
            print(f"  [Warning] Failed to load pool from {filename}: {e}")

    # ... (以下の solve_rmp, pricing, solve, save_pool_to_csv はそのまま) ...
    
    def solve_rmp(self, integer=False, mip_time_limit=30, mip_gap=0.05, log_path=None):
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
            
        if integer:
            # log_pathが指定されている場合はログを出力する
            if log_path:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit, gapRel=mip_gap, logPath=log_path)
            else:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit, gapRel=mip_gap)
        else:
            solver = pulp.PULP_CBC_CMD(msg=0)
        
        model.solve(solver)
        elapsed = time.perf_counter() - t_start
        if integer: self.stats['time_mip'] += elapsed
        else: self.stats['time_rmp'] += elapsed

        if model.status != pulp.LpStatusOptimal: return None

        if integer:
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
        
    def parse_cbc_log(self, log_path):
        """CBCのログから(経過時間, 目的関数値)のリストを抽出する"""
        trajectory = []
        if not os.path.exists(log_path):
            return trajectory
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # パターン: "Integer solution of 12345 found after ... (0.12 seconds)"
        # ※ CBCのバージョンにより多少異なる場合がありますが、標準的な形式に対応
        pattern = re.compile(r"Integer solution of\s+([-\d\.]+)\s+found.*?\(([\d\.]+)\s+seconds\)")
        
        matches = pattern.findall(content)
        for obj_str, time_str in matches:
            try:
                t = float(time_str)
                obj = float(obj_str)
                trajectory.append((t, obj))
            except ValueError:
                continue
        
        # 時間順にソート
        trajectory.sort(key=lambda x: x[0])
        return trajectory

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

    def solve(self, max_iter=50, time_limit=300, tol=1e-4, patience=3, mip_rc_threshold=500.0, mip_gap=0.01):
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] 
        prev_obj = float('inf')
        no_improve_iter = 0
        last_pi = None
        last_sigma = None
        
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

        # prev_obj には最後のRMP(LP)の目的関数値が入っています
        self.stats['rmp_obj_lp'] = prev_obj
        
        # 最終的なMIPを解く部分でログを取得する
        timestamp = int(time.time())
        log_file = f"cbc_mip_log_{timestamp}.txt"
        
        res_mip = self.solve_rmp(integer=True, mip_time_limit=3600, log_path=log_file, mip_gap=mip_gap) # 時間制限等は適宜調整
        
        # ログを解析してstatsに保存
        self.stats['mip_trajectory'] = self.parse_cbc_log(log_file)
        
        # 一時ファイルを削除（残したい場合はコメントアウト）
        if os.path.exists(log_file):
            os.remove(log_file)
        if res_mip: final_obj, final_schedule = res_mip
        else:
            final_obj = 0.0
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            
        self.stats['pool_size'] = len(self.pool)
        return final_obj, time.time() - start_total, self.stats, final_schedule
    
    def save_pool_to_csv(self, filename):
        data = []
        for col in self.pool:
            emp = self.prob.employees[col['group_id']]
            sched_str = "".join(map(str, map(int, col['schedule'])))
            data.append({
                'col_id': col['id'],
                'emp_id': col['group_id'],
                'emp_type': emp['type'],
                'cost': col['cost'],
                'schedule_pattern': sched_str,
            })
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"  -> Pool saved to: {filename} (Total {len(df)} columns)")
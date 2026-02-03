from solver_cg import ColumnGenerationSolver
from problem import GraphBuilder
import time
import numpy as np
import os
import networkx as nx

class ColumnGenerationSolverWithPruning(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=8000):
        # 親クラスの初期化
        super().__init__(problem, use_pool)
        
        self.pool_cleanup_threshold = pool_cleanup_threshold
        
        # --- 実験設定パラメータ ---
        self.cg_stall_limit = 3         # CG終了条件: 3回停滞
        self.ip_selection_ratio = 0.20  # IP列選定: 上位20%
        self.mip_time_limit = 120        # MIP制限時間: 30秒
        
        # 新規追加: RMP除外の閾値
        self.rmp_pruning_threshold = 100000.0 

        # Iterationごとの列追加上限
        self.max_pool_cols_per_iter = 5 # プールからは最大3つ
        self.max_graph_cols_per_iter = 1 # グラフからは最大1つ

    def reset_for_new_period(self):
        """
        新しい週の開始時に呼び出される。
        プールサイズが肥大化しすぎていないかチェックし、必要なら掃除する。
        """
        self.rmp_indices = []
        self.final_selected_ids = set()
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        else:
            if len(self.pool) > self.pool_cleanup_threshold:
                self.cleanup_pool(pi=None, sigma=None)

    def cleanup_pool(self, pi=None, sigma=None):
        """
        プール全体の列数が threshold を超えた場合、古い列や質の悪い列を削除する。
        """
        active_indices_set = set(self.rmp_indices)
        total_cols = len(self.pool)
        
        if total_cols <= self.pool_cleanup_threshold:
            return

        new_pool = []
        new_pattern_to_id = {}
        new_rmp_indices = []
        
        keep_recent_count = max(1, int(self.pool_cleanup_threshold * 0.5))
        cutoff_index = total_cols - keep_recent_count
        
        for old_idx, col in enumerate(self.pool):
            should_keep = False
            
            # A. 現在RMPに含まれている
            if col['id'] in active_indices_set:
                should_keep = True
            # B. 最近追加された
            elif old_idx >= cutoff_index:
                should_keep = True
            # C. 被約費用が良い
            elif pi is not None and sigma is not None:
                k = col['group_id']
                sched = np.array(col['schedule'])
                val = np.dot(pi, sched)
                rc = col['cost'] - val - sigma[k]
                if rc < 1e-5:
                    should_keep = True
            
            if should_keep:
                self._add_col_to_new_lists(col, new_pool, new_pattern_to_id, new_rmp_indices, active_indices_set)
            
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        self.rmp_indices = new_rmp_indices
        
        # print(f"DEBUG: Pool cleanup done. New total size: {len(self.pool)}")

    def _add_col_to_new_lists(self, col, new_pool, new_pattern_to_id, new_rmp_indices, active_indices_set):
        was_active = (col['id'] in active_indices_set)
        new_id = len(new_pool)
        col['id'] = new_id
        new_pool.append(col)
        pat_key = (col['group_id'], tuple(col['schedule']))
        new_pattern_to_id[pat_key] = new_id
        if was_active:
            new_rmp_indices.append(new_id)

    def prune_rmp_by_rc(self, pi, sigma):
        """
        【追加機能】
        RMPに含まれている列のうち、被約費用(RC)が閾値(10^5)を超えるものを
        RMPのインデックスリストから削除する。
        """
        original_count = len(self.rmp_indices)
        new_indices = []
        
        for idx in self.rmp_indices:
            col = self.pool[idx]
            k = col['group_id']
            # 被約費用の計算: RC = Cost - (pi * schedule) - sigma
            sched = np.array(col['schedule'])
            val = np.dot(pi, sched)
            rc = col['cost'] - val - sigma[k]
            
            # 閾値以下のものだけ残す
            if rc <= self.rmp_pruning_threshold:
                new_indices.append(idx)
        
        self.rmp_indices = new_indices
        removed_count = original_count - len(self.rmp_indices)
        # 必要であればログ出力
        # if removed_count > 0:
        #     print(f"  [Pruning] Removed {removed_count} cols with RC > {self.rmp_pruning_threshold}")

    def pricing(self, pi, sigma):
        """
        Pricing問題を解くメソッド（オーバーライド）。
        従業員ごとに「プール内探索」を行い、被約費用が負の列が見つかればそれを追加して終了。
        見つからなかった場合のみ「グラフ探索」を行う、二段階構成。
        """
        pool_added_count = 0
        graph_added_count = 0
        
        # 各従業員のプール内候補列を計算
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
            found_in_pool = False
            
            # -----------------------------------------------------------
            # Step 1: Pool Check
            # プール内に有望な列があるか確認
            # -----------------------------------------------------------
            if candidates_by_emp[k]:
                # 被約費用の小さい順（昇順）にソート
                candidates_by_emp[k].sort(key=lambda x: x[0])
                
                added_count = 0
                for _, idx in candidates_by_emp[k]:
                    if added_count >= self.max_pool_cols_per_iter:
                        break
                    
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        pool_added_count += 1
                        self.stats['count_pool_hit'] += 1
                        added_count += 1
                
                # プールから1つでも追加できれば、この従業員のグラフ探索はスキップ
                if added_count > 0:
                    found_in_pool = True

            # -----------------------------------------------------------
            # Step 2: Graph Search (Conditional)
            # プールで見つからなかった場合のみ実行
            # -----------------------------------------------------------
            if not found_in_pool:
                # グラフ構築（キャッシュ利用）
                if k not in self.graphs: self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
                G, src, sink = self.graphs[k]
                emp = self.prob.employees[k]
                
                # エッジ重みの更新
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
                    # 最短路探索（1つだけ）
                    path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                    
                    sched = [0]*self.prob.T
                    rc_val = 0
                    for u, v in zip(path, path[1:]):
                        d = G[u][v]
                        rc_val += d['weight']
                        if d.get('type') in ['work_start', 'work_cont']: sched[d['time']] = 1
                    
                    # 負の被約費用を持つ場合のみ追加
                    if rc_val < -1e-5:
                        idx = self.add_column(k, sched)
                        if idx not in self.rmp_indices:
                            self.rmp_indices.append(idx)
                            graph_added_count += 1
                            self.stats['count_graph_new'] += 1
                
                except nx.NetworkXNoPath:
                    pass
                except Exception as e:
                    print(f"Warning: Pricing failed for emp {k}: {e}")

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        return pool_added_count, graph_added_count

    def solve(self, max_iter=1000, time_limit=3600, tol=1e-8, patience=10, mip_rc_threshold=1e10, mip_gap=0.0001):
        """
        メインのsolveメソッド
        """
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = []
        
        print("START: Column Generation Phase")
        obj_history = []
        
        current_pi = None
        current_sigma = None
        
        for i in range(max_iter):
            if time.time() - start_total > time_limit:
                print("INFO: Time limit reached during CG.")
                break
                
            res = self.solve_rmp(integer=False)
            if res is None: break
            current_obj, pi, sigma = res
            current_pi, current_sigma = pi, sigma

            # --- RMP Pruning (RC Check) ---
            # ここで被約費用が大きすぎる列をRMPから除外する
            self.prune_rmp_by_rc(pi, sigma)
            
            # --- Stall Check ---
            obj_history.append(current_obj)
            if len(obj_history) >= self.cg_stall_limit:
                recent = obj_history[-self.cg_stall_limit:]
                if max(recent) - min(recent) < 1e-6:
                    print(f"INFO: CG Stalled. Stopping CG.")
                    break

            # Pricing (Overrideしたメソッドが呼ばれる)
            pool_add, graph_add = self.pricing(pi, sigma)
            
            self.stats['iterations'] += 1
            self.history.append({'iter': i + 1, 'obj': current_obj, 'pool_hits': pool_add, 'graph_gen': graph_add})
            
            if pool_add + graph_add == 0:
                print("INFO: No new columns found.")
                break
                
            # --- Cleanup (Global) ---
            if self.use_pool:
                if len(self.pool) > self.pool_cleanup_threshold:
                    self.cleanup_pool(pi, sigma)
        
        if obj_history:
            self.stats['rmp_obj_lp'] = obj_history[-1]
                
        # --- MIP Phase ---
        print("START: MIP Phase with Column Selection")
        
        if current_pi is not None:
            pool_with_rc = []
            for col in self.pool:
                sched = np.array(col['schedule'])
                val = np.dot(current_pi, sched)
                rc = col['cost'] - val - current_sigma[col['group_id']]
                pool_with_rc.append((rc, col))
            
            pool_with_rc.sort(key=lambda x: x[0])
            
            # 上位 N% を選定
            cutoff_count = int(len(pool_with_rc) * self.ip_selection_ratio)
            selected_columns = [item[1] for item in pool_with_rc[:cutoff_count]]
            
            self.rmp_indices = [col['id'] for col in selected_columns]
            self.stats['mip_total_columns'] = len(self.rmp_indices)
            self.stats['mip_filtered_columns'] = len(self.pool) - len(self.rmp_indices)
            print(f"INFO: Selected {len(selected_columns)} columns for MIP.")

        remaining_total = max(1.0, time_limit - (time.time() - start_total))
        actual_mip_limit = min(remaining_total, self.mip_time_limit)
        
        print(f"INFO: Solving MIP with time limit = {actual_mip_limit:.1f}s")
        
        timestamp = int(time.time())
        log_file = f"cbc_mip_log_pruning_{timestamp}.txt"
        
        res_mip = self.solve_rmp(integer=True, mip_time_limit=actual_mip_limit, log_path=log_file, mip_gap=mip_gap)
        
        trajectory, lb = self.parse_cbc_log(log_file)
        self.stats['mip_trajectory'] = trajectory
        self.stats['mip_lower_bound'] = lb
        
        if os.path.exists(log_file):
            os.remove(log_file)
            
        final_schedule = None
        final_obj = 0.0
        if res_mip:
            final_obj, final_schedule, selected_ids = res_mip
            self.final_selected_ids = set(selected_ids)
            if trajectory:
                self.stats['time_first_sol'] = (time.time() - start_total - actual_mip_limit) + trajectory[0][0]
            else:
                self.stats['time_first_sol'] = time.time() - start_total
        
        self.stats['pool_size'] = len(self.pool)
        return final_obj, time.time() - start_total, self.stats, final_schedule
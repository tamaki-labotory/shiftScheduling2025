# solver_cg_smart.py (New File)
import numpy as np
from collections import Counter
from solver_cg import ColumnGenerationSolver
from problem import GraphBuilder

class ColumnGenerationSolverSmart(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_max_size=2000, historical_patterns=None):
        """
        historical_patterns: {time_index: frequency, ...} 形式の辞書。
                             過去の運用で頻出した時間帯情報。
        """
        super().__init__(problem, use_pool)
        self.pool_max_size = pool_max_size
        self.historical_freq = historical_patterns if historical_patterns else {}
        self.batch_penalty_alpha = 50.0  # バッチペナルティの強さ
        
        # 初期化時にグラフにペナルティを埋め込む
        self._apply_batch_graph_adjustments()

    def _apply_batch_graph_adjustments(self):
        """
        過去の履歴に基づき、頻出パターンに対応するアークの重みを
        あらかじめ増加させておく（Static Penalty）。
        """
        self.graphs = {} # リセット
        
        # 頻度データの最大値で正規化
        max_freq = max(self.historical_freq.values()) if self.historical_freq else 1
        
        for k in range(self.prob.K):
            self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, _, _ = self.graphs[k]
            
            for u, v, d in G.edges(data=True):
                # 勤務時間帯アークに対してペナルティを付与
                if d.get('type') in ['work_start', 'work_cont']:
                    t = d['time']
                    freq = self.historical_freq.get(t, 0)
                    penalty = self.batch_penalty_alpha * (freq / max_freq)
                    d['static_penalty'] = penalty
                else:
                    d['static_penalty'] = 0.0

    def reset_for_new_period(self):
        """
        週替わりなどで呼ばれるリセット処理。
        スマート掃除を行うためにオーバーライドするが、
        ここでは単にRMPインデックスをクリアし、プールサイズ超過なら掃除する。
        """
        self.rmp_indices = []
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {}
        else:
            # グラフの再構築（ペナルティ情報を維持するため、完全消去はしない方が良いが、
            # historical_patternsが更新されるなら再構築が必要）
            # ここでは簡易的に、プールサイズチェックだけ行う
            if len(self.pool) > self.pool_max_size:
                # 注: 掃除にはpi, sigmaが必要だが、リセット時点では手に入らない。
                # そのため、solve() のループ内や終了時に掃除する戦略をとる。
                # ここでは「強制的に古い順」ではなく、何もしない（solve内で掃除）か、
                # 簡易的なランダム削除などが考えられる。今回はsolve内で行うためパス。
                pass

    def pricing(self, pi, sigma):
        """
        グラフ探索時に static_penalty を考慮するようにオーバーライド。
        親クラスの pricing ロジックをコピーして、重み更新部分だけ変えるのが確実。
        """
        # --- 1. Pool Pricing (親クラスと同じ) ---
        pool_added_count = 0
        graph_added_count = 0
        import time
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

        # --- 2. Graph Pricing with Static Penalty (修正) ---
        t_graph_start = time.perf_counter()
        import networkx as nx
        
        for k in range(self.prob.K):
            if k not in self.graphs: 
                # 通常ここには来ないはずだが、念のため再構築
                self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
                
            G, src, sink = self.graphs[k]
            emp = self.prob.employees[k]
            
            # ★重み更新ロジックの変更点★
            for u, v, d in G.edges(data=True):
                etype = d.get('type')
                if etype in ['work_start', 'work_cont']:
                    t = d['time']
                    # static_penalty を加算して探索多様性を高める
                    w = (emp['hourly_wage'] + emp['rho'][t]) - pi[t] + d.get('static_penalty', 0.0)
                    d['weight'] = w
                elif etype == 'start': d['weight'] = -sigma[k]
                elif etype == 'leave': d['weight'] = 0
                else: d['weight'] = 0
            
            try:
                path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                sched = [0]*self.prob.T
                
                # ペナルティ込みで経路は見つけたが、登録するコストは「真のコスト」であるべき
                # 列生成の収束条件(RC < 0)は真のコストで判定するのが一般的だが、
                # ここでは多様性重視のため「ペナルティ込みRC < 0」で見つかった列を、
                # そのまま追加しても問題ない（真のRCが正ならLPで使われないだけ）。
                # ただし、厳密な収束判定のためには、最後にペナルティなしチェックが必要かも知れない。
                # 今回は簡略化のため、見つかった列はとりあえず追加する。
                
                for u, v in zip(path, path[1:]):
                    d = G[u][v]
                    if d.get('type') in ['work_start', 'work_cont']: sched[d['time']] = 1
                
                # 重複登録チェックなどは add_column 内で行われる
                # RCの再計算（真の値で判定したい場合）
                true_cost = np.sum(np.array(sched) * (emp['hourly_wage'] + emp['rho']))
                true_rc = true_cost - np.dot(pi, sched) - sigma[k]
                
                if true_rc < -1e-5:
                    idx = self.add_column(k, sched)
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        graph_added_count += 1
                        self.stats['count_graph_new'] += 1
                        
            except nx.NetworkXNoPath: pass

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        return pool_added_count, graph_added_count

    def solve(self, max_iter=50, time_limit=300, tol=1e-4, patience=3, mip_rc_threshold=500.0):
        """
        親クラスの solve を呼び出すが、各反復後にプール掃除をするフックを入れたい。
        しかし親クラスの solve はループが閉じているため、オーバーライドして実装し直す。
        """
        import time
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] 
        prev_obj = float('inf')
        no_improve_iter = 0
        last_pi, last_sigma = None, None
        
        for i in range(max_iter):
            if time.time() - start_total > time_limit: break
            
            # 1. RMP
            res = self.solve_rmp(integer=False)
            if res is None: break
            obj, pi, sigma = res
            last_pi, last_sigma = pi, sigma
            
            # ★ここでスマート掃除を実行（プールサイズオーバー時）
            if self.use_pool and len(self.pool) > self.pool_max_size:
                self.cleanup_pool_smart(pi, sigma)

            # 収束判定
            if prev_obj != float('inf'):
                improvement = (prev_obj - obj) / abs(prev_obj + 1e-9)
                if improvement < tol: no_improve_iter += 1
                else: no_improve_iter = 0
            prev_obj = obj

            # 2. Pricing
            pool_add, graph_add = self.pricing(pi, sigma)
            total_added = pool_add + graph_add
            
            self.stats['iterations'] += 1
            self.history.append({'iter': i + 1, 'obj': obj, 'pool_hits': pool_add, 'graph_gen': graph_add})
            
            if total_added == 0: break
            if no_improve_iter >= patience: break

        # MIP前のフィルタリング
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
        
        # Final MIP
        res_mip = self.solve_rmp(integer=True)
        if res_mip: final_obj, final_schedule = res_mip
        else:
            final_obj = 0.0
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            
        self.stats['pool_size'] = len(self.pool)
        return final_obj, time.time() - start_total, self.stats, final_schedule

    def cleanup_pool_smart(self, pi, sigma):
        """
        多様性と有用性に基づく削除
        """
        active_indices = set(self.rmp_indices)
        
        # パターン特徴量（開始時間、長さ）の頻度カウント
        pattern_signatures = []
        for col in self.pool:
            sched = col['schedule']
            if 1 in sched:
                start = tuple(sched).index(1)
                duration = sum(sched)
                sig = (start, duration)
            else:
                sig = (-1, 0)
            pattern_signatures.append(sig)
        sig_counts = Counter(pattern_signatures)
        
        candidates = []
        for i, col in enumerate(self.pool):
            if i in active_indices: continue
            
            k = col['group_id']
            rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
            redundancy = sig_counts[pattern_signatures[i]]
            
            # スコア計算: 高いほど削除対象
            # RCが高い（悪い）ほど削除したい
            # Redundancyが高い（ありふれている）ほど削除したい
            score = rc + (redundancy * 0.5)
            
            candidates.append((score, i))
            
        # スコアが高い順に削除
        candidates.sort(key=lambda x: x[0], reverse=True)
        num_remove = len(self.pool) - self.pool_max_size
        if num_remove <= 0: return
        
        remove_indices = set(idx for score, idx in candidates[:num_remove])
        
        new_pool = []
        new_rmp_indices = []
        new_pattern_to_id = {}
        
        for old_idx, col in enumerate(self.pool):
            if old_idx not in remove_indices:
                new_id = len(new_pool)
                if old_idx in self.rmp_indices: # 再チェック（activeは削除リストに入らないはずだが）
                    new_rmp_indices.append(new_id)
                col['id'] = new_id
                new_pool.append(col)
                new_pattern_to_id[(col['group_id'], tuple(col['schedule']))] = new_id
                
        self.pool = new_pool
        self.rmp_indices = new_rmp_indices
        self.pattern_to_id = new_pattern_to_id
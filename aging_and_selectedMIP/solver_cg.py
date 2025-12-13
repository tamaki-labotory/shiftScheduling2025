import pulp
import time
import numpy as np
import networkx as nx
from problem import GraphBuilder

class ColumnGenerationSolver:
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=2000):
        """
        :param problem: ShiftProblemData インスタンス
        :param use_pool: プール機能を使うかどうかのフラグ
        :param pool_cleanup_threshold: プールサイズがこの値を超えたら掃除を行う
        """
        self.prob = problem
        self.use_pool = use_pool
        self.pool_cleanup_threshold = pool_cleanup_threshold
        
        # 列プール: 辞書のリスト [{'id':0, 'group_id':k, 'schedule':[], 'cost':100, 'active':True}, ...]
        self.pool = []
        
        # RMPで現在使用している列のIDリスト
        self.rmp_indices = []
        
        # グラフキャッシュ
        self.graphs = {}
        
        # 重複チェック用辞書 Key: (group_id, tuple(schedule)), Value: col_id
        self.pattern_to_id = {} 
        
        self.history = []
        
        # 統計情報
        self.stats = {
            'time_rmp_lp': 0.0,
            'time_rmp_mip': 0.0,
            'time_pool': 0.0,
            'time_graph': 0.0,
            'count_pool_hit': 0,
            'count_graph_new': 0,
            'count_graph_skip': 0,
            'iterations': 0,
            'pool_size': 0
        }

    def reset_stats(self):
        """統計情報のリセット"""
        self.stats = {k: 0 for k in self.stats}

    def reset_for_new_period(self):
        """
        新しい期間（Week）に移る際の処理。
        - プールあり(Proposed): プールは保持するが、RMPのインデックスはリセット。
        - プールなし(Standard): 全てリセット。
        """
        self.rmp_indices = []
        if not self.use_pool:
            # Standard: 完全リセット
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        else:
            # Proposed: プール掃除（サイズが大きすぎる場合）
            if len(self.pool) > self.pool_cleanup_threshold:
                self.cleanup_pool()

    def cleanup_pool(self):
        """
        プールのメンテナンスを行うメソッド。
        前の期間で使用されなかった列などを削除し、IDを振り直す。
        """
        # ここでは単純に「現在RMPに入っている列」+「直近で生成された列」などを残す戦略をとる
        # 今回は簡易的に「すべての列」を対象にIDを詰め直す処理だけ実装するが、
        # 必要に応じて「Reduced Costが極端に悪い列」をフィルタリングして削除するロジックをここに入れる
        
        new_pool = []
        new_pattern_to_id = {}
        
        for col in self.pool:
            # ★ここで将来的に「不要な列」をスキップする条件を追加可能
            # 現状はIDの再整理のみを行う（メモリ断片化防止と整合性維持）
            new_id = len(new_pool)
            col['id'] = new_id
            new_pool.append(col)
            
            pat_key = (col['group_id'], tuple(col['schedule']))
            new_pattern_to_id[pat_key] = new_id
            
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        # print(f"DEBUG: Pool cleaned. Size: {len(self.pool)}")

    def initialize_rmp(self):
        """初期解（ダミー列＋基本列）の生成"""
        self.rmp_indices = []
        
        # 全従業員に対して初期列を追加
        for k in range(self.prob.K):
            # 1. Nullシフト（勤務なし）
            idx_null = self.add_column(k, [0]*self.prob.T)
            if idx_null not in self.rmp_indices:
                self.rmp_indices.append(idx_null)
            
            # 2. 初期実行可能解を得やすくするための単純シフト（例: 10時から数時間）
            emp = self.prob.employees[k]
            sched = [0]*self.prob.T
            start_t = 10
            # 必ずL_min以上確保
            duration = emp['L_min']
            end_t = min(start_t + duration, self.prob.T)
            
            for t in range(start_t, end_t): 
                sched[t] = 1
            
            idx_simple = self.add_column(k, sched)
            if idx_simple not in self.rmp_indices:
                self.rmp_indices.append(idx_simple)

    def add_column(self, k, schedule):
        """
        列をプールに追加する。既に存在する場合は既存IDを返す。
        """
        sched_tuple = tuple(schedule)
        pattern_key = (k, sched_tuple)

        # 重複チェック
        if pattern_key in self.pattern_to_id:
            return self.pattern_to_id[pattern_key]

        col_id = len(self.pool)
        emp = self.prob.employees[k]
        
        # コスト計算 (時給 + 該当時間のペナルティrho)
        # ※週が変わるとrhoが変わる可能性がある場合、このcostはsolve時に再計算が必要だが、
        # 今回はProblemData側で固定と仮定するか、必要ならsolveループ内で更新する。
        # ここでは生成時のコストを保持。
        cost = np.sum(np.array(schedule) * (emp['hourly_wage'] + emp['rho']))
        
        new_col = {
            'id': col_id, 
            'group_id': k, 
            'schedule': schedule, 
            'cost': cost
        }
        
        self.pool.append(new_col)
        self.pattern_to_id[pattern_key] = col_id
        
        return col_id

    def solve_rmp(self, integer=False):
        """
        制限主問題(RMP)を解く
        integer=True ならMIPとして解き、最終スケジュールの行列を返す
        integer=False ならLPとして解き、(目的関数値, 双対変数pi, 双対変数sigma) を返す
        """
        t_start = time.perf_counter()
        
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        
        # 現在RMPに含まれている列のみを対象にする
        active_cols = [self.pool[i] for i in self.rmp_indices]
        
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        
        # 変数定義
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=cat) for c in active_cols}
        
        # 需給ギャップ変数 (スラック変数)
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        # 目的関数: 総人件費 + ペナルティ
        model += pulp.lpSum([c['cost'] * x[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * d for d in delta])
        
        # 制約1: 各時間帯の需要充足
        cons_d = []
        for t in range(self.prob.T):
            # sum(schedule[t] * x) + delta >= Demand
            expr = pulp.lpSum([c['schedule'][t] * x[c['id']] for c in active_cols]) + delta[t]
            model += expr >= self.prob.demand[t]
            # 制約オブジェクトを保存（双対変数取得用）
            cons_d.append(model.constraints[list(model.constraints.keys())[-1]])
            
        # 制約2: 各従業員は1つのシフトパターンのみ
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

        if model.status != pulp.LpStatusOptimal:
            # 稀に実行不能になる場合のガード
            return None

        if integer:
            # 最適解の構築
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            for c in active_cols:
                val = x[c['id']].varValue
                if val is not None and val > 0.5:
                    final_schedule[c['group_id']] = c['schedule']
            return pulp.value(model.objective), final_schedule
        else:
            # 双対変数の取得
            pi = [c.pi for c in cons_d]
            sigma = [c.pi for c in cons_c]
            return pulp.value(model.objective), pi, sigma

    def pricing(self, pi, sigma):
        """
        部分問題（Pricing Problem）。
        1. まずプール内を探索 (Pricing over Pool)
        2. 見つからなければグラフ探索 (Pricing over Graph)
        """
        pool_added_count = 0
        graph_added_count = 0
        
        # --- 1. Pool Pricing ---
        t_pool_start = time.perf_counter()
        
        candidates = []
        if self.use_pool:
            # 既存プールの中から、現在の双対変数でReduced Costが負になるものを探す
            for i, col in enumerate(self.pool):
                if i in self.rmp_indices: continue 
                
                k = col['group_id']
                # Reduced Cost = Cost - (pi * schedule) - sigma
                rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
                
                if rc < -1e-5:
                    candidates.append((rc, i))
        
            # RCが良い順にソートして上位を追加
            candidates.sort(key=lambda x: x[0])
            
            # 一度に追加する列数を制限（RMP肥大化防止）
            limit_add = self.prob.K * 2 
            
            for rc, i in candidates[:limit_add]:
                self.rmp_indices.append(i)
                pool_added_count += 1
                self.stats['count_pool_hit'] += 1
        
        self.stats['time_pool'] += (time.perf_counter() - t_pool_start)

        # ★重要: プールから十分な候補が見つかった場合、重いグラフ探索をスキップする（ヒューリスティック）
        # これが「提案手法」の高速化の鍵
        if self.use_pool and pool_added_count > 5:
            self.stats['count_graph_skip'] += self.prob.K 
            return pool_added_count, 0

        # --- 2. Graph Pricing (Exact) ---
        t_graph_start = time.perf_counter()
        
        for k in range(self.prob.K):
            if k not in self.graphs:
                self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, src, sink = self.graphs[k]
            
            emp = self.prob.employees[k]
            
            # エッジ重みの更新
            # w_t = (Wage + Penalty_t) - pi_t
            for u, v, d in G.edges(data=True):
                etype = d.get('type')
                if etype in ['work_start', 'work_cont']:
                    t = d['time']
                    w = (emp['hourly_wage'] + emp['rho'][t]) - pi[t]
                    d['weight'] = w
                elif etype == 'start':
                    d['weight'] = -sigma[k] # 開始エッジに定数項（双対変数sigma）を乗せる
                elif etype == 'leave':
                    d['weight'] = 0
                else:
                    d['weight'] = 0
            
            # 最短路探索 (SPPRCの簡易版)
            try:
                path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                
                # パスからスケジュール復元
                sched = [0]*self.prob.T
                rc_val = 0
                for u, v in zip(path, path[1:]):
                    d = G[u][v]
                    rc_val += d['weight']
                    if d.get('type') in ['work_start', 'work_cont']:
                        sched[d['time']] = 1
                
                # Reduced Costが負なら列生成
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

    def solve(self, max_iter=50, time_limit=300):
        """
        列生成法のメインループ
        """
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        self.history = [] 
        
        for i in range(max_iter):
            # 時間切れチェック
            if time.time() - start_total > time_limit:
                print("Time limit reached.")
                break
                
            # 1. RMP緩和問題を解く
            res = self.solve_rmp(integer=False)
            if res is None: break
            obj, pi, sigma = res
            
            # 2. Pricing（列生成）
            pool_add, graph_add = self.pricing(pi, sigma)
            total_added = pool_add + graph_add
            
            self.stats['iterations'] += 1
            
            self.history.append({
                'iter': i + 1,
                'obj': obj,
                'pool_hits': pool_add,
                'graph_gen': graph_add
            })
            
            # 改善列がなければ終了（最適性規準）
            if total_added == 0:
                break
            
        # 3. 最後に整数計画として解く
        res_mip = self.solve_rmp(integer=True)
        if res_mip:
            final_obj, final_schedule = res_mip
        else:
            final_obj = 0.0
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            
        self.stats['pool_size'] = len(self.pool)
        
        return final_obj, time.time() - start_total, self.stats, final_schedule
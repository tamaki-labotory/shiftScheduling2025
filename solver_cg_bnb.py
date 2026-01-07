import time
import pulp
import numpy as np
import heapq
from solver_cg import ColumnGenerationSolver

class BnBNode:
    """Branch-and-Boundの探索ノード"""
    def __init__(self, node_id, constraints, lower_bound, parent_id=None):
        self.node_id = node_id
        self.constraints = constraints  # List of (k, t, val)
        self.lower_bound = lower_bound
        self.parent_id = parent_id
        self.schedule_prob = None # このノードでの緩和解(確率)

    # heapqで比較するためのメソッド（Lower Boundが小さい順）
    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

class ColumnGenerationBnBSolver(ColumnGenerationSolver):
    """
    完全なBranch-and-Priceソルバー。
    探索木(Tree)を構築し、Best-First Searchでノードを探索します。
    バックトラックが可能で、厳密な整数解を目指します。
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.nodes_explored = 0
        self.best_integer_obj = float('inf')
        self.best_integer_solution = None
        
        # グラフの初期重みをキャッシュ
        self._original_graph_weights = {}

    def solve(self, max_iter=50, time_limit=300, node_limit=100, tol=1e-4):
        start_total = time.time()
        self.reset_stats()
        self.initialize_rmp()
        
        # オリジナルのグラフ重みを保存（復元用）
        self._cache_original_weights()

        # --- Root Node ---
        print("[BnB] Processing Root Node...")
        root_obj, root_sched = self.solve_node([], max_iter, tol)
        
        if root_obj is None:
            print("[BnB] Root is infeasible.")
            return None, 0, self.stats, None

        # ルートがいきなり整数の場合
        if self.is_integer_solution(root_sched):
            print("[BnB] Root solution is already integer.")
            return root_obj, time.time() - start_total, self.stats, np.round(root_sched)

        # 優先度付きキュー (Lower Boundが小さい順に取り出す)
        open_nodes = []
        root_node = BnBNode(0, [], root_obj)
        root_node.schedule_prob = root_sched
        heapq.heappush(open_nodes, root_node)
        
        self.best_integer_obj = float('inf')
        self.best_integer_solution = None
        node_counter = 0

        print(f"[BnB] Start Tree Search (Best-First). Time Limit={time_limit}s")

        while open_nodes:
            # 1. 終了条件チェック
            if time.time() - start_total > time_limit:
                print("[BnB] Time limit reached.")
                break
            if self.nodes_explored >= node_limit:
                print("[BnB] Node limit reached.")
                break
            
            # 2. 最良ノードの取り出し
            current_node = heapq.heappop(open_nodes)
            
            # Pruning (枝刈り): もしこのノードのLBが、すでに見つかった最良整数解より悪ければ捨てる
            if current_node.lower_bound >= self.best_integer_obj - 1e-5:
                continue

            # ルート以外はまだ計算していないので、ここでCGを実行してLBと解を更新
            # (ルートは計算済みだが、実装を簡単にするためキューに入れた後に再度チェックするフローにする)
            # ただし、今回はノード作成時に親のLBを入れているので、正確なLB計算はここで行う必要がある
            if current_node.node_id != 0: # Rootは計算済み
                # ノードの制約を適用してCG実行
                obj_val, sched_prob = self.solve_node(current_node.constraints, max_iter, tol)
                
                # InfeasibleならPrune
                if obj_val is None:
                    continue
                
                current_node.lower_bound = obj_val
                current_node.schedule_prob = sched_prob
                
                # 再度Pruningチェック (CG実行後にLBが上がっている可能性があるため)
                if current_node.lower_bound >= self.best_integer_obj - 1e-5:
                    continue

            self.nodes_explored += 1
            if self.nodes_explored % 5 == 0:
                print(f"  [BnB] Node {self.nodes_explored}: LB={current_node.lower_bound:.2f}, OpenNodes={len(open_nodes)}, BestUB={self.best_integer_obj:.2f}")

            # 3. 整数解チェック
            if self.is_integer_solution(current_node.schedule_prob):
                print(f"  [BnB] Found Integer Solution at Node {current_node.node_id}! Obj={current_node.lower_bound:.2f}")
                if current_node.lower_bound < self.best_integer_obj:
                    self.best_integer_obj = current_node.lower_bound
                    self.best_integer_solution = np.round(current_node.schedule_prob)
                continue # この枝はこれ以上掘る必要なし（葉ノード）

            # 4. Branching (分岐)
            # 最も小数に近い変数を選ぶ
            k, t, val = self.select_branching_variable(current_node.schedule_prob)
            
            # Branch 1: x[k,t] = 0 (休暇固定)
            cons_0 = current_node.constraints + [(k, t, 0)]
            node_0 = BnBNode(node_counter + 1, cons_0, current_node.lower_bound, parent_id=current_node.node_id)
            heapq.heappush(open_nodes, node_0)
            
            # Branch 2: x[k,t] = 1 (勤務固定)
            cons_1 = current_node.constraints + [(k, t, 1)]
            node_1 = BnBNode(node_counter + 2, cons_1, current_node.lower_bound, parent_id=current_node.node_id)
            heapq.heappush(open_nodes, node_1)
            
            node_counter += 2

        # 探索終了
        elapsed = time.time() - start_total
        
        if self.best_integer_solution is not None:
            print(f"[BnB] Optimization finished. Best Integer Obj: {self.best_integer_obj:.2f}")
            return self.best_integer_obj, elapsed, self.stats, self.best_integer_solution
        else:
            print("[BnB] No integer solution found within limits. Returning best fractional or empty.")
            # 解が見つからなかった場合、最後に一番良かったFractionalな状態を返すか、諦める
            return None, elapsed, self.stats, np.zeros((self.prob.K, self.prob.T))

    def solve_node(self, constraints, max_iter, tol):
        """
        特定のノード（制約セット）に対してCGを実行し、緩和解を返す
        """
        # 1. グラフとプールの状態をこのノード用にセットアップ
        self.apply_constraints(constraints)
        
        # 2. CGループを実行
        # ノードごとの計算なので、反復回数は少なめでも良いが、精度のためある程度回す
        # ここではPricingが制約付きグラフに基づいて行われる
        obj, _, _, current_prob = self.run_cg_loop(max_iter=max_iter, tol=tol)
        
        return obj, current_prob

    def run_cg_loop(self, max_iter, tol=1e-3):
        """内部CGループ (Diving/BnB共通で使える構造)"""
        prev_obj = float('inf')
        no_improve_iter = 0
        current_prob = None
        current_obj = None
        
        # RMP初期化: 現在のプールで制約違反のものを無効化済みの状態でスタートしたい
        # しかし solve_rmp 内でフィルタリングするのでここではループを回すだけで良い
        
        for i in range(max_iter):
            # RMP
            res = self.solve_rmp_with_filtering()
            if res[0] is None: return None, None, 0, None # Infeasible
            
            current_obj, current_prob = res
            
            # Dual (制約下のDual)
            lp_val, pi, sigma = self.solve_rmp_dual_with_filtering()
            
            # 収束判定
            if prev_obj != float('inf'):
                if abs(prev_obj - lp_val) < tol * abs(prev_obj + 1e-9):
                    no_improve_iter += 1
                else:
                    no_improve_iter = 0
            prev_obj = lp_val
            
            # Pricing (制約適用済みグラフを使用)
            pool_add, graph_add = self.pricing(pi, sigma)
            
            if pool_add + graph_add == 0: break
            if no_improve_iter >= 3: break # Tailing off対応
            
        return prev_obj, None, 0, current_prob

    def apply_constraints(self, constraints):
        """
        このノードの制約を「グラフ」と「現在のBranching制約リスト」に適用する
        """
        self.current_constraints = constraints # solve_rmp_with_filteringで使用
        
        # グラフの重みをリセットしてから適用
        penalty = 1e7 # Soft Branching (Infeasible回避のため完全無限大にはしない)
        
        for k in range(self.prob.K):
            if k not in self.graphs:
                from problem import GraphBuilder
                self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            
            G, _, _ = self.graphs[k]
            
            # 重みリセット
            if k in self._original_graph_weights:
                for (u, v), w in self._original_graph_weights[k].items():
                    G[u][v]['weight'] = w
            
            # ノード制約を適用
            node_cons = [c for c in constraints if c[0] == k]
            if not node_cons: continue
            
            for u, v, d in G.edges(data=True):
                etype = d.get('type')
                time_idx = d.get('time')
                
                if time_idx is not None:
                    for (_, bt, bval) in node_cons:
                        if time_idx == bt:
                            if bval == 1: # Must work -> Ban non-work edges
                                if etype not in ['work_start', 'work_cont']:
                                    d['weight'] += penalty
                            elif bval == 0: # Must NOT work -> Ban work edges
                                if etype in ['work_start', 'work_cont']:
                                    d['weight'] += penalty

    def solve_rmp_with_filtering(self):
        """
        現在の制約に違反するプール内の列を除外(Upper Bound=0)してRMPを解く
        """
        model = pulp.LpProblem("RMP_BnB", pulp.LpMinimize)
        
        # 有効な列の選別
        active_cols = []
        for idx in self.rmp_indices:
            col = self.pool[idx]
            if self.is_col_feasible(col, self.current_constraints):
                active_cols.append(col)
        
        # 変数定義
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=pulp.LpContinuous) for c in active_cols}
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        model += pulp.lpSum([c['cost']*x[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * d for d in delta])
        
        for t in range(self.prob.T):
            model += pulp.lpSum([c['schedule'][t]*x[c['id']] for c in active_cols]) + delta[t] >= self.prob.demand[t]
            
        for k in range(self.prob.K):
            model += pulp.lpSum([x[c['id']] for c in active_cols if c['group_id'] == k]) == 1
            
        solver = pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)
        
        if model.status != pulp.LpStatusOptimal:
            return None, None
            
        # 確率解の構築
        sched_prob = np.zeros((self.prob.K, self.prob.T))
        for c in active_cols:
            val = x[c['id']].varValue
            if val and val > 1e-5:
                sched_prob[c['group_id']] += np.array(c['schedule']) * val
                
        return pulp.value(model.objective), sched_prob

    def solve_rmp_dual_with_filtering(self):
        """Dual取得用 (Filteringあり)"""
        # solve_rmp_with_filtering とほぼ同じ構成で Dual を返す
        # (コード簡略化のためロジックを再記述。実運用では共通化推奨)
        model = pulp.LpProblem("RMP_Dual", pulp.LpMinimize)
        active_cols = [self.pool[i] for i in self.rmp_indices if self.is_col_feasible(self.pool[i], self.current_constraints)]
        
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=pulp.LpContinuous) for c in active_cols}
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
            
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if model.status != pulp.LpStatusOptimal:
            return 0.0, [0]*self.prob.T, [0]*self.prob.K
            
        pi = [c.pi for c in cons_d]
        sigma = [c.pi for c in cons_c]
        return pulp.value(model.objective), pi, sigma

    def is_col_feasible(self, col, constraints):
        """列が制約リストを満たすかチェック"""
        k = col['group_id']
        sched = col['schedule']
        for (bk, bt, bval) in constraints:
            if k == bk and sched[bt] != bval:
                return False
        return True

    def _cache_original_weights(self):
        """グラフの初期重みを保存"""
        for k in range(self.prob.K):
            if k not in self.graphs:
                from problem import GraphBuilder
                self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, _, _ = self.graphs[k]
            if k not in self._original_graph_weights:
                self._original_graph_weights[k] = {}
            for u, v, d in G.edges(data=True):
                self._original_graph_weights[k][(u,v)] = d.get('weight', 0)

    def select_branching_variable(self, schedule_prob):
        dist = np.abs(schedule_prob - 0.5)
        k_idx, t_idx = np.unravel_index(np.argmin(dist), dist.shape)
        val = schedule_prob[k_idx, t_idx]
        return k_idx, t_idx, val

    def is_integer_solution(self, schedule_prob):
        return np.all((schedule_prob < 1e-3) | (schedule_prob > 1 - 1e-3))
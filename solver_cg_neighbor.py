import time
import numpy as np
from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverNeighbor(ColumnGenerationSolver):
    """
    PDF資料の「IP1B: 近傍探索による列生成」を実装したクラス。
    列生成補助問題において、厳密解法(グラフ探索)を行う前に、
    既存列の近傍探索(Neighborhood Search)を行い、負の被約費用を持つ列を高速に見つける。
    """
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.label = "CG Neighbor"
    
    def pricing(self, pi, sigma):
        """
        Pricing step override.
        1. まず近傍探索ヒューリスティックを実行 (IP1Bの手法) [cite: 191]
        2. 列が見つかればそれを追加して終了 (高速化)
        3. 見つからなければ、厳密解法(親クラスのグラフ探索)へフォールバック (収束性保証のため)
        """
        t_start = time.perf_counter()
        
        # --- 1. 近傍探索 (Heuristic Pricing) ---
        neighbor_added_count = self._pricing_heuristic_neighborhood(pi, sigma)
        
        self.stats['time_pool'] += (time.perf_counter() - t_start) # ヒューリスティック時間はPool/その他に計上
        
        # 近傍探索で有効な列が見つかった場合、グラフ探索をスキップして高速化
        if neighbor_added_count > 0:
            self.stats['count_pool_hit'] += neighbor_added_count # 統計上はPool Hit扱いに加算(便宜上)
            return neighbor_added_count, 0

        # --- 2. 厳密探索 (Exact Graph Pricing) ---
        # ヒューリスティックで見つからない場合のみ、厳密解法を実行して最適性を保証する
        # (資料のIP1Bは近似解法ですが、ベンチマークで公平に比較するため、収束停止を防ぐフォールバックを入れます)
        return super().pricing(pi, sigma)

    def _pricing_heuristic_neighborhood(self, pi, sigma):
        """
        既存のPool内の列に対して近傍操作を行い、負の被約費用を持つ列を探索する。
        近傍定義: 開始時刻・終了時刻の ±1 時間の変更
        """
        added_count = 0
        candidates = []
        
        # 探索対象: 現在のPoolにある列 (Step 0: k=ni ... Step 2: k<-k-1) [cite: 192, 194]
        # 最近追加された列の方が有望な可能性があるため、後ろから走査するのが一般的
        current_pool_indices = list(range(len(self.pool)))
        np.random.shuffle(current_pool_indices) # ランダム性を持たせて局所解回避
        
        search_limit = min(len(current_pool_indices), 200) # 計算時間短縮のため探索数を制限
        
        for idx in current_pool_indices[:search_limit]:
            col = self.pool[idx]
            emp_id = col['group_id']
            base_schedule = col['schedule']
            
            # 近傍パターンの生成
            neighbors = self._generate_neighbors(base_schedule, emp_id)
            
            for sched_n in neighbors:
                # 既知のパターンかチェック (重複登録防止)
                sched_tuple = tuple(sched_n)
                if (emp_id, sched_tuple) in self.pattern_to_id:
                    continue
                
                # コスト計算
                emp = self.prob.employees[emp_id]
                cost_n = np.sum(np.array(sched_n) * (emp['hourly_wage'] + emp['rho']))
                
                # 被約費用計算: rc = cost - sum(aij * pi) - sigma
                # [cite: 121, 186]
                rc = cost_n - np.dot(pi, sched_n) - sigma[emp_id]
                
                if rc < -1e-5:
                    candidates.append((rc, emp_id, sched_n, cost_n))

        # 有望な上位の列を追加
        candidates.sort(key=lambda x: x[0])
        limit_add = self.prob.K  # 一度に追加する列の上限
        
        for rc, eid, sched, cost in candidates[:limit_add]:
            # 重複再チェック(ループ内で追加済みの場合)
            sched_tuple = tuple(sched)
            if (eid, sched_tuple) in self.pattern_to_id:
                continue
                
            new_id = len(self.pool)
            self.pool.append({'id': new_id, 'group_id': eid, 'schedule': sched, 'cost': cost})
            self.pattern_to_id[(eid, sched_tuple)] = new_id
            
            if new_id not in self.rmp_indices:
                self.rmp_indices.append(new_id)
                added_count += 1
                
        return added_count

    def _generate_neighbors(self, schedule, emp_id):
        """
        スケジュールの近傍を生成する。
        定義: 連続勤務区間の「開始」または「終了」を前後1つずらす。
        """
        neighbors = []
        T = self.prob.T
        emp = self.prob.employees[emp_id]
        L_min = emp['L_min']
        L_max = emp['L_max']
        
        # 現在の勤務区間を検出 (簡易的に単一シフトと仮定、あるいは最初のシフトを操作)
        # scheduleは [0, 0, 1, 1, 1, 0, ...]
        starts = []
        ends = []
        is_working = False
        for t in range(T):
            if schedule[t] == 1 and not is_working:
                starts.append(t)
                is_working = True
            elif schedule[t] == 0 and is_working:
                ends.append(t - 1) # endはinclusive
                is_working = False
        if is_working:
            ends.append(T - 1)
            
        if not starts:
            return []

        # 各シフト区間に対して操作
        for s, e in zip(starts, ends):
            current_len = e - s + 1
            
            # 操作1: 開始時間を早める (s-1)
            if s > 0:
                new_sched = list(schedule)
                new_sched[s-1] = 1
                if L_min <= current_len + 1 <= L_max: # 制約チェック
                    neighbors.append(new_sched)
            
            # 操作2: 開始時間を遅らせる (s+1)
            if current_len > 1:
                new_sched = list(schedule)
                new_sched[s] = 0
                if L_min <= current_len - 1 <= L_max:
                    neighbors.append(new_sched)

            # 操作3: 終了時間を延ばす (e+1)
            if e < T - 1:
                new_sched = list(schedule)
                new_sched[e+1] = 1
                if L_min <= current_len + 1 <= L_max:
                    neighbors.append(new_sched)

            # 操作4: 終了時間を早める (e-1)
            if current_len > 1:
                new_sched = list(schedule)
                new_sched[e] = 0
                if L_min <= current_len - 1 <= L_max:
                    neighbors.append(new_sched)
                    
            # 操作5: シフト全体を左にずらす
            if s > 0:
                new_sched = list(schedule)
                new_sched[s-1] = 1
                new_sched[e] = 0
                neighbors.append(new_sched)

            # 操作6: シフト全体を右にずらす
            if e < T - 1:
                new_sched = list(schedule)
                new_sched[s] = 0
                new_sched[e+1] = 1
                neighbors.append(new_sched)

        return neighbors
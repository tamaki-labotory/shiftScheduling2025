import pulp
import time
import numpy as np
import os
from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverLRU(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=2000, usage_threshold=5):
        """
        Args:
            problem: 問題インスタンス
            use_pool: プールを使用するかどうか
            pool_cleanup_threshold: プール掃除を実行する列数の閾値
            usage_threshold: 「頻繁に使用された」とみなす使用回数の閾値
        """
        super().__init__(problem, use_pool)
        self.pool_cleanup_threshold = pool_cleanup_threshold
        self.usage_threshold = usage_threshold

    def add_column(self, k, schedule):
        """
        親クラスのadd_columnを呼び出し、usage_countを初期化する。
        """
        col_id = super().add_column(k, schedule)
        # 既存列の場合はIDが返ってくるだけなので、新規追加時のみ初期化したいが、
        # 辞書へのアクセスは安価なので念のため存在確認して初期化
        col = self.pool[col_id]
        if 'usage_count' not in col:
            col['usage_count'] = 0
        return col_id

    def solve_rmp(self, integer=False, mip_time_limit=30, mip_gap=0.05, log_path=None):
        """
        親クラスのsolve_rmpをオーバーライドして、使用回数(usage_count)のカウントアップ処理を追加。
        """
        t_start = time.perf_counter()
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        active_cols = [self.pool[i] for i in self.rmp_indices]
        
        # 変数定義
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=cat) for c in active_cols}
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        # 目的関数
        model += pulp.lpSum([c['cost']*x[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * d for d in delta])
        
        # 制約条件: 需要
        cons_d = []
        for t in range(self.prob.T):
            expr = pulp.lpSum([c['schedule'][t]*x[c['id']] for c in active_cols]) + delta[t]
            model += expr >= self.prob.demand[t]
            cons_d.append(model.constraints[list(model.constraints.keys())[-1]])
            
        # 制約条件: 各従業員1パターン
        cons_c = []
        for k in range(self.prob.K):
            expr = pulp.lpSum([x[c['id']] for c in active_cols if c['group_id'] == k])
            model += expr == 1
            cons_c.append(model.constraints[list(model.constraints.keys())[-1]])
            
        # ソルバー実行
        if integer:
            # log_pathがある場合は渡す
            if log_path:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit, gapRel=mip_gap, logPath=log_path)
            else:
                solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit, gapRel=mip_gap)
        else:
            solver = pulp.PULP_CBC_CMD(msg=0)
        
        model.solve(solver)
        elapsed = time.perf_counter() - t_start
        if integer: self.stats['time_rmp_mip'] += elapsed
        else: self.stats['time_rmp_lp'] += elapsed

        if model.status != pulp.LpStatusOptimal: return None

        # --- 追加箇所: 使用回数の更新 ---
        # 整数解法時だけでなく、線形緩和時も「基底（正の値）」になっていればカウントする
        for c in active_cols:
            val = x[c['id']].varValue
            if val is not None and val > 1e-5:
                if 'usage_count' not in c:
                    c['usage_count'] = 0
                c['usage_count'] += 1
        # -----------------------------

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

    def reset_for_new_period(self):
        """
        プール掃除を行ってからリセットする。
        元のagingコードではリセット後に掃除していたため、active判定が効かないバグがあった可能性があるため修正。
        """
        if self.use_pool and len(self.pool) > self.pool_cleanup_threshold:
            self.cleanup_pool()
        
        # 掃除が終わってからインデックスをクリア
        self.rmp_indices = []
        
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 

    def cleanup_pool(self):
        """
        プールのメンテナンスを行う。
        以下のいずれかに該当する場合は保持する（OR条件）。
        1. 直近のRMPに含まれている（active）
        2. 直近に生成された新規列（recent）
        3. 過去に頻繁に使用された列（frequent）
        """
        active_indices = set(self.rmp_indices)
        
        keep_recent_count = 1000
        total_cols = len(self.pool)
        cutoff_index = total_cols - keep_recent_count
        
        new_pool = []
        new_pattern_to_id = {}
        new_rmp_indices = []
        
        print(f"DEBUG: Cleaning pool (Aging2)... Current: {len(self.pool)}")
        
        for old_idx, col in enumerate(self.pool):
            # 条件判定
            is_active = (col['id'] in active_indices)
            is_recent = (old_idx >= cutoff_index)
            is_frequent = (col.get('usage_count', 0) >= self.usage_threshold)
            
            # いずれかの条件を満たせば保持
            if is_active or is_recent or is_frequent:
                new_id = len(new_pool)
                
                # RMPインデックスの更新マッピング
                if is_active:
                    new_rmp_indices.append(new_id)
                
                # ID更新
                col['id'] = new_id
                new_pool.append(col)
                
                pat_key = (col['group_id'], tuple(col['schedule']))
                new_pattern_to_id[pat_key] = new_id
        
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        self.rmp_indices = new_rmp_indices
        
        print(f"DEBUG: Pool cleaned. New: {len(self.pool)} (Removed {total_cols - len(self.pool)})")
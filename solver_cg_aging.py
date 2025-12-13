from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverWithAging(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=2000):
        # 親クラスの初期化を呼び出す
        super().__init__(problem, use_pool)
        self.pool_cleanup_threshold = pool_cleanup_threshold

    def reset_for_new_period(self):
        """
        親クラスのメソッドをオーバーライド（上書き）。
        プール掃除のロジックを追加する。
        """
        self.rmp_indices = []
        if not self.use_pool:
            # 標準動作: 全リセット
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        else:
            # Proposed動作: プール掃除（閾値を超えた場合）
            if len(self.pool) > self.pool_cleanup_threshold:
                self.cleanup_pool()

    def cleanup_pool(self):
        """
        プール内の不要な列を整理してIDを振り直す
        """
        new_pool = []
        new_pattern_to_id = {}
        
        for col in self.pool:
            # ここに将来的なフィルタリング条件（例: 長期間使われていない列の削除）を追加可能
            new_id = len(new_pool)
            col['id'] = new_id
            new_pool.append(col)
            
            pat_key = (col['group_id'], tuple(col['schedule']))
            new_pattern_to_id[pat_key] = new_id
            
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
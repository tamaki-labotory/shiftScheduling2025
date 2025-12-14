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
        プールのメンテナンスを行う。
        1. 現在RMP（基底）に含まれている列は絶対に削除しない
        2. それ以外の列のうち、「古いもの」や「コストが悪いもの」を削除する
        """
        # 現在RMPで使用中の列IDセット（これらは消してはいけない）
        active_indices = set(self.rmp_indices)
        
        # 残す列の数（例: 直近の1000列は無条件で残す、など）
        keep_recent_count = 1000
        total_cols = len(self.pool)
        cutoff_index = total_cols - keep_recent_count
        
        new_pool = []
        new_pattern_to_id = {}
        
        # 新しいRMPインデックスのリスト（IDがずれるため作り直す）
        new_rmp_indices = []
        
        print(f"DEBUG: Cleaning pool... Current size: {len(self.pool)}")
        
        for old_idx, col in enumerate(self.pool):
            # 【残す条件】
            # A. 現在RMPに含まれている（active_indicesにある）
            # OR
            # B. 最近追加された列である（old_idx >= cutoff_index）
            
            is_active = (col['id'] in active_indices)
            is_recent = (old_idx >= cutoff_index)
            
            if is_active or is_recent:
                # 新しいIDを付与
                new_id = len(new_pool)
                
                # RMPに含まれていた列なら、新しいIDを記録しておく
                if is_active:
                    new_rmp_indices.append(new_id)
                
                # 列データを更新して追加
                col['id'] = new_id
                new_pool.append(col)
                
                # 重複チェック用辞書も更新
                pat_key = (col['group_id'], tuple(col['schedule']))
                new_pattern_to_id[pat_key] = new_id
        
        # メンバ変数を更新
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        self.rmp_indices = new_rmp_indices # IDが変わったので更新が必要
        
        print(f"DEBUG: Pool cleaned. New size: {len(self.pool)} (Removed {total_cols - len(self.pool)} cols)")
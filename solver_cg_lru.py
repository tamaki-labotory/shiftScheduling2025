from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverWithLRU(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=2000):
        super().__init__(problem, use_pool)
        self.pool_cleanup_threshold = pool_cleanup_threshold
        # 【追加】現在時刻（イテレーション数）を管理するカウンタ
        self.current_iteration = 0

    def reset_for_new_period(self):
        self.rmp_indices = []
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        else:
            if len(self.pool) > self.pool_cleanup_threshold:
                self.cleanup_pool()

    def update_column_usage(self):
        """
        【重要】RMPを解いた直後にこのメソッドを必ず呼び出してください。
        現在RMPの基底に含まれている列の「最終使用時刻」を更新します。
        """
        self.current_iteration += 1
        current_time = self.current_iteration
        
        # 現在activeな列の last_used を現在時刻に更新
        for idx in self.rmp_indices:
            # 列データに 'last_used' キーがなければ作成、あれば更新
            self.pool[idx]['last_used'] = current_time

    def cleanup_pool(self):
        """
        LRU方式によるプール掃除
        1. Activeな列は無条件で残す
        2. Inactiveな列は 'last_used' が新しい順に残す
        """
        # 現在RMPで使用中の列IDセット
        active_indices = set(self.rmp_indices)
        
        # 残したいInactive列の数（全体でthreshold以下になるように調整）
        # 例: threshold=2000, active=100 なら、inactiveからは1900個残す
        max_pool_size = self.pool_cleanup_threshold
        keep_inactive_count = max(0, max_pool_size - len(active_indices))
        
        print(f"DEBUG: Cleaning pool (LRU)... Current: {len(self.pool)}, Active: {len(active_indices)}")

        # --- 分類と選定 ---
        
        # 1. Activeな列（保持確定）と、Inactiveな列に分ける
        active_cols = []
        inactive_cols = []
        
        for idx, col in enumerate(self.pool):
            # もし 'last_used' が記録されていない場合は 0 (最古) 扱いにする
            if 'last_used' not in col:
                col['last_used'] = 0
            
            if idx in active_indices:
                active_cols.append(col)
            else:
                inactive_cols.append(col)
        
        # 2. Inactiveな列を「最終使用時刻(last_used)」の降順（新しい順）にソート
        inactive_cols.sort(key=lambda x: x['last_used'], reverse=True)
        
        # 3. 上位 N 個だけを残す
        kept_inactive_cols = inactive_cols[:keep_inactive_count]
        
        # --- 再構築 ---
        
        new_pool = []
        new_pattern_to_id = {}
        new_rmp_indices = []
        
        # Active列を先頭に追加（順序は問わないが、管理上わかりやすく）
        for col in active_cols:
            new_id = len(new_pool)
            col['id'] = new_id
            new_rmp_indices.append(new_id) # 新しいIDを記録
            new_pool.append(col)
            
            pat_key = (col['group_id'], tuple(col['schedule']))
            new_pattern_to_id[pat_key] = new_id

        # 選定されたInactive列を追加
        for col in kept_inactive_cols:
            new_id = len(new_pool)
            col['id'] = new_id
            # Inactiveなので rmp_indices には追加しない
            new_pool.append(col)
            
            pat_key = (col['group_id'], tuple(col['schedule']))
            new_pattern_to_id[pat_key] = new_id
            
        # メンバ変数を更新
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        self.rmp_indices = new_rmp_indices
        
        print(f"DEBUG: Pool cleaned. New size: {len(self.pool)} (Active: {len(active_cols)}, Inactive Kept: {len(kept_inactive_cols)})")
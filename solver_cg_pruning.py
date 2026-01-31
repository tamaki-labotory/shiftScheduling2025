from solver_cg import ColumnGenerationSolver
import math
import time
import numpy as np
from collections import defaultdict
import os

class ColumnGenerationSolverWithPruning(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True, pool_cleanup_threshold=2000):
        # 親クラスの初期化
        super().__init__(problem, use_pool)
        
        # ★変更点: この閾値は「1従業員あたりの最大保持数列」として扱います
        self.pool_cleanup_threshold = pool_cleanup_threshold
        
        # --- 実験設定パラメータ ---
        self.cg_stall_limit = 3
        self.ip_selection_ratio = 0.20
        self.mip_time_limit = 30

    def reset_for_new_period(self):
        """
        新しい週の開始時に呼び出される。
        プールサイズが肥大化しすぎていないかチェックし、必要なら掃除する。
        """
        self.rmp_indices = []
        if not self.use_pool:
            self.pool = []
            self.graphs = {}
            self.pattern_to_id = {} 
        else:
            # 掃除が必要かどうかの簡易チェック
            # 「従業員数 x 閾値」を超えていれば、誰かが溢れている可能性が高いので詳細チェックへ
            total_threshold = self.prob.K * self.pool_cleanup_threshold
            if len(self.pool) > total_threshold:
                self.cleanup_pool(pi=None, sigma=None)

    def cleanup_pool(self, pi=None, sigma=None):
        """
        【従業員別】プールのメンテナンスを行う。
        各従業員について、保持している列数が threshold を超えた場合のみ、
        古い列や質の悪い列を削除する。
        """
        # 現在RMPで使用中の列IDセット（これらは絶対に消さない）
        active_indices_set = set(self.rmp_indices)
        
        # 1. 従業員ごとに列を分類する
        # cols_by_emp[k] = [col_obj, col_obj, ...]
        cols_by_emp = defaultdict(list)
        for col in self.pool:
            cols_by_emp[col['group_id']].append(col)
            
        new_pool = []
        new_pattern_to_id = {}
        new_rmp_indices = []
        
        removed_count = 0
        
        print(f"DEBUG: Checking pool limits per employee (Threshold={self.pool_cleanup_threshold})...")

        # 2. 従業員ごとに選別処理
        # 全従業員ループ（プールに列がない従業員もいるかもしれないが、cols_by_empのキー分で十分）
        # ただし、順序保持のため 0..K-1 で回すのが安全
        sorted_emp_ids = sorted(cols_by_emp.keys())
        
        for k in sorted_emp_ids:
            emp_cols = cols_by_emp[k]
            num_cols = len(emp_cols)
            
            # 閾値を超えていなければ、そのまま全列キープ
            if num_cols <= self.pool_cleanup_threshold:
                for col in emp_cols:
                    self._add_col_to_new_lists(col, new_pool, new_pattern_to_id, new_rmp_indices, active_indices_set)
                continue
            
            # --- 閾値を超えている場合の選別ロジック ---
            
            # 基準1: Recency (最近追加された N 列は残す)
            # emp_cols は self.pool への追加順（時系列順）に並んでいると仮定できる
            keep_recent_count = 1000 # 直近1000列は無条件保護
            cutoff_index = num_cols - keep_recent_count
            
            kept_for_k = 0
            
            for local_idx, col in enumerate(emp_cols):
                should_keep = False
                
                # A. 現在RMPに含まれている (Must Keep)
                if col['id'] in active_indices_set:
                    should_keep = True
                
                # B. 最近追加された (Recency)
                elif local_idx >= cutoff_index:
                    should_keep = True
                    
                # C. 被約費用が良い (Quality)
                elif pi is not None and sigma is not None:
                    # まだKeep判定されていない古い列についてのみ計算
                    sched = np.array(col['schedule'])
                    val = np.dot(pi, sched)
                    rc = col['cost'] - val - sigma[k]
                    # 有望なら残す (閾値は適宜調整、ここでは負または0に近いもの)
                    if rc < 1e-5:
                        should_keep = True
                
                if should_keep:
                    self._add_col_to_new_lists(col, new_pool, new_pattern_to_id, new_rmp_indices, active_indices_set)
                    kept_for_k += 1
                else:
                    removed_count += 1
            
            # (Option) もし削りすぎてRMP維持に必要な列まで消えるリスクがあるなら、
            # 最低限の列数を確保するロジックをここに入れるが、
            # 上記 A. で active_indices_set を守っているので大丈夫。

        # 3. メンバ変数を更新
        self.pool = new_pool
        self.pattern_to_id = new_pattern_to_id
        self.rmp_indices = new_rmp_indices
        
        print(f"DEBUG: Pool cleanup done. Removed {removed_count} cols. New total size: {len(self.pool)}")

    def _add_col_to_new_lists(self, col, new_pool, new_pattern_to_id, new_rmp_indices, active_indices_set):
        """
        残すと決めた列を新しいリストに追加し、IDを振り直すヘルパー
        """
        # 元のIDがActiveだったか？
        was_active = (col['id'] in active_indices_set)
        
        # 新しいIDを発行
        new_id = len(new_pool)
        
        # 列オブジェクトのIDを更新
        # (注意: colは辞書なので参照渡し。ここで書き換えても良いが、念のためコピーはしない)
        col['id'] = new_id
        new_pool.append(col)
        
        # マップ更新
        pat_key = (col['group_id'], tuple(col['schedule']))
        new_pattern_to_id[pat_key] = new_id
        
        # RMPインデックス更新
        if was_active:
            new_rmp_indices.append(new_id)

    def solve(self, max_iter=1000, time_limit=3600, tol=1e-8, patience=10, mip_rc_threshold=1e10, mip_gap=0.0001):
        """
        メインのsolveメソッド（シグネチャはmain.pyからの呼び出しに合わせる）
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
            
            # --- Stall Check ---
            obj_history.append(current_obj)
            if len(obj_history) >= self.cg_stall_limit:
                recent = obj_history[-self.cg_stall_limit:]
                if max(recent) - min(recent) < 1e-6:
                    print(f"INFO: CG Stalled. Stopping CG.")
                    break

            # Pricing
            pool_add, graph_add = self.pricing(pi, sigma)
            
            self.stats['iterations'] += 1
            self.history.append({'iter': i + 1, 'obj': current_obj, 'pool_hits': pool_add, 'graph_gen': graph_add})
            
            if pool_add + graph_add == 0:
                print("INFO: No new columns found.")
                break
                
            # --- Cleanup (Per Employee) ---
            if self.use_pool:
                # 誰か一人でも閾値を超えているかチェックするために、総数をチェック
                # （厳密には個別にチェックすべきだが、ループごとのオーバーヘッドを減らすため
                #   全体サイズが「K * threshold」を超えたときだけ詳細チェックに入る運用とする）
                total_threshold = self.prob.K * self.pool_cleanup_threshold
                if len(self.pool) > total_threshold:
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
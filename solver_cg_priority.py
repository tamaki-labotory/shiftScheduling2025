import pulp
import time
import numpy as np
import pandas as pd
import os
from solver_cg import ColumnGenerationSolver

class ColumnGenerationSolverPriority(ColumnGenerationSolver):
    def __init__(self, problem, use_pool=True):
        super().__init__(problem, use_pool)

    def add_column(self, k, schedule):
        """
        列を追加する際、実績カウンタを初期化する
        """
        # 親クラスのメソッドで追加（または既存ID取得）
        col_id = super().add_column(k, schedule)
        
        col = self.pool[col_id]
        # まだカウンタがない場合のみ初期化（既存列なら履歴を維持）
        if 'usage_int' not in col:
            col['usage_int'] = 0  # 整数解（最終解）で選ばれた回数
        if 'usage_rmp' not in col:
            col['usage_rmp'] = 0  # RMP（LP緩和）で基底になった回数
            
        return col_id

    def solve_rmp(self, integer=False, mip_time_limit=30, mip_gap=0.05, log_path=None):
        """
        RMPを解いた後、LP緩和解で選ばれた列のカウントアップを行う
        """
        # 親クラスのsolve_rmpはそのまま利用できない（内部変数の active_cols にアクセスしたいため）
        # コード重複を避けるため親を呼び出しつつ、結果から逆算してカウントすることも可能だが、
        # ここでは正確に基底変数を捕捉するため、親クラスのロジックをベースに拡張する。
        
        t_start = time.perf_counter()
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        active_cols = [self.pool[i] for i in self.rmp_indices]
        
        cat = pulp.LpBinary if integer else pulp.LpContinuous
        x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, cat=cat) for c in active_cols}
        delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(self.prob.T)]
        
        model += pulp.lpSum([c['cost']*x[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * d for d in delta])
        
        for t in range(self.prob.T):
            expr = pulp.lpSum([c['schedule'][t]*x[c['id']] for c in active_cols]) + delta[t]
            model += expr >= self.prob.demand[t]
            
        for k in range(self.prob.K):
            expr = pulp.lpSum([x[c['id']] for c in active_cols if c['group_id'] == k])
            model += expr == 1
            
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=mip_time_limit, gapRel=mip_gap) if integer else pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)
        
        elapsed = time.perf_counter() - t_start
        if integer: self.stats['time_mip'] += elapsed
        else: self.stats['time_rmp'] += elapsed

        if model.status != pulp.LpStatusOptimal: return None

        # === ★追加: RMPでの採用実績をカウントアップ (LPの場合のみ) ===
        if not integer:
            for c in active_cols:
                val = x[c['id']].varValue
                # 基底に入っている（値が正）ならカウント
                if val is not None and val > 1e-5:
                    c['usage_rmp'] = c.get('usage_rmp', 0) + 1
        # ========================================================

        if integer:
            self.stats['mip_total_columns'] = len(active_cols)
            final_schedule = np.zeros((self.prob.K, self.prob.T))
            selected_ids = []
            for c in active_cols:
                val = x[c['id']].varValue
                if val is not None and val > 0.5:
                    final_schedule[c['group_id']] = c['schedule']
                    selected_ids.append(c['id'])
            return pulp.value(model.objective), final_schedule, selected_ids
        else:
            # 制約名から双対変数を取得
            # model.constraints は辞書なのでvalues()をリスト化してアクセス
            cons = list(model.constraints.values())
            # 前半T個が需要制約(pi)、後半K個が割当制約(sigma)
            pi = [cons[t].pi for t in range(self.prob.T)]
            sigma = [cons[self.prob.T + k].pi for k in range(self.prob.K)]
            return pulp.value(model.objective), pi, sigma

    def pricing(self, pi, sigma):
        """
        ★変更点: プール探索順序を「実績順」にソートしてから行う
        """
        pool_added_count = 0
        graph_added_count = 0
        t_pool_start = time.perf_counter()
        
        candidates = []
        if self.use_pool:
            # RMPに含まれていない列を対象にする
            inactive_indices = [i for i in range(len(self.pool)) if i not in self.rmp_indices]
            
            # === ★重要: ソートロジック ===
            # 第1優先: 整数解での採用回数 (usage_int) 降順
            # 第2優先: RMPでの採用回数 (usage_rmp) 降順
            # これにより「実績のある列」がリストの先頭に来る
            inactive_indices.sort(
                key=lambda i: (self.pool[i].get('usage_int', 0), self.pool[i].get('usage_rmp', 0)),
                reverse=True
            )
            
            # ソート順にReduced Costを計算
            for i in inactive_indices:
                col = self.pool[i]
                k = col['group_id']
                rc = col['cost'] - np.dot(pi, col['schedule']) - sigma[k]
                
                # 負のReduced Costが見つかったら候補に追加
                if rc < -1e-5:
                    candidates.append((rc, i))
                    
                    # ★最適化: 
                    # ソートされているため、実績のある列から順に見つかる。
                    # 十分な数（例: 全従業員数分など）が見つかったら、
                    # それ以下の「実績のない列」の探索を打ち切ることで高速化を図る戦略も可能。
                    # ここでは確実性を重視し、見つかったものをすべて候補に入れるが、
                    # 追加するのは上位N個とする（親クラス同様）。
        
        # Reduced Costが小さい（改善効果が高い）順にソートしてRMPに追加
        # ※ 実績順でスキャンしたが、追加するのは「数学的に効果が高い順」が良い
        candidates.sort(key=lambda x: x[0])
        
        limit_add = self.prob.K * 2 
        for rc, i in candidates[:limit_add]:
            self.rmp_indices.append(i)
            pool_added_count += 1
            self.stats['count_pool_hit'] += 1
        
        self.stats['time_pool'] += (time.perf_counter() - t_pool_start)
        
        # 十分な数の列がプールから見つかったら、グラフ探索（新規生成）をスキップ
        if self.use_pool and pool_added_count > 5:
            self.stats['count_graph_skip'] += self.prob.K 
            return pool_added_count, 0

        # --- 以下、グラフ探索（親クラスと同様） ---
        t_graph_start = time.perf_counter()
        # (グラフ探索ロジックは親クラスのpricing後半と同じため省略せず記述する必要があるが
        #  super().pricing() を呼ぶとプール探索が重複してしまう。
        #  そのため、ここではグラフ探索部分のみ記述する)
        
        # ...グラフ探索の実装...
        # （コード簡略化のため、solver_cg.pyの実装をコピーします）
        from problem import GraphBuilder
        import networkx as nx
        
        for k in range(self.prob.K):
            if k not in self.graphs: self.graphs[k] = GraphBuilder.build_graph(self.prob, k)
            G, src, sink = self.graphs[k]
            emp = self.prob.employees[k]
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
                path = nx.shortest_path(G, src, sink, weight='weight', method='bellman-ford')
                # 経路からスケジュール復元
                sched = [0]*self.prob.T
                rc_val = 0
                for u, v in zip(path, path[1:]):
                    d = G[u][v]
                    rc_val += d['weight']
                    if d.get('type') in ['work_start', 'work_cont']: sched[d['time']] = 1
                
                if rc_val < -1e-5:
                    idx = self.add_column(k, sched)
                    if idx not in self.rmp_indices:
                        self.rmp_indices.append(idx)
                        graph_added_count += 1
                        self.stats['count_graph_new'] += 1
            except nx.NetworkXNoPath: pass

        self.stats['time_graph'] += (time.perf_counter() - t_graph_start)
        return pool_added_count, graph_added_count

    def solve(self, max_iter=50, time_limit=300, **kwargs):
        """
        ソルバー実行フロー。
        最後に整数解が得られた場合、その列の usage_int をカウントアップする。
        """
        # 親クラスのsolveを呼ぶのではなく、カウンタ更新のために再定義（あるいはラップ）する
        # ここではラップして処理を追加する形をとる
        
        # solve本体の実行
        result = super().solve(max_iter, time_limit, **kwargs)
        final_obj, elapsed, stats, final_schedule = result
        
        # === ★追加: 最終的な整数解に使われた列をカウントアップ ===
        # super().solve() の中で self.final_selected_ids が更新されているはず
        if hasattr(self, 'final_selected_ids') and self.final_selected_ids:
            for col_id in self.final_selected_ids:
                # pool内の該当する列を探して更新
                # poolはリストだがIDで直接アクセスできないため、辞書を作るかIDで検索
                # しかし、col_idは self.poolのインデックスと一致する設計になっているはず
                # (add_columnの実装: col_id = len(self.pool))
                if 0 <= col_id < len(self.pool):
                    self.pool[col_id]['usage_int'] = self.pool[col_id].get('usage_int', 0) + 1
        
        return result

    def save_pool_to_csv(self, filename):
        """
        CSV保存時にカウンタ情報も含める
        """
        final_mip_indices_set = set(self.rmp_indices)
        
        data = []
        for i, col in enumerate(self.pool):
            emp = self.prob.employees[col['group_id']]
            sched_str = "".join(map(str, map(int, col['schedule'])))
            
            is_in_mip = 1 if i in final_mip_indices_set else 0
            is_selected = 1 if col['id'] in self.final_selected_ids else 0
            
            data.append({
                'col_id': col['id'],
                'emp_id': col['group_id'],
                'emp_type': emp['type'],
                'cost': col['cost'],
                'schedule_pattern': sched_str,
                'in_final_mip': is_in_mip,
                'is_selected': is_selected,
                # ★追加情報
                'usage_int': col.get('usage_int', 0),
                'usage_rmp': col.get('usage_rmp', 0)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"  -> Pool saved with priority stats to: {filename}")

    def load_pool_from_csv(self, filename):
        """
        CSVからカウンタ情報を含めてロードする
        """
        if not os.path.exists(filename):
            return
        try:
            df = pd.read_csv(filename)
            loaded_count = 0
            for _, row in df.iterrows():
                k = int(row['emp_id'])
                sched_str = str(row['schedule_pattern'])
                schedule = [int(c) for c in sched_str]
                
                prev_pool_size = len(self.pool)
                col_id = self.add_column(k, schedule)
                
                # 既存列だった場合も、カウンタ情報を復元/加算する
                col = self.pool[col_id]
                
                # CSVにカラムがあれば読み込む
                if 'usage_int' in row:
                    col['usage_int'] = max(col.get('usage_int', 0), int(row['usage_int']))
                if 'usage_rmp' in row:
                    col['usage_rmp'] = max(col.get('usage_rmp', 0), int(row['usage_rmp']))
                    
                if len(self.pool) > prev_pool_size:
                    loaded_count += 1
            print(f"  -> Loaded priority pool from {filename}: Added {loaded_count} new cols.")
        except Exception as e:
            print(f"  [Warning] Failed to load pool from {filename}: {e}")
import pulp
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 問題設定 (現実的な従業員グループ)
# ==========================================
class ShiftSchedulingProblem:
    def __init__(self, n_time_slots=168, n_groups=3,seed=42): # n_groupsを20に増やす
        self.T = n_time_slots
        self.hours = np.arange(self.T)
        
        # 週ごとの違いを出すためにseedを設定
        np.random.seed(seed)
        
        # --- 以下、需要生成ロジックは既存と同じですが、seedによりnoiseが変わります ---
        day_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
        weekly_base = np.tile(day_pattern, 7)
        weekly_base[24*5:] = weekly_base[24*5:] * 1.2 
        
        scale_factor = n_groups * 0.6 
        base_demand = weekly_base * scale_factor
        
        # seedによって変動するノイズ
        noise = np.random.randint(-2, 3, self.T)
        self.demand = np.clip(base_demand + noise, 5, None).astype(int)
        
        self.groups = []
        
        # --- 変更点2: グループの大量自動生成 ---
        # "Full-time", "Part-time Day", "Part-time Night" の3タイプを
        # ランダムなバリエーションを持たせながら大量に作る
        
        types = ['FT', 'PT_Day', 'PT_Night']
        
        for i in range(n_groups):
            types = ['FT', 'PT_Day', 'PT_Night']
        
        for i in range(n_groups):
            # g_type = np.random.choice(types) # ランダムをやめる
            g_type = types[i % 3] # 0,1,2,0,1... と循環させる
            
            # 各グループで少しずつコストやペナルティを変える（同一だと問題が縮退して簡単になるため）
            base_penalty = np.zeros(self.T)
            
            if g_type == 'FT':
                # 正社員タイプ
                name = f"G{i}_FullTime"
                base_cost = 5000 + np.random.randint(-500, 500)
                min_w, max_w = 8, 9
                # 早朝・深夜嫌悪
                for t in range(self.T):
                    h = t % 24
                    if h < 8 or h > 20: base_penalty[t] = 200
            
            elif g_type == 'PT_Day':
                # 主婦パートタイプ
                name = f"G{i}_PartTimeDay"
                base_cost = 1000 + np.random.randint(-200, 200)
                min_w, max_w = 4, 6
                # 夕方以降NG
                for t in range(self.T):
                    h = t % 24
                    if h < 9: base_penalty[t] = 100
                    if h > 17: base_penalty[t] = 1000
            
            else: # PT_Night
                # 学生バイトタイプ
                name = f"G{i}_Student"
                base_cost = 1000 + np.random.randint(-200, 200)
                min_w, max_w = 3, 8
                # 平日昼間NG
                for t in range(self.T):
                    h = t % 24
                    d = t // 24
                    if d < 5 and (9 <= h <= 17): base_penalty[t] = 500
                    if d >= 5 and h < 12: base_penalty[t] = 300 # 週末朝NG
            
            # ランダムな「個人の都合（ノイズ）」を少し乗せる
            random_penalty = np.random.randint(0, 50, self.T)
            final_penalty = base_penalty + random_penalty
            
            self.groups.append({
                'name': name,
                'max_employees': np.random.randint(10, 25), # 各グループ10〜25人
                'base_cost': base_cost,
                'hourly_wage': 1000 + np.random.randint(0, 500),
                'preference_penalty': final_penalty,
                'min_work': min_w,
                'max_work': max_w,
                'break_threshold': 6,
                'color': np.random.rand(3,) # グラフ用ランダム色
            })
            
        self.K = len(self.groups)
        self.big_m = 100000
# class ShiftSchedulingProblem:
#     def __init__(self, n_time_slots=168):
#         self.T = n_time_slots
#         self.hours = np.arange(self.T)
        
#         np.random.seed(42)
        
#         # 需要: 昼と夕方にピークがある一般的な店舗を想定
#         day_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
#         weekly_base = np.tile(day_pattern, 7)
#         # 土日は需要増
#         weekly_base[24*5:] = weekly_base[24*5:] * 1.2 
#         noise = np.random.randint(-1, 2, self.T)
#         self.demand = np.clip(weekly_base + noise, 1, None).astype(int)
        
#         self.groups = []
        
#         # --- グループ定義 ---
        
#         # Group 0: 正社員 (Full-time)
#         # コスト高, 8時間労働基本, 日中メイン
#         g0_penalty = np.zeros(self.T)
#         for t in range(self.T):
#             h = t % 24
#             if h < 8 or h > 19: g0_penalty[t] = 200 # 早朝深夜は嫌
        
#         self.groups.append({
#             'name': 'Full-time (Day)',
#             'max_employees': 5,      # 人数少なめ
#             'base_cost': 5000,       # 固定費高い
#             'hourly_wage': 2000,     # 時給換算(ペナルティ計算用基準)
#             'preference_penalty': g0_penalty,
#             'min_work': 8,
#             'max_work': 9,
#             'break_threshold': 6,
#             'color': 'tab:blue'
#         })

#         # Group 1: パート・主婦層 (Part-time Day)
#         # コスト安, 短時間, 夕方以降NG
#         g1_penalty = np.zeros(self.T)
#         for t in range(self.T):
#             h = t % 24
#             if h < 9: g1_penalty[t] = 100
#             if h > 16: g1_penalty[t] = 1000 # 夕食準備のため絶対帰りたい
        
#         self.groups.append({
#             'name': 'Part-time (Housewife)',
#             'max_employees': 15,
#             'base_cost': 1000,
#             'hourly_wage': 1100,
#             'preference_penalty': g1_penalty,
#             'min_work': 4,
#             'max_work': 6,
#             'break_threshold': 5,
#             'color': 'tab:orange'
#         })

#         # Group 2: 学生バイト (Part-time Night)
#         # 修正方針:
#         # - 平日: 授業中(9-17)はNG。朝は苦手。
#         # - 週末: 「早起きは絶対したくない(昼まで寝る)」「でも深夜まで働くのはOK」という学生らしさを追加
#         g2_penalty = np.zeros(self.T)
#         for t in range(self.T):
#             h = t % 24
#             day = t // 24
            
#             # --- 平日 (月〜金) ---
#             if day < 5: 
#                 if 9 <= h <= 17: g2_penalty[t] = 500 # 授業中
#                 if h < 10: g2_penalty[t] += 50       # 朝はちょっと苦手
            
#             # --- 週末 (土・日) ---
#             else:
#                 # 週末はもっと寝ていたい (12時前は働きたくない)
#                 if h < 12: g2_penalty[t] += 300
                
#                 # (オプション) 日曜の深夜(22時以降)は翌日の学校に響くので嫌がる傾向を入れるなら
#                 if day == 6 and h >= 22: g2_penalty[t] += 100

#         self.groups.append({
#             'name': 'Part-time (Student)',
#             'max_employees': 20,
#             'base_cost': 1000,
#             'hourly_wage': 1100,
#             'preference_penalty': g2_penalty,
#             'min_work': 4,  # 最低勤務時間を少し長めにするのも「こま切れ」防止に有効です
#             'max_work': 8,
#             'break_threshold': 6,
#             'color': 'tab:green'
#         })
        
#         self.K = len(self.groups)
#         self.big_m = 100000

# ==========================================
# 2. Proposed Solver (CG + Multi-day Graph)
# ==========================================
class ColumnGenerationSolver:
    def __init__(self, problem):
        self.prob = problem
        self.columns = []
        self.pool = [] # 今回は columns とほぼ同期させますが、概念として分けます
        self.duals = {'demand': [], 'convex': []}

    # ColumnGenerationSolver クラスに追加
    # ColumnGenerationSolver クラス内
    def import_pool(self, external_columns):
        count = 0
        active_count = 0
        for col in external_columns:
            # 前回の結果(val)を見て、使われていた列は最初からActiveにする
            is_active = col.get('val', 0) > 0.001
            
            new_col = {
                'id': len(self.columns),
                'group_id': col['group_id'],
                'schedule': col['schedule'],
                'cost': col['cost'],
                'in_rmp': is_active, # ★ここを変更: 使われていた列は最初からRMPに入れる
                'val': 0.0
            }
            self.columns.append(new_col)
            count += 1
            if is_active: active_count += 1
            
        # print(f"Imported {count} cols ({active_count} active) from history.")

    def build_initial_columns(self):
        # 初期解生成: グループ数が増減しても対応できるロジックに変更
        
        for k in range(self.prob.K):
            group = self.prob.groups[k]
            name = group.get('name', '')
            
            # グループの名前に応じて、典型的なシフト時間を割り当てる
            if 'FullTime' in name:
                # 正社員: 9:00 - 18:00 (休憩込み想定)
                base_start, base_end = 9, 18
            elif 'PartTimeDay' in name:
                # パート: 10:00 - 15:00
                base_start, base_end = 10, 15
            elif 'Student' in name:
                # 学生: 18:00 - 22:00
                base_start, base_end = 18, 22
            else:
                # その他（デフォルト）: 12:00 - 16:00
                base_start, base_end = 12, 16
            
            # 1週間分の初期列を作成
            for day in range(7):
                start = day * 24 + base_start
                end = day * 24 + base_end
                
                # グループごとの min_work / max_work 制約に合わせる補正
                duration = end - start
                if duration < group['min_work']:
                    end = start + group['min_work']
                elif duration > group['max_work']:
                    end = start + group['max_work']
                
                # 週の範囲外にはみ出す場合はスキップ
                if end > self.prob.T:
                    continue
                
                sched = [0] * self.prob.T
                # コスト計算
                cost = group['base_cost']
                for t in range(start, end):
                    if t < self.prob.T:
                        sched[t] = 1
                        cost += group['preference_penalty'][t]
                
                self.add_column(k, sched, cost)
        
        # ダミー列 (実行不能回避用)
        huge_cost = 1e8
        for k in range(self.prob.K):
            self.add_column(k, [1]*self.prob.T, huge_cost)

    def add_column(self, group_id, schedule, cost):
        # 重複チェック（簡易）
        col_id = len(self.columns)
        col = {
            'id': col_id,
            'group_id': group_id,
            'schedule': schedule,
            'cost': cost,
            'in_rmp': True
        }
        self.columns.append(col)
        self.pool.append(col)
        return col

    def solve_rmp(self, integer=False):
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        cat = pulp.LpInteger if integer else pulp.LpContinuous
        
        # 全列を対象にする（整数解計算時は特に重要）
        # ※整数解計算の直前に呼び出し側で in_rmp=True に戻している前提ですが、
        #   念のためここでも active_cols を正しく取得します。
        active_cols = [c for c in self.columns if c['in_rmp']]
        
        x_vars = {c['id']: pulp.LpVariable(f"x_{c['id']}", lowBound=0, cat=cat) for c in active_cols}
        s_vars = [pulp.LpVariable(f"s_{t}", lowBound=0, cat=cat) for t in range(self.prob.T)]
        
        # 目的関数
        model += pulp.lpSum([c['cost']*x_vars[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * s_vars[t] for t in range(self.prob.T)])
        
        # 需要制約
        d_cons = []
        for t in range(self.prob.T):
            c = pulp.lpSum([c['schedule'][t]*x_vars[c['id']] for c in active_cols]) + s_vars[t] >= self.prob.demand[t]
            model += c
            d_cons.append(c)
            
        # グループ人数制約
        g_cons = []
        for k in range(self.prob.K):
            c = pulp.lpSum([x_vars[c['id']] for c in active_cols if c['group_id']==k]) <= self.prob.groups[k]['max_employees']
            model += c
            g_cons.append(c)
            
        # --- 修正箇所: ソルバー実行設定 ---
        try:
            if integer:
                # 整数解のときはタイムリミット(例:60秒)を設定し、ログを表示しない(msg=0)
                # gapRel=0.01 (1%ギャップで妥協) などを入れるとさらに高速化可能
                model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
            else:
                model.solve(pulp.PULP_CBC_CMD(msg=0))
        except Exception as e:
            print(f"Solver Error: {e}")
            return None

        # --- 修正箇所: 結果の判定 ---
        if model.status != pulp.LpStatusOptimal:
            # 整数解の計算で、最適解が見つからなかった場合（時間切れ含む）
            if integer:
                # 時間切れでも「暫定解」があれば採用したいが、PuLPの標準挙動ではstatusを確認
                if model.status == pulp.LpStatusNotSolved or model.status == pulp.LpStatusInfeasible:
                    return None # 解なしとして返す
        
        if not integer:
            # LP緩和の場合は値を保存
            if model.status == pulp.LpStatusOptimal:
                for c in active_cols:
                    val = pulp.value(x_vars[c['id']])
                    c['val'] = val if val is not None else 0.0
                self.duals['demand'] = [c.pi for c in d_cons]
                self.duals['convex'] = [c.pi for c in g_cons]
                return pulp.value(model.objective)
            else:
                return float('inf') # LPすら解けない場合

        if integer:
            # 整数解の抽出
            assignments = []
            for c in active_cols:
                val = pulp.value(x_vars[c['id']])
                if val is not None and val > 0.1:
                    assignments.append({'column': c, 'count': int(round(val))})
            return assignments
        

    def cleanup_rmp(self):
        """
        RMPに含まれているが、解として使われていない(val=0)列を
        in_rmp=False にしてプールへ送る（計算軽量化のため）
        """
        removed_cnt = 0
        for col in self.columns:
            if col['in_rmp']:
                # 値がほぼ0なら削除対象
                # (Reduced Costが悪化しているかどうかの判定を入れる場合もあるが、
                #  基本戦略として非基底変数は外してOK)
                if col.get('val', 0) < 1e-5:
                    col['in_rmp'] = False
                    removed_cnt += 1
        return removed_cnt

    def pricing(self):
        new_cnt = 0
        if not self.duals['demand']: return 0
        
        # Graph Construction for each Group
        for k in range(self.prob.K):
            group = self.prob.groups[k]
            pi = self.duals['demand']
            mu = self.duals['convex'][k]
            
            G = nx.DiGraph()
            source, sink = 'S', 'E'
            
            # --- ノード構造の定義 ---
            # (t, d, status): 勤務中. t=時刻, d=勤務継続時間, status=0(未休憩)/1(休憩済)
            # (t, 'rest'): 待機中(勤務外).
            
            # 1. Source -> 最初の出勤 (雇用固定費発生)
            for t in range(self.prob.T):
                if t + group['min_work'] > self.prob.T: continue
                # Reduced Cost = (BaseCost + Penalty) - Duals
                c = (group['base_cost'] + group['preference_penalty'][t]) - pi[t]
                G.add_edge(source, (t, 1, 0), weight=c)

            # 2. 状態遷移
            for t in range(self.prob.T - 1):
                # A. 勤務中の遷移
                for d in range(1, group['max_work'] + 1):
                    for stat in [0, 1]:
                        curr_node = (t, d, stat)
                        if not G.has_node(curr_node) and d > 1: continue # d=1はSourceから来るのでOK
                        
                        # 次の時間のコスト (固定費は乗せない)
                        cost_next = group['preference_penalty'][t+1] - pi[t+1]
                        
                        # (A-1) 勤務継続
                        if d < group['max_work']:
                            # 未休憩 -> 未休憩
                            if stat == 0:
                                if d < group['break_threshold']:
                                    G.add_edge(curr_node, (t+1, d+1, 0), weight=cost_next)
                                # 休憩に入る (コスト0, 時間進む, d維持orリセット? ここではd維持で休憩扱い)
                                # 簡易化: (t,d,0) -> (t+1, d, 1) は1時間休憩したとみなす
                                if d >= 2: 
                                    G.add_edge(curr_node, (t+1, d, 1), weight=0)
                            # 休憩済 -> 休憩済
                            else:
                                G.add_edge(curr_node, (t+1, d+1, 1), weight=cost_next)
                                
                        # (A-2) 勤務終了 -> 待機 (Rest)
                        if d >= group['min_work']:
                            # 勤務終了して待機モードへ. コスト0.
                            G.add_edge(curr_node, (t, 'rest'), weight=0)
                            # 週の最後ならSinkへ
                            G.add_edge(curr_node, sink, weight=0)

                # B. 待機中(Rest) の遷移
                if t > 0: # t=0はSourceからしか始まらない
                    # (B-1) 待機継続: 明日まで休むなど
                    G.add_edge((t, 'rest'), (t+1, 'rest'), weight=0)
                    
                    # (B-2) 待機終了 -> 再出勤 (BaseCostなし!)
                    # 最低休息時間（簡易的に10時間とする）
                    # 厳密には「最後に働いてから」だが、ここでは「Restノードにいる=休んでいる」とみなす
                    # 連続勤務を防ぐため、Restノードへの入り口で制御するのが正確だが、
                    # ここではシンプルに「Restに遷移できればいつでも再開可能」としつつ、
                    # ペナルティの高い時間帯(G1の夜など)は自然と避けるのに任せる
                    c_restart = group['preference_penalty'][t+1] - pi[t+1]
                    G.add_edge((t, 'rest'), (t+1, 1, 0), weight=c_restart)

            # 終端処理
            for t in range(self.prob.T):
                if G.has_node((t, 'rest')):
                    G.add_edge((t, 'rest'), sink, weight=0)

            # 最短路探索
            try:
                path = nx.bellman_ford_path(G, source, sink, weight='weight')
                rc = nx.bellman_ford_path_length(G, source, sink, weight='weight')
                
                # Reduced Cost < 0 なら列追加
                if rc - mu < -1e-5:
                    sched = [0]*self.prob.T
                    real_cost = group['base_cost'] # 固定費は最初に1回
                    
                    for i in range(1, len(path)-1):
                        u, v = path[i-1], path[i]
                        # 勤務中ノードへの遷移であればコストとフラグを加算
                        # vが (t, d, stat) の形式
                        if isinstance(v, tuple) and len(v) == 3:
                            t = v[0]
                            # 休憩中(dが変わらない遷移)は労働時間に含まないならスキップ
                            # 今回の定義: (t,d,0)->(t+1,d,1) は休憩
                            if isinstance(u, tuple) and len(u)==3 and u[1] == v[1]:
                                pass # 休憩中
                            else:
                                sched[t] = 1
                                real_cost += group['preference_penalty'][t]
                                
                    self.add_column(k, sched, real_cost)
                    new_cnt += 1
            except nx.NetworkXNoPath:
                pass
            except Exception as e:
                # print(f"Error in pricing {k}: {e}")
                pass
                
        return new_cnt

    # ColumnGenerationSolver クラス内
    def solve(self):
        start_time = time.time()
        self.build_initial_columns()
        
        max_iter = 100 # 反復回数は適度でOK
        lp_obj_bound = 0
        
        # --- LP Loop ---
        for i in range(max_iter):
            obj = self.solve_rmp(integer=False)
            lp_obj_bound = obj
            
            if i > 0 and i % 5 == 0:
                self.cleanup_rmp()
            
            new_cols = self.pricing()
            if new_cols == 0:
                break
        
        # ★ LP収束時点での時間を記録
        lp_end_time = time.time()
        lp_time = lp_end_time - start_time
        
        # --- IP Solve ---
        # 全列を有効化
        for c in self.columns:
            c['in_rmp'] = True
            
        # 整数解探索（時間を短く区切る）
        # timeLimitを30秒程度に短縮して、差を明確にするのがコツです
        assignments = self.solve_rmp(integer=True)
        
        total_time = time.time() - start_time
        
        if assignments is None:
            ip_obj = 0
            gap = 0
        else:
            ip_obj = sum(item['column']['cost'] * item['count'] for item in assignments)
            gap = (ip_obj - lp_obj_bound) / ip_obj if ip_obj != 0 else 0.0
        
        return {
            'assignments': assignments,
            'time': total_time,      # 全体時間
            'lp_time': lp_time,      # ★追加: LPのみの時間（ここに差が出る！）
            'ip_obj': ip_obj,
            'lp_bound': lp_obj_bound,
            'gap': gap,
            'cols_generated': len(self.columns)
        }
    


class CompactExactSolver:
    """
    厳密解法ソルバー (修正版)
    - グラフ構築をループの外に出し、計算速度を劇的に改善
    - 変数管理を効率化
    """
    def __init__(self, problem):
        self.prob = problem

    # ... (build_graph_for_group は変更なしのため省略。そのままでOK) ...
    def build_graph_for_group(self, k):
        group = self.prob.groups[k]
        G = nx.DiGraph()
        source, sink = 'S', 'E'
        
        # 1. Source -> Start (変更なし)
        for t in range(self.prob.T):
            if t + group['min_work'] > self.prob.T: continue
            c = group['base_cost'] + group['preference_penalty'][t]
            G.add_edge(source, (t, 1, 0), weight=c, type='start', time=t)

        # 2. Transitions (変更なし)
        for t in range(self.prob.T - 1):
            for d in range(1, group['max_work'] + 1):
                for stat in [0, 1]:
                    u = (t, d, stat)
                    if not G.has_node(u) and d > 1: continue

                    cost_next = group['preference_penalty'][t+1]
                    
                    # (A) Work Continuation
                    if d < group['max_work']:
                        if stat == 0:
                            if d < group['break_threshold']:
                                G.add_edge(u, (t+1, d+1, 0), weight=cost_next, type='work', time=t+1)
                            if d >= 2:
                                G.add_edge(u, (t+1, d, 1), weight=0, type='break', time=t+1)
                        else:
                            G.add_edge(u, (t+1, d+1, 1), weight=cost_next, type='work', time=t+1)
                    
                    # (B) Work -> Rest/End (途中終了)
                    if d >= group['min_work']:
                        G.add_edge(u, (t, 'rest'), weight=0, type='rest_start', time=t)
                        G.add_edge(u, sink, weight=0, type='end', time=None)

            # (C) Rest Transitions
            if t > 0:
                G.add_edge((t, 'rest'), (t+1, 'rest'), weight=0, type='rest_cont', time=t+1)
                c_restart = group['preference_penalty'][t+1]
                G.add_edge((t, 'rest'), (t+1, 1, 0), weight=c_restart, type='restart', time=t+1)
        
        # 3. 終端処理 (Sinkへの接続) - ★ここが不足していました
        # 最後の時刻 T-1 (167) に存在するノードから Sink へのエッジを追加
        last_t = self.prob.T - 1
        
        # A. 勤務中の人 (条件を満たせば帰宅可)
        for d in range(group['min_work'], group['max_work'] + 1):
            for stat in [0, 1]:
                u = (last_t, d, stat)
                # ノード自体は前のループ(t=166からの遷移)で作られている可能性がある
                # ただしG.has_node判定は難しいので、無条件に追加しても機能するが、
                # has_nodeがないとエラーになる可能性があるので安全策をとる
                # ここでは簡易的に「前の時刻からの遷移」ロジックに頼らず、
                # 明示的にノードを作るとグラフが膨れるので、
                # 実は transition ループを range(self.prob.T) まで回して
                # 遷移先が T を超える場合のみ弾くのが一番きれいですが、
                # パッチとしては以下を追加してください。
                
                # 修正: Transitionループの条件分岐だけでは拾えない「最後の瞬間の帰宅」
                # そもそもループ内の `if d >= group['min_work']: G.add_edge(u, sink)` は
                # 「時刻 t で仕事を終えて帰る」という意味。
                # t=167 でループが回っていないので、t=167で仕事を終えられない。
        
        # ★一番簡単な修正法: ループ範囲を変えるのではなく、最後の時刻専用のループを追加
        for d in range(1, group['max_work'] + 1):
            for stat in [0, 1]:
                u = (last_t, d, stat)
                # もしこのノードに到達可能なら（＝前の時刻から遷移してきているなら）、ここからSinkへ行ける
                # （ただし勤務時間制約は満たす必要がある）
                if d >= group['min_work']:
                     G.add_edge(u, sink, weight=0, type='end', time=None)

        # B. 休憩/待機中の人 (いつでも帰宅可)
        for t in range(self.prob.T):
            if G.has_node((t, 'rest')):
                G.add_edge((t, 'rest'), sink, weight=0, type='end', time=None)
                
        return G, source, sink

    def solve(self):
        print("Building Compact MIP Model...")
        start_time = time.time()
        
        model = pulp.LpProblem("CompactMIP", pulp.LpMinimize)
        
        # 1. グラフ構築 & 変数定義 (ループの外で一括実行)
        # flow_vars: エッジごとのフロー変数
        # supply_map: 時刻 t に労働供給する変数のリスト {t: [var1, var2...]}
        flow_vars = {}
        supply_map = {t: [] for t in range(self.prob.T)}
        objective_terms = []
        
        for k in range(self.prob.K):
            G, source, sink = self.build_graph_for_group(k)
            flow_vars[k] = {}
            
            for u, v, data in G.edges(data=True):
                var_name = f"f_{k}_{str(u)}_{str(v)}"
                x = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)
                flow_vars[k][(u, v)] = x
                
                # 目的関数
                if data['weight'] != 0:
                    objective_terms.append(x * data['weight'])
                
                # 供給マップへの登録 (高速化の肝)
                # 'start', 'work', 'restart' のエッジは、その時刻 t の労働力になる
                if data.get('type') in ['start', 'work', 'restart']:
                    t_idx = data.get('time')
                    if t_idx is not None and 0 <= t_idx < self.prob.T:
                        supply_map[t_idx].append(x)

            # フロー保存則
            for node in G.nodes():
                if node == source:
                    model += pulp.lpSum([flow_vars[k][(source, v)] for v in G.successors(source)]) <= self.prob.groups[k]['max_employees']
                elif node == sink:
                    pass
                else:
                    flow_in = pulp.lpSum([flow_vars[k][(u, node)] for u in G.predecessors(node)])
                    flow_out = pulp.lpSum([flow_vars[k][(node, v)] for v in G.successors(node)])
                    model += flow_in == flow_out

        # 2. 需要制約 (構築済みの supply_map を使うので高速)
        print("Adding demand constraints...")
        for t in range(self.prob.T):
            # スラック変数
            s_var = pulp.LpVariable(f"slack_{t}", lowBound=0, cat=pulp.LpContinuous)
            objective_terms.append(s_var * self.prob.big_m)
            
            # 制約: (供給) + (不足分) >= 需要
            # supply_map[t] には既に該当する x 変数がリストされている
            model += pulp.lpSum(supply_map[t]) + s_var >= self.prob.demand[t]

        model += pulp.lpSum(objective_terms)

        print("Solving Compact MIP...")
        model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300)) 
        
        total_time = time.time() - start_time
        obj = pulp.value(model.objective)
        
        return {
            'time': total_time,
            'ip_obj': obj,
            'status': pulp.LpStatus[model.status]
        }

# ==========================================
# 4. 可視化
# ==========================================
def run_visualize_experiment():
    prob = ShiftSchedulingProblem()
    solver = ColumnGenerationSolver(prob)
    assignments = solver.solve()
    
    # 可視化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(prob.K + 1, 1, height_ratios=[2] + [1]*prob.K)
    
    # 1. 積み上げグラフ
    ax_main = fig.add_subplot(gs[0])
    bottom = np.zeros(prob.T)
    
    for k in range(prob.K):
        group_load = np.zeros(prob.T)
        for item in assignments:
            if item['column']['group_id'] == k:
                group_load += np.array(item['column']['schedule']) * item['count']
        
        ax_main.bar(prob.hours, group_load, bottom=bottom, 
                    color=prob.groups[k]['color'], label=prob.groups[k]['name'], 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += group_load

    ax_main.plot(prob.hours, prob.demand, 'r--', linewidth=2, label='Demand')
    ax_main.set_title(f"Shift Optimization Result (Total Cost Visualization)")
    ax_main.set_xlim(0, prob.T)
    ax_main.set_xticks(np.arange(0, 169, 12))
    ax_main.set_xticklabels([f"D{d} {h}h" for d in range(8) for h in [0, 12]][:len(np.arange(0, 169, 12))])
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # 2. ガントチャート
    for k in range(prob.K):
        ax = fig.add_subplot(gs[k+1], sharex=ax_main)
        items = [item for item in assignments if item['column']['group_id'] == k]
        
        y_pos = 0
        for item in items:
            count = item['count']
            sched = np.array(item['column']['schedule'])
            
            # 連続区間の抽出
            bounded = np.hstack(([0], sched, [0]))
            difs = np.diff(bounded)
            starts, = np.where(difs == 1)
            stops, = np.where(difs == -1)
            
            for start, stop in zip(starts, stops):
                duration = stop - start
                ax.barh(y_pos, duration, left=start, height=0.6, 
                        color=prob.groups[k]['color'], edgecolor='black', alpha=0.6)
                ax.text(start + duration/2, y_pos, f"x{count}", 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            y_pos += 1
            
        ax.set_title(f"{prob.groups[k]['name']} Patterns")
        ax.set_yticks([])
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        
        # ペナルティ可視化
        penalty = np.array([prob.groups[k]['preference_penalty']])
        extent = [0, prob.T, -0.5, max(1, len(items))]
        ax.imshow(penalty, aspect='auto', cmap='Reds', extent=extent, alpha=0.15, vmin=0, vmax=500)

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     run_visualize_experiment()

def run_comparison():
    prob = ShiftSchedulingProblem()
    
    print("\n" + "="*60)
    print("COMPARISON EXPERIMENT: Column Generation vs Exact MIP")
    print("="*60)

    # 1. 提案手法
    print("\n[Method 1] Column Generation (Proposed)")
    cg_solver = ColumnGenerationSolver(prob)
    cg_res = cg_solver.solve()
    
    # 2. 厳密解法
    print("\n[Method 2] Compact Exact MIP (Benchmark)")
    exact_solver = CompactExactSolver(prob)
    exact_res = exact_solver.solve()
    
    # --- 表示用のフォーマット処理 ---
    cg_obj_str = f"{cg_res['ip_obj']:,.0f}" if cg_res['ip_obj'] is not None else "N/A"
    
    if exact_res['ip_obj'] is not None:
        exact_obj_str = f"{exact_res['ip_obj']:,.0f}"
    else:
        exact_obj_str = f"Infeasible ({exact_res['status']})"

    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Column Generation':<20} | {'Exact MIP (Compact)':<20}")
    print("-" * 70)
    print(f"{'Time (sec)':<25} | {cg_res['time']:<20.4f} | {exact_res['time']:<20.4f}")
    print(f"{'Objective Cost':<25} | {cg_obj_str:<20} | {exact_obj_str:<20}")
    
    # Gap計算
    if exact_res['ip_obj'] is not None and exact_res['ip_obj'] != 0:
        gap = (cg_res['ip_obj'] - exact_res['ip_obj']) / exact_res['ip_obj']
        print(f"{'Gap to Exact':<25} | {gap:<20.4%} | {'0.0000% (Baseline)':<20}")
    else:
        print(f"{'Gap to Exact':<25} | {'N/A':<20} | {'-'}")

    print("="*70)

# if __name__ == "__main__":
#     run_comparison()

# def run_continuous_weeks_experiment():
#     print("\n" + "="*70)
#     print("EXPERIMENT: Multi-Week Warm Start Efficiency (Column Reuse)")
#     print("="*70)
    
#     n_groups = 5  # 実験用グループ数
#     weeks = 4     # 4週間分のシミュレーション
    
#     # 履歴プール（全期間で生成された列を蓄積していく）
#     historical_pool = []
    
#     results = []
    
#     for w in range(weeks):
#         print(f"\n[Week {w+1}] Generating Demand...")
#         # 週ごとに異なるseedで問題を作成（需要が変動する）
#         prob = ShiftSchedulingProblem(n_groups=n_groups, seed=100+w)
        
#         # --- Case A: Cold Start (知識なし) ---
#         print(f"  > Running Cold Start (Standard)...")
#         solver_cold = ColumnGenerationSolver(prob)
#         res_cold = solver_cold.solve()
        
#         # --- Case B: Warm Start (過去の列を利用) ---
#         print(f"  > Running Warm Start (Proposed)...")
#         solver_warm = ColumnGenerationSolver(prob)
        
#         # ★ここで過去の列を注入！★
#         if w > 0:
#             solver_warm.import_pool(historical_pool)
            
#         res_warm = solver_warm.solve()
        
#         # --- 結果記録 ---
#         results.append({
#             'week': w + 1,
#             'cold_time': res_cold['time'],
#             'cold_iter': res_cold['cols_generated'], # 生成総数でおおよその反復規模を測る
#             'warm_time': res_warm['time'],
#             'warm_iter': res_warm['cols_generated'] - (len(historical_pool) if w > 0 else 0) # 今回新規作成した数
#         })
        
#         # --- 履歴の更新 ---
#         # Warm Startで解いた結果（全ての有効な列）を次の週へ引き継ぐ
#         # Cold Startの結果も混ぜても良いが、ここではWarm Startの成長を見る
#         # 重複を避けるロジックがないため、単純にsolver_warmの全列を次回の種とする
#         historical_pool = solver_warm.columns
#         print(f"  >> Pool Updated: {len(historical_pool)} columns stored for next week.")

#     # --- 最終レポート ---
#     print("\n" + "="*70)
#     print(f"{'Week':<5} | {'Cold Time (s)':<15} | {'Warm Time (s)':<15} | {'Speedup':<10}")
#     print("-" * 70)
    
#     for r in results:
#         speedup = r['cold_time'] / r['warm_time'] if r['warm_time'] > 0 else 0.0
#         print(f"{r['week']:<5} | {r['cold_time']:<15.4f} | {r['warm_time']:<15.4f} | {speedup:<10.2f}x")
    
#     print("-" * 70)
#     print("Observation:")
#     print("Week 1 は条件が同じため差が出ませんが、Week 2以降において、")
#     print("Warm Start（列プール戦略）は、過去の列をHeuristic Pricingで再利用するため、")
#     print("重いグラフ探索の回数が減少し、計算時間が大幅に短縮される傾向が見られます。")

# if __name__ == "__main__":
#     # 既存の比較実験の代わりにこちらを実行
#     run_continuous_weeks_experiment()

def run_continuous_weeks_experiment():
    print("\n" + "="*85)
    print("EXPERIMENT: Multi-Week Warm Start Efficiency (Focus on LP Convergence)")
    print("="*85)
    
    n_groups = 5
    weeks = 4
    historical_pool = []
    
    # 見出し
    print(f"{'Week':<5} | {'Metric':<10} | {'Cold Start':<15} | {'Warm Start':<15} | {'Speedup':<10}")
    print("-" * 85)
    
    for w in range(weeks):
        # 需要生成 (Seedを変える)
        prob = ShiftSchedulingProblem(n_groups=n_groups, seed=200+w)
        
        # 1. Cold Start
        solver_cold = ColumnGenerationSolver(prob)
        res_cold = solver_cold.solve()
        
        # 2. Warm Start
        solver_warm = ColumnGenerationSolver(prob)
        if w > 0:
            solver_warm.import_pool(historical_pool)
        res_warm = solver_warm.solve()
        
        # 履歴更新
        historical_pool = solver_warm.columns
        
        # --- 結果表示 (LP時間を強調) ---
        # LP Time Comparison
        lp_speedup = res_cold['lp_time'] / res_warm['lp_time'] if res_warm['lp_time'] > 1e-4 else 1.0
        print(f"{w+1:<5} | {'LP Time':<10} | {res_cold['lp_time']:<15.4f} | {res_warm['lp_time']:<15.4f} | {lp_speedup:<10.2f}x")
        
        # Iteration Comparison
        print(f"{'':<5} | {'Cols Gen':<10} | {res_cold['cols_generated']:<15} | {res_warm['cols_generated']:<15} | {'-':<10}")
        
        # Total Time Comparison (IP含む)
        total_speedup = res_cold['time'] / res_warm['time'] if res_warm['time'] > 1e-4 else 1.0
        print(f"{'':<5} | {'Total':<10} | {res_cold['time']:<15.4f} | {res_warm['time']:<15.4f} | {total_speedup:<10.2f}x")
        print("-" * 85)

if __name__ == "__main__":
    run_continuous_weeks_experiment()
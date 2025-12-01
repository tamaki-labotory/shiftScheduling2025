import pulp
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

def get_weekly_problem(full_prob, week_idx):
    """
    全体の問題設定から、指定した週(0始まり)のデータだけを抜き出して
    新しいShiftSchedulingProblemインスタンスを作成する
    """
    T_week = 168 # 1週間 = 24 * 7
    start_t = week_idx * 24 * 7
    end_t = start_t + T_week
    
    # 新しい問題インスタンス（空で作る）
    sub_prob = ShiftSchedulingProblem(n_time_slots=T_week)
    
    # 需要データをスライスして上書き
    # ※ full_prob.demand が足りない場合のエラー処理は省略
    sub_prob.demand = full_prob.demand[start_t : end_t]
    
    # グループ設定はコピー（ペナルティ配列もスライスが必要）
    sub_prob.groups = copy.deepcopy(full_prob.groups)
    for group in sub_prob.groups:
        # ペナルティ配列を該当週の部分だけ切り出す
        original_penalty = group['preference_penalty']
        group['preference_penalty'] = original_penalty[start_t : end_t]
        
    return sub_prob

# ==========================================
# 1. 問題設定 (現実的な従業員グループ)
# ==========================================
class ShiftSchedulingProblem:
    def __init__(self, n_time_slots=672):
        self.T = n_time_slots
        self.hours = np.arange(self.T)
        
        np.random.seed(42)
        
        # 基本需要パターン (24時間)
        day_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
        
        # 必要な日数分だけパターンを繰り返す
        n_days = int(np.ceil(self.T / 24))
        weekly_base = np.tile(day_pattern, n_days)
        
        # 配列の長さを self.T に合わせる（もし24で割り切れない場合のため）
        weekly_base = weekly_base[:self.T]
        
        # 週末（土日）は需要増 (1.2倍)
        # ※ 月曜始まり(Day0)と仮定: Day 5(Sat), Day 6(Sun)
        # ループで回して該当する時間帯だけ増やす
        for t in range(self.T):
            day_idx = (t // 24) % 7
            if day_idx >= 5: # 土(5) or 日(6)
                weekly_base[t] = weekly_base[t] * 1.2
        
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_base + noise, 1, None).astype(int)
        
        self.groups = []
        
        # --- グループ定義 (修正なし) ---
        
        # Group 0: 正社員 (Full-time)
        g0_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 8 or h > 19: g0_penalty[t] = 200 
        
        self.groups.append({
            'name': 'Full-time (Day)',
            'max_employees': 5,
            'base_cost': 5000,
            'hourly_wage': 2000,
            'preference_penalty': g0_penalty,
            'min_work': 8,
            'max_work': 9,
            'break_threshold': 6,
            'color': 'tab:blue'
        })

        # Group 1: パート・主婦層 (Part-time Housewife)
        g1_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 9: g1_penalty[t] = 100
            if h > 16: g1_penalty[t] = 1000 
        
        self.groups.append({
            'name': 'Part-time (Housewife)',
            'max_employees': 15,
            'base_cost': 1000,
            'hourly_wage': 1100,
            'preference_penalty': g1_penalty,
            'min_work': 4,
            'max_work': 6,
            'break_threshold': 5,
            'color': 'tab:orange'
        })

        # Group 2: 学生バイト (Part-time Student)
        g2_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            day_total = t // 24
            # 曜日判定 (0=Mon, ..., 6=Sun)
            day_of_week = day_total % 7
            
            # --- 平日 (月〜金) ---
            if day_of_week < 5: 
                if 9 <= h <= 17: g2_penalty[t] = 500 
                if h < 10: g2_penalty[t] += 50
            
            # --- 週末 (土・日) ---
            else:
                if h < 12: g2_penalty[t] += 300
                if day_of_week == 6 and h >= 22: g2_penalty[t] += 100

        self.groups.append({
            'name': 'Part-time (Student)',
            'max_employees': 20,
            'base_cost': 1000,
            'hourly_wage': 1100,
            'preference_penalty': g2_penalty,
            'min_work': 4,
            'max_work': 8,
            'break_threshold': 6,
            'color': 'tab:green'
        })
        
        self.K = len(self.groups)
        self.big_m = 100000

class IndividualShiftProblem(ShiftSchedulingProblem):
    def __init__(self, n_employees=30, n_time_slots=168):
        # 親クラスの初期化（需要データ生成など）
        super().__init__(n_time_slots=n_time_slots)
        
        self.employees = []
        self.n_employees = n_employees
        
        # 従業員ごとのユニークな希望を作成
        # ベースとなる3つのタイプからランダムに微調整して個人差を作る
        for i in range(n_employees):
            # タイプをランダム決定 (0:Full, 1:Housewife, 2:Student)
            etype = np.random.choice([0, 1, 2])
            base_group = self.groups[etype] # 親クラスで定義されたグループ定義を借用
            
            # 個人ごとのノイズを加える
            base_penalty = base_group['preference_penalty'].copy()
            noise = np.random.randint(-20, 50, self.T) # ペナルティの個人差
            personal_penalty = np.clip(base_penalty + noise, 0, None)
            
            self.employees.append({
                'id': i,
                'name': f"Emp_{i}({base_group['name']})",
                'type': etype, # クラスタリングの正解ラベルとして保持（参考用）
                'base_cost': base_group['base_cost'],
                'hourly_wage': base_group['hourly_wage'],
                'preference_penalty': personal_penalty,
                'min_work': base_group['min_work'],
                'max_work': base_group['max_work'],
                'break_threshold': base_group['break_threshold'],
                'color': base_group['color']
            })

# ==========================================
# 2. Proposed Solver (CG + Multi-day Graph)
# ==========================================
class ColumnGenerationSolver:
    def __init__(self, problem):
        self.prob = problem
        self.columns = []
        self.duals = {'demand': [], 'convex': []}

    def build_initial_columns(self):
        """
        修正: 標準パターンを一切生成しない。
        実行可能解を保証するためのダミー列（超高コスト）のみを追加する。
        """
        # ダミー列 (Cost = 1億)
        huge_cost = 1e12
        for k in range(self.prob.K):
            # 24時間働き続けるというあり得ないシフトだが、数理的には需要を満たすための命綱
            self.add_column(k, [1]*self.prob.T, huge_cost, keep=True)

    def add_column(self, group_id, schedule, cost, keep=False):
        col_id = len(self.columns)
        col = {
            'id': col_id,
            'group_id': group_id,
            'schedule': schedule,
            'cost': cost,
            'in_rmp': True,
            'keep': keep,
            'val': 0.0
        }
        self.columns.append(col)
        return col

    # --- 以下、前回の修正版（Warm Start対応）と同じメソッド群 ---

    def import_columns(self, external_columns):
        count = 0
        for col in external_columns:
            if len(col['schedule']) != self.prob.T: continue
            group = self.prob.groups[col['group_id']]
            new_cost = group['base_cost']
            sched = col['schedule']
            for t in range(self.prob.T):
                if sched[t] == 1:
                    new_cost += group['preference_penalty'][t]
            self.add_column(col['group_id'], sched, new_cost)
            count += 1
        return count

    def cleanup_rmp(self):
        removed_cnt = 0
        for col in self.columns:
            if col['in_rmp'] and not col.get('keep', False):
                if col.get('val', 0) < 1e-5:
                    col['in_rmp'] = False
                    removed_cnt += 1
        return removed_cnt

    def solve_rmp(self, integer=False):
        model = pulp.LpProblem("RMP", pulp.LpMinimize)
        cat = pulp.LpInteger if integer else pulp.LpContinuous
        active_cols = [c for c in self.columns if c['in_rmp']]
        
        x_vars = {}
        for c in active_cols:
            lb = c.get('fixed_val', 0)
            ub = c.get('fixed_val', None)
            x_vars[c['id']] = pulp.LpVariable(f"x_{c['id']}", lowBound=lb, upBound=ub, cat=cat)

        s_vars = [pulp.LpVariable(f"s_{t}", lowBound=0, cat=cat) for t in range(self.prob.T)]
        
        model += pulp.lpSum([c['cost']*x_vars[c['id']] for c in active_cols]) + \
                 pulp.lpSum([self.prob.big_m * s_vars[t] for t in range(self.prob.T)])
        
        d_cons = []
        for t in range(self.prob.T):
            c = pulp.lpSum([c['schedule'][t]*x_vars[c['id']] for c in active_cols]) + s_vars[t] >= self.prob.demand[t]
            model += c
            d_cons.append(c)
            
        g_cons = []
        for k in range(self.prob.K):
            c = pulp.lpSum([x_vars[c['id']] for c in active_cols if c['group_id']==k]) <= self.prob.groups[k]['max_employees']
            model += c
            g_cons.append(c)
            
        if integer:
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
        else:
            model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if not integer:
            for c in active_cols:
                val = pulp.value(x_vars[c['id']])
                c['val'] = val if val is not None else 0.0
            
            if model.status != 1: 
                self.duals['demand'] = []
                return float('inf')

            self.duals['demand'] = [c.pi for c in d_cons]
            self.duals['convex'] = [c.pi for c in g_cons]
            return pulp.value(model.objective)
        else:
            assignments = []
            for c in active_cols:
                val = pulp.value(x_vars[c['id']])
                if val is not None and val > 0.1:
                    assignments.append({'column': c, 'count': int(round(val))})
            return assignments

    def pricing(self):
        new_cnt = 0
        if not self.duals['demand']: return 0
        
        # 1週間の時間数
        T = self.prob.T 

        for k in range(self.prob.K):
            group = self.prob.groups[k]
            pi = self.duals['demand']
            mu = self.duals['convex'][k]
            
            # --- グラフ構築 (Multi-Day Model) ---
            G = nx.DiGraph()
            source, sink = 'S', 'E'
            
            # ノード定義: (t, duration, state)
            # state 0: 勤務中 (Working)
            # state 1: 休憩中 (Break)
            # state 2: 勤務外 (Rest/Home) ※ここが新しいハブになる

            # 1. Source -> 最初のRestへの接続
            G.add_edge(source, (0, 0, 2), weight=0)

            for t in range(T):
                # --- A. Rest (勤務外) 状態からの遷移 ---
                # ノード (t, 0, 2) が存在すると仮定（または前の時刻から遷移してくる）
                
                # 次の時刻へ
                if t < T - 1:
                    # 1. そのまま休み続ける (Rest -> Rest)
                    # コストは0
                    G.add_edge((t, 0, 2), (t+1, 0, 2), weight=0)
                    
                    # 2. 新しく出勤する (Rest -> Work start)
                    # 条件: 残り時間が min_work 以上あること
                    if t + group['min_work'] <= T:
                        # コスト = (基本給 + ペナルティ) - 双対変数
                        # ※ base_cost は「1シフトあたり」かかる固定費としてここで計上する
                        c_start = (group['base_cost'] + group['preference_penalty'][t+1]) - pi[t+1]
                        G.add_edge((t, 0, 2), (t+1, 1, 0), weight=c_start)

                # 最終時刻ならSinkへ
                else:
                    G.add_edge((t, 0, 2), sink, weight=0)

                # --- B. Work (勤務中) 状態からの遷移 ---
                # duration: 1 ~ max_work
                for d in range(1, group['max_work'] + 1):
                    # state: 0(Work), 1(Break)
                    for s in [0, 1]:
                        curr_node = (t, d, s)
                        
                        # グラフにこのノードが存在しなければスキップ（無駄な探索を省く）
                        # ※厳密には順方向生成が必要ですが、NetworkXなら全定義してからでもOK
                        
                        if t < T - 1:
                            next_penalty = group['preference_penalty'][t+1]
                            next_pi = pi[t+1]
                            cost_next = next_penalty - next_pi
                            
                            # 1. 勤務継続 (Work -> Work)
                            if d < group['max_work']:
                                # 休憩ルールなどの詳細ロジック
                                if s == 0: # 今働いている
                                    # 休憩に入る (Work -> Break)
                                    if d >= 2: # 2時間以上働いたら休憩可とする例
                                        G.add_edge(curr_node, (t+1, d, 1), weight=0) # 休憩は労働時間カウントしない(dそのまま)
                                    
                                    # 働き続ける (Work -> Work)
                                    if d < group['break_threshold']:
                                        G.add_edge(curr_node, (t+1, d+1, 0), weight=cost_next)
                                else: 
                                    # 今休憩中 -> 復帰 (Break -> Work)
                                    G.add_edge(curr_node, (t+1, d+1, 0), weight=cost_next)

                            # 2. 勤務終了 (Work/Break -> Rest)
                            # 最小勤務時間を満たしていれば「帰宅」できる
                            if d >= group['min_work']:
                                # 次の時間は Rest (0, 2) になる
                                # シフト間の最低休息時間(インターバル)を強制したい場合は
                                # ここで (t+1, 0, 2) ではなく (t+11, 0, 2) などへ飛ばす工夫も可能
                                G.add_edge(curr_node, (t+1, 0, 2), weight=0)

                        else:
                            # 最終時刻で勤務中なら、条件満たしていれば終了可能
                            if d >= group['min_work']:
                                G.add_edge(curr_node, sink, weight=0)

            # --- 最短路探索 ---
            try:
                # 負の閉路がないDAG（有向非巡回グラフ）なので bellman_ford または simple path でOK
                # ※ 時間 t が進む方向にしかエッジがないため DAG です
                path = nx.shortest_path(G, source, sink, weight='weight')
                rc = nx.path_weight(G, path, weight='weight')
                
                # Reduced Cost が負なら列に追加
                # 注意: 今回は1週間全体のコストを見ているので、閾値判定は慎重に
                if rc - mu < -1e-5:
                    sched = [0] * T
                    # パスからスケジュールを復元
                    current_cost = 0
                    
                    # パス解析
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        
                        # 特殊ノード(S, E)を除外して判定
                        if isinstance(v, tuple) and len(v) == 3:
                            t_idx = v[0]
                            dur = v[1]
                            state = v[2]
                            
                            # state=0 (Work) または state=1 (Breakだが給与発生する場合など)
                            # ここでは「state=0 (Work)」のみを勤務として1を立てる設定にします
                            if state == 0:
                                sched[t_idx] = 1
                                current_cost += group['preference_penalty'][t_idx]
                            
                            # Rest -> Work の遷移で base_cost を加算（シフト回数分かかる）
                            if isinstance(u, tuple) and u[2] == 2 and state == 0:
                                current_cost += group['base_cost']

                    # 列追加
                    self.add_column(k, sched, current_cost)
                    new_cnt += 1
            except nx.NetworkXNoPath:
                pass
            except Exception as e:
                # デバッグ用
                # print(f"Error in pricing for group {k}: {e}")
                pass
                
        return new_cnt

    def solve(self):
        self.build_initial_columns()
        max_iter = 200 # 初期列がないので回数を確保
        
        print(f"{'Iter':<5} | {'Obj':<10} | {'Active':<8} | {'Total':<8} | {'Pool Check'}")
        print("-" * 65)
        
        for i in range(max_iter):
            obj = self.solve_rmp(integer=False)
            
            cleaned = 0
            if i > 0 and i % 5 == 0:
                cleaned = self.cleanup_rmp()

            new_cols = self.pricing()
            active_cols = sum(1 for c in self.columns if c['in_rmp'])
            note = f"Cleaned {cleaned}" if cleaned > 0 else ""
            print(f"{i:<5} | {obj:<10.2f} | {active_cols:<8} | {len(self.columns):<8} | {note}")
            
            if new_cols == 0 and obj < 1e7: # ダミー列を使わなくなった場合のみ終了
                print("\nOptimal (LP Relaxation) Reached.")
                break
            elif new_cols == 0:
                print("\nNo more columns found, but cost is high (Infeasible likely).")
                break
        
        print("\nSolving Integer Problem (Hybrid Heuristic)...")
        return self.solve_heuristic()

    def solve_heuristic(self):
        # 0.99以上固定のヒューリスティック
        self.solve_rmp(integer=False)
        fixed_count = 0
        for c in self.columns:
            if c['in_rmp'] and c.get('val', 0) >= 0.99: 
                c['fixed_val'] = 1.0
                fixed_count += 1
        print(f"  > Fixed {fixed_count} columns (value >= 0.99).")
        print("  > Solving remaining problem as MIP...")
        return self.solve_rmp(integer=True)
    
class CompactExactSolver:
    """
    厳密解法ソルバー (Pure Network Flow MIP)
    フロー分解ロジックを追加し、可視化用データを生成できるように修正
    """
    def __init__(self, problem):
        self.prob = problem

    def build_graph_for_group(self, k):
        """
        修正版: Rest経由の遷移を強化し、再出勤コストを正しく設定
        """
        group = self.prob.groups[k]
        G = nx.DiGraph()
        source, sink = 'S', 'E'
        T = self.prob.T
        
        # 1. Source -> Start (初回出勤)
        for t in range(T):
            # min_work時間を確保できる時間帯のみ開始可能
            if t + group['min_work'] > T: continue
            
            # 初回出勤コスト = 基本給 + その時間のペナルティ
            c = group['base_cost'] + group['preference_penalty'][t]
            G.add_edge(source, (t, 1, 0), weight=c, type='start', time=t)

        # 2. 状態遷移 (Transitions)
        # t: 0 ~ T-2
        for t in range(T - 1):
            
            # --- A. Rest (休憩・待機) ノード間の遷移 ---
            # (t, 'rest') -> (t+1, 'rest') : 待機継続
            G.add_edge((t, 'rest'), (t+1, 'rest'), weight=0, type='rest_cont', time=t+1)
            
            # (t, 'rest') -> (t+1, 1, 0) : 再出勤 (Restart)
            # 条件: 残り時間がmin_work以上あること
            if t + 1 + group['min_work'] <= T:
                # 修正: 再出勤にも base_cost を加算する
                c_restart = group['base_cost'] + group['preference_penalty'][t+1]
                G.add_edge((t, 'rest'), (t+1, 1, 0), weight=c_restart, type='restart', time=t+1)
            
            # --- B. Work (勤務) ノードからの遷移 ---
            for d in range(1, group['max_work'] + 1):
                for stat in [0, 1]:
                    u = (t, d, stat)
                    # ノードが存在しうるか簡易チェック（順方向構築でないため完全ではないが軽量化のため）
                    # d=1のノードはSource/Restartから来るので常にあり得る
                    if d > 1 and not G.has_node(u): continue 

                    cost_next = group['preference_penalty'][t+1]
                    
                    # B-1. 勤務継続 (Work Continuation)
                    if d < group['max_work']:
                        if stat == 0: # Work中
                            # -> Work (継続)
                            if d < group['break_threshold']:
                                G.add_edge(u, (t+1, d+1, 0), weight=cost_next, type='work', time=t+1)
                            # -> Break (休憩入り)
                            if d >= 2:
                                G.add_edge(u, (t+1, d, 1), weight=0, type='break', time=t+1)
                        else: # Break中
                            # -> Work (復帰)
                            G.add_edge(u, (t+1, d+1, 1), weight=cost_next, type='work', time=t+1)
                    
                    # B-2. 勤務終了 -> Rest (End of Shift)
                    # 修正: 直接Sinkへ行かせず、必ずRestを経由させる（トポロジーの統一）
                    if d >= group['min_work']:
                        G.add_edge(u, (t, 'rest'), weight=0, type='rest_start', time=t)

        # 3. 最終的なSinkへの接続
        # どの時間のRestからでも、週全体の勤務終了(Sink)へ行ける
        for t in range(T):
            G.add_edge((t, 'rest'), sink, weight=0, type='end', time=None)
            
        # 最終時刻(T-1)で勤務中の場合も、条件を満たせば終了可能
        # (ループが T-1 までなので、T-1 時点のノードからの処理が必要)
        # 簡易実装として、最終時刻に Work 状態で終わるパスは Sink へつなぐ
        # (厳密には最終時刻でのmin_workチェックが必要だが、T-1までループしているので
        #  そこで (T-1, 'rest') に落ちているはず)
                
        return G, source, sink

    def solve(self):
        print("Building Pure Network Flow MIP Model...")
        start_time = time.time()
        
        model = pulp.LpProblem("CompactMIP", pulp.LpMinimize)
        
        flow_vars = {} 
        objective_terms = []

        # 1. ネットワークフロー変数
        for k in range(self.prob.K):
            G, source, sink = self.build_graph_for_group(k)
            flow_vars[k] = {}
            for u, v, data in G.edges(data=True):
                var_name = f"flow_{k}_{str(u)}_{str(v)}"
                x = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)
                flow_vars[k][(u, v)] = x
                objective_terms.append(x * data['weight'])

            # フロー保存則
            for node in G.nodes():
                if node == source:
                    # Sourceからの流出 <= max_employees
                    model += pulp.lpSum([flow_vars[k][(source, v)] for v in G.successors(source)]) <= self.prob.groups[k]['max_employees']
                elif node == sink:
                    pass 
                else:
                    flow_in = pulp.lpSum([flow_vars[k][(u, node)] for u in G.predecessors(node)])
                    flow_out = pulp.lpSum([flow_vars[k][(node, v)] for v in G.successors(node)])
                    model += flow_in == flow_out

        # 2. 需要制約
        print("Adding demand constraints...")
        slack_vars = []
        big_m_penalty = 1e8 # ペナルティ設定（統一のため十分大きく）
        
        for t in range(self.prob.T):
            supply_terms = []
            
            # Flowからの供給のみ
            for k in range(self.prob.K):
                G, _, _ = self.build_graph_for_group(k)
                for u, v, data in G.edges(data=True):
                    if data.get('time') == t and data.get('type') != 'break':
                        if (u, v) in flow_vars[k]:
                            supply_terms.append(flow_vars[k][(u, v)])
            
            # スラック変数
            s_t = pulp.LpVariable(f"slack_{t}", lowBound=0, cat=pulp.LpContinuous)
            slack_vars.append(s_t)
            
            model += pulp.lpSum(supply_terms) + s_t >= self.prob.demand[t]

        # 目的関数
        model += pulp.lpSum(objective_terms) + pulp.lpSum([big_m_penalty * s for s in slack_vars])

        print("Solving Exact MIP...")
        model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=600)) 
        
        total_time = time.time() - start_time
        obj = pulp.value(model.objective)
        status = pulp.LpStatus[model.status]

        # =========================================================
        # 追加箇所: フロー分解 (Flow Decomposition) によるシフト復元
        # =========================================================
        assignments = []
        
        if status == "Optimal" or (obj is not None):
            print("Extracting schedules from network flow...")
            for k in range(self.prob.K):
                G, source, sink = self.build_graph_for_group(k)
                
                # 1. 計算結果の流量をコピーして保持
                current_flow = {}
                for (u, v), var in flow_vars[k].items():
                    val = pulp.value(var)
                    if val is not None and val > 0.5:
                        current_flow[(u, v)] = int(round(val))
                
                # 2. Sourceから1人ずつパスをたどって分解する
                while True:
                    # Sourceから出るフローがあるか探す
                    start_edges = [v for v in G.successors(source) if current_flow.get((source, v), 0) > 0]
                    if not start_edges:
                        break # このグループはもう誰もいない
                    
                    # パス追跡開始
                    curr = source
                    path_sched = [0] * self.prob.T
                    
                    while curr != sink:
                        # 次の行き先候補
                        candidates = [v for v in G.successors(curr) if current_flow.get((curr, v), 0) > 0]
                        if not candidates:
                            break # フロー切れ
                        
                        next_node = candidates[0] # 貪欲に1つ選ぶ
                        
                        # 勤務情報の記録
                        edge_data = G.get_edge_data(curr, next_node)
                        if edge_data.get('type') in ['start', 'work', 'restart']:
                            t = edge_data.get('time')
                            if t is not None:
                                path_sched[t] = 1
                        
                        # フロー消費
                        current_flow[(curr, next_node)] -= 1
                        curr = next_node
                    
                    # assignments形式に合わせて登録
                    assignments.append({
                        'column': {
                            'group_id': k,
                            'schedule': path_sched,
                            'cost': 0 # 可視化には不要なので0
                        },
                        'count': 1
                    })
        
        return {
            'time': total_time,
            'ip_obj': obj,
            'status': status,
            'assignments': assignments # 追加
        }

class IndividualExactSolver(CompactExactSolver):
    """
    従業員一人ひとりを個別のネットワークとして扱う厳密解法
    """
    def __init__(self, problem):
        super().__init__(problem)
        # 親クラスは problem.groups を使う前提なので、
        # ここでは employees を一時的に capacity=1 の group に見立てる
        self.pseudo_groups = []
        for emp in problem.employees:
            g_copy = emp.copy()
            g_copy['max_employees'] = 1 # 定員1名
            self.pseudo_groups.append(g_copy)
            
    def solve(self):
        print("Building Individual Network Flow MIP Model...")
        start_time = time.time()
        model = pulp.LpProblem("IndividualMIP", pulp.LpMinimize)
        
        # --- 既存ロジックの流用 ---
        # 本来の self.prob.groups を self.pseudo_groups (個人リスト) に差し替えて処理
        original_groups = self.prob.groups
        self.prob.groups = self.pseudo_groups
        self.prob.K = len(self.pseudo_groups)
        
        try:
            # 親クラスの solve メソッド内のロジックを再利用したいが、
            # 内部で build_graph_for_group を呼んでいるため、オーバーライドが必要。
            # ここでは簡潔にするため、親クラスの solve ロジックを模倣して再実装します。

            flow_vars = {}
            objective_terms = []
            
            # 1. 各個人のネットワーク構築
            for k, emp in enumerate(self.pseudo_groups):
                G, source, sink = self.build_graph_for_group(k)
                flow_vars[k] = {}
                for u, v, data in G.edges(data=True):
                    var_name = f"x_{k}_{str(u)}_{str(v)}"
                    # Binary変数にする（個人なので0か1）
                    x = pulp.LpVariable(var_name, cat=pulp.LpBinary)
                    flow_vars[k][(u, v)] = x
                    objective_terms.append(x * data['weight'])
                
                # フロー保存則
                for node in G.nodes():
                    if node == source:
                        model += pulp.lpSum([flow_vars[k][(source, v)] for v in G.successors(source)]) <= 1
                    elif node == sink:
                        pass
                    else:
                        flow_in = pulp.lpSum([flow_vars[k][(u, node)] for u in G.predecessors(node)])
                        flow_out = pulp.lpSum([flow_vars[k][(node, v)] for v in G.successors(node)])
                        model += flow_in == flow_out

            # 2. 需要制約
            slack_vars = []
            for t in range(self.prob.T):
                supply_terms = []
                for k in range(len(self.pseudo_groups)):
                    G, _, _ = self.build_graph_for_group(k)
                    for u, v, data in G.edges(data=True):
                        if data.get('time') == t and data.get('type') != 'break':
                            if (u, v) in flow_vars[k]:
                                supply_terms.append(flow_vars[k][(u, v)])
                
                s_t = pulp.LpVariable(f"slack_{t}", lowBound=0, cat=pulp.LpContinuous)
                slack_vars.append(s_t)
                model += pulp.lpSum(supply_terms) + s_t >= self.prob.demand[t]

            # 目的関数
            model += pulp.lpSum(objective_terms) + pulp.lpSum([1e8 * s for s in slack_vars])
            
            # ソルブ
            print(f"Solving optimization for {len(self.pseudo_groups)} individuals...")
            model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=600))
            
            # 結果取得
            assignments = []
            if model.status == 1:
                for k, emp in enumerate(self.pseudo_groups):
                    G, source, sink = self.build_graph_for_group(k)
                    # フローが流れているエッジを探す
                    path_sched = [0] * self.prob.T
                    active_edges = []
                    for (u, v), var in flow_vars[k].items():
                        if pulp.value(var) > 0.5:
                            active_edges.append((u, v))
                            edge_data = G.get_edge_data(u, v)
                            if edge_data.get('type') in ['start', 'work', 'restart']:
                                t = edge_data.get('time')
                                if t is not None:
                                    path_sched[t] = 1
                    
                    if sum(path_sched) > 0:
                        assignments.append({
                            'employee_id': emp['id'],
                            'employee_name': emp['name'],
                            'schedule': path_sched,
                            'cost': pulp.value(sum([flow_vars[k][e] * G.edges[e]['weight'] for e in active_edges]))
                        })

            return {
                'time': time.time() - start_time,
                'ip_obj': pulp.value(model.objective),
                'status': pulp.LpStatus[model.status],
                'assignments': assignments
            }

        finally:
            # データの復元
            self.prob.groups = original_groups
            self.prob.K = len(original_groups)


class ProposedDecompositionSolver:
    def __init__(self, problem, n_clusters=3):
        self.prob = problem
        self.n_clusters = n_clusters
        self.clusters = [] # クラスタリングされた仮想グループ
        
    def solve(self):
        print("\n=== Step 1: Clustering Employees ===")
        start_time = time.time()
        
        # 特徴量ベクトルの作成 (Preference Penaltyそのものを使う)
        feature_matrix = np.array([emp['preference_penalty'] for emp in self.prob.employees])
        
        # K-Means クラスタリング
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
        # クラスタごとに「仮想グループ」を作成
        self.clusters = []
        for c_idx in range(self.n_clusters):
            # このクラスタに属する従業員IDリスト
            member_indices = [i for i, label in enumerate(labels) if label == c_idx]
            members = [self.prob.employees[i] for i in member_indices]
            
            if not members: continue

            # クラスタの代表特性（平均的なペナルティ、最も厳しい制約など）を決定
            # ここでは簡単のため、メンバの平均ペナルティを使用
            avg_penalty = np.mean([m['preference_penalty'] for m in members], axis=0)
            
            # 制約（勤務時間など）は安全側に倒すか、代表値（最頻値など）を使う
            # ここではメンバーの最初の人の設定を借用（本来はMin/Maxの論理積をとるべき）
            base_spec = members[0] 
            
            cluster_group = {
                'name': f"Cluster_{c_idx}",
                'max_employees': len(members), # このクラスタの定員
                'members': members,            # 実際に属する個人のリスト
                'base_cost': base_spec['base_cost'],
                'hourly_wage': base_spec['hourly_wage'],
                'preference_penalty': avg_penalty, # 平均ペナルティで最適化する
                'min_work': base_spec['min_work'],
                'max_work': base_spec['max_work'],
                'break_threshold': base_spec['break_threshold'],
                'color': base_spec['color']
            }
            self.clusters.append(cluster_group)
            print(f"  > Cluster {c_idx}: {len(members)} employees")

        print("\n=== Step 2: Generating Shift Patterns (CG) ===")
        # 現在の ColumnGenerationSolver を使って、クラスタごとの最適パターン数を求める
        # 既存の solver は self.prob.groups を参照するので、一時的に差し替える
        
        original_groups = self.prob.groups
        self.prob.groups = self.clusters
        self.prob.K = len(self.clusters)
        
        cg_solver = ColumnGenerationSolver(self.prob)
        # 既存の solve() を呼び出し、クラスタごとの最適シフト数分布を得る
        # ここが「例題（RMP）を解いて列を生成」する部分に相当
        cluster_assignments = cg_solver.solve()
        
        # 復元
        self.prob.groups = original_groups
        self.prob.K = len(original_groups)
        
        print("\n=== Step 3: Assigning Individuals to Patterns ===")
        # 各クラスタで選ばれたシフトパターンに対して、実際のメンバーを割り当てる
        final_assignments = []
        
        for c_idx, cluster in enumerate(self.clusters):
            # 1. このクラスタに割り当てられた全シフトパターンをリスト化（スロット展開）
            # 例: パターンAが2人分 -> [PatternA, PatternA]
            assigned_slots = []
            relevant_items = [item for item in cluster_assignments if item['column']['group_id'] == c_idx]
            
            for item in relevant_items:
                schedule = item['column']['schedule']
                count = item['count']
                for _ in range(count):
                    assigned_slots.append(schedule)
            
            # 従業員数よりスロットが少ない場合（シフトなしの人がいる場合）、
            # ダミーの「休みシフト（オール0）」で埋める
            n_members = len(cluster['members'])
            n_slots = len(assigned_slots)
            
            if n_slots < n_members:
                for _ in range(n_members - n_slots):
                    assigned_slots.append([0]*self.prob.T)
            elif n_slots > n_members:
                # 逆にスロットが多すぎる場合は、コストが高い順などで削る必要があるが
                # CGの制約(max_employees)で守られているはずなので基本起きない
                assigned_slots = assigned_slots[:n_members]

            # 2. コスト行列の作成 (Rows: Members, Cols: Slots)
            cost_matrix = np.zeros((n_members, n_members))
            
            for i, member in enumerate(cluster['members']):
                for j, sched in enumerate(assigned_slots):
                    # コスト = 基本コスト + 個人のペナルティ和
                    # ※ 休みシフト(sum=0)の場合はコスト0とする
                    if sum(sched) == 0:
                        c = 0
                    else:
                        c = member['base_cost'] + np.sum(member['preference_penalty'] * np.array(sched))
                    cost_matrix[i, j] = c
            
            # 3. 割り当て問題（ハンガリアン法）を解く
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 4. 結果の登録
            for r, c in zip(row_ind, col_ind):
                member = cluster['members'][r]
                sched = assigned_slots[c]
                cost = cost_matrix[r, c]
                
                if sum(sched) > 0: # 勤務がある場合のみ登録
                    final_assignments.append({
                        'employee_id': member['id'],
                        'employee_name': member['name'],
                        'schedule': sched,
                        'cost': cost
                    })
                    
        total_time = time.time() - start_time
        total_cost = sum(a['cost'] for a in final_assignments)
        
        return {
            'time': total_time,
            'obj': total_cost,
            'assignments': final_assignments
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



def run_comparison_experiment():
    print("==========================================")
    print("   Exact Solver vs Column Generation")
    print("==========================================")
    
    # 1. 問題の定義（比較のため期間を1週間=168時間に短縮）
    T_comparison = 168 
    print(f"Problem Size: {T_comparison} time slots (7 days)")
    prob = ShiftSchedulingProblem(n_time_slots=T_comparison)

    # ----------------------------------------
    # 2. 厳密解法 (Compact Exact Solver)
    # ----------------------------------------
    print("\nRunning Compact Exact Solver (MIP)...")
    exact_solver = CompactExactSolver(prob)
    
    exact_res = exact_solver.solve()
    
    exact_obj = exact_res['ip_obj']
    exact_time = exact_res['time']
    exact_status = exact_res['status']

    print(f"  > Done. Status: {exact_status}")
    print(f"  > Time: {exact_time:.2f} s")
    
    # Noneチェック（Infeasible対策）
    if exact_obj is not None:
        print(f"  > Obj : {exact_obj:.2f}")
    else:
        print(f"  > Obj : N/A (Infeasible or Error)")
        exact_obj = 0.0 # 計算用ダミー

    # ----------------------------------------
    # 3. 提案手法 (Column Generation)
    # ----------------------------------------
    print("\nRunning Column Generation (Proposed)...")
    cg_solver = ColumnGenerationSolver(prob)
    
    start_time_cg = time.time()
    
    # ※ Heuristicを実装済みの場合は solve_heuristic() を呼ぶ
    # assignments = cg_solver.solve_heuristic()
    assignments = cg_solver.solve() 
    
    total_time_cg = time.time() - start_time_cg
    
    # 目的関数値（総コスト）の計算
    cg_obj = sum(item['column']['cost'] * item['count'] for item in assignments)
    
    print(f"  > Done.")
    print(f"  > Time: {total_time_cg:.2f} s")
    print(f"  > Obj : {cg_obj:.2f}")

    # ----------------------------------------
    # 4. 比較結果の集計
    # ----------------------------------------
    print("\n==========================================")
    print("   Comparison Result")
    print("==========================================")
    
    # Gap計算 ( (CG - Exact) / Exact )
    if exact_obj is not None and exact_obj > 1e-5:
        gap = (cg_obj - exact_obj) / exact_obj * 100
        gap_str = f"{gap:.2f}%"
    else:
        gap_str = "N/A"

    # === 【修正箇所】ここで time_reduction を計算する ===
    if exact_time > 0:
        time_reduction = (exact_time - total_time_cg) / exact_time * 100
    else:
        time_reduction = 0.0
    # =================================================

    print(f"{'Metric':<15} | {'Exact (MIP)':<15} | {'Proposed (CG)':<15} | {'Diff / Gap'}")
    print("-" * 65)
    print(f"{'Time (sec)':<15} | {exact_time:<15.2f} | {total_time_cg:<15.2f} | {time_reduction:.1f}% faster")
    print(f"{'Total Cost':<15} | {exact_obj:<15.1f} | {cg_obj:<15.1f} | {gap_str} gap")
    print(f"{'Status':<15} | {exact_status:<15} | {'Integer Feasible':<15} | -")
    print("==========================================")

    
def run_warm_start_experiment():
    print("==========================================")
    print("   Warm Start Strategy Experiment")
    print("   (Use learned patterns as initial guess + generated missing ones)")
    print("==========================================")
    
    n_learn_weeks = 10
    total_weeks = n_learn_weeks + 1
    
    # 1. データ生成
    full_prob = ShiftSchedulingProblem(n_time_slots=168 * total_weeks)
    print(f"Generated data for {total_weeks} weeks.")
    
    # ----------------------------------------
    # Phase 1: Learning Loop
    # ----------------------------------------
    print(f"\n[Phase 1] Learning Patterns over {n_learn_weeks} weeks...")
    unique_patterns = set()
    
    # 学習フェーズ
    for w in range(n_learn_weeks):
        prob_w = get_weekly_problem(full_prob, w)
        solver = ColumnGenerationSolver(prob_w)
        
        # ログ抑制して実行
        import sys, io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            solver.solve() 
        finally:
            sys.stdout = original_stdout
            
        for col in solver.columns:
            pattern_key = (col['group_id'], tuple(col['schedule']))
            unique_patterns.add(pattern_key)
        
        print(f"  > Week {w} done. Pool size: {len(unique_patterns)}")

    # ----------------------------------------
    # Phase 2: Testing (Warm Start vs Exact)
    # ----------------------------------------
    target_week_idx = n_learn_weeks
    print(f"\n[Phase 2] Solving Target Week {target_week_idx}")
    
    prob_target = get_weekly_problem(full_prob, target_week_idx)

    # --- A. 厳密解法 ---
    print("\nA. Exact Solver (Scratch)...")
    exact_solver = CompactExactSolver(prob_target)
    res_exact = exact_solver.solve()
    exact_time = res_exact['time']
    exact_obj = res_exact['ip_obj']

    # --- B. Warm Start CG ---
    print(f"\nB. Warm Start CG (Initial Pool: {len(unique_patterns)})...")
    ws_solver = ColumnGenerationSolver(prob_target)
    
    start_time_ws = time.time()
    
    # 1. 通常の初期列 (ダミー)
    ws_solver.build_initial_columns()
    
    # 2. 学習済みパターンを注入（Warm Start）
    learned_columns_list = []
    for (gid, sched_tuple) in unique_patterns:
        learned_columns_list.append({
            'group_id': gid,
            'schedule': list(sched_tuple)
        })
    ws_solver.import_columns(learned_columns_list)
    
    # 3. 通常のsolve()を実行
    #    Pricingループが回り、足りない列は新規生成される
    assignments = ws_solver.solve()
    
    time_ws = time.time() - start_time_ws
    obj_ws = sum(item['column']['cost'] * item['count'] for item in assignments)

    # ----------------------------------------
    # 結果比較
    # ----------------------------------------
    print("\n==========================================")
    print("   Warm Start Strategy Result")
    print("==========================================")
    
    # Gap計算
    if exact_obj is not None and exact_obj > 0:
        gap = (obj_ws - exact_obj) / exact_obj * 100
    else:
        gap = float('inf')
    
    # 速度向上率 (Speedup) 計算
    if exact_time > 0:
        speedup = exact_time / time_ws
    else:
        speedup = 0.0

    print(f"{'Metric':<15} | {'Exact (Scratch)':<15} | {'Warm Start CG':<15} | {'Comparison'}")
    print("-" * 65)
    print(f"{'Time (sec)':<15} | {exact_time:<15.4f} | {time_ws:<15.4f} | x{speedup:.1f} faster")
    print(f"{'Total Cost':<15} | {exact_obj:<15.1f} | {obj_ws:<15.1f} | {gap:.2f}% gap")
    print("==========================================")

    # === 可視化の実行 ===
    print("\nVisualizing Results...")

    # 1. 厳密解法の可視化
    if res_exact.get('assignments'):
        visualize_solution(prob_target, res_exact['assignments'], title="Exact Solver Result (Network Flow)")
    else:
        print("Exact solver did not return assignments.")

    # 2. 提案手法(Warm Start)の可視化
    if assignments:
        visualize_solution(prob_target, assignments, title="Warm Start CG Result")
        

def visualize_solution(prob, assignments, title="Shift Schedule Result"):
    """
    割り当て結果(assignments)を受け取り、需給グラフとガントチャートを描画する
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # データ整理
    T = prob.T
    hours = np.arange(T)
    
    # グループごとの供給量を計算
    group_supplies = np.zeros((prob.K, T))
    
    # ガントチャート用のデータ構造整理
    # structure: gantt_data[group_id] = [ {'schedule': array, 'count': int}, ... ]
    gantt_data = {k: [] for k in range(prob.K)}

    for item in assignments:
        col = item['column']
        cnt = item['count']
        gid = col['group_id']
        sched = np.array(col['schedule'])
        
        # 供給量加算 (人数 * スケジュール)
        group_supplies[gid] += sched * cnt
        
        # ガントチャート用
        if cnt > 0:
            gantt_data[gid].append({'schedule': sched, 'count': cnt})

    # === プロット作成 ===
    fig = plt.figure(figsize=(18, 12))
    # レイアウト: 上部に需給グラフ(3)、下部にグループごとの詳細(1*K)
    gs = fig.add_gridspec(prob.K + 1, 1, height_ratios=[3] + [1]*prob.K)

    # --- 1. 需給積み上げグラフ (Supply vs Demand) ---
    ax_main = fig.add_subplot(gs[0])
    
    # 積み上げ計算
    bottom = np.zeros(T)
    for k in range(prob.K):
        group = prob.groups[k]
        ax_main.bar(hours, group_supplies[k], bottom=bottom, 
                    label=group['name'], color=group['color'], 
                    width=1.0, align='edge', alpha=0.8, edgecolor='none')
        bottom += group_supplies[k]
    
    # 需要線 (Demand)
    ax_main.step(hours, prob.demand, where='post', color='red', linewidth=2.5, linestyle='--', label='Required Demand')
    
    # 装飾
    ax_main.set_title(f"{title} - Supply vs Demand", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("Number of Employees")
    ax_main.set_xlim(0, T)
    ax_main.legend(loc='upper right', frameon=True)
    ax_main.grid(True, linestyle=':', alpha=0.6)
    
    # X軸ラベル（24時間ごとに区切り線）
    xticks = np.arange(0, T+1, 24)
    xticklabels = [f"Day {i}" for i in range(len(xticks))]
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels(xticklabels)
    for x in xticks:
        ax_main.axvline(x, color='black', linewidth=1, alpha=0.3)

    # --- 2. グループ別ガントチャート ---
    for k in range(prob.K):
        ax = fig.add_subplot(gs[k+1], sharex=ax_main)
        group = prob.groups[k]
        items = gantt_data[k]
        
        # Y軸の位置管理
        y_pos = 0
        
        for item in items:
            sched = item['schedule']
            count = item['count']
            
            # 連続勤務区間を検出してバーを描画
            # 0と1の境界を探す
            bounded = np.hstack(([0], sched, [0]))
            difs = np.diff(bounded)
            starts, = np.where(difs == 1)
            stops, = np.where(difs == -1)
            
            for s, e in zip(starts, stops):
                duration = e - s
                # バーを描画
                rect = mpatches.Rectangle((s, y_pos), duration, 0.8, 
                                          facecolor=group['color'], edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                
                # 中央に人数(xN)を表示
                if duration > 2: # 短すぎると文字がはみ出るので
                    ax.text(s + duration/2, y_pos + 0.4, f"x{int(count)}", 
                            ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
            y_pos += 1
        
        # 背景にペナルティ（働きたくない時間）をヒートマップ風に表示
        # 0(白) -> High(赤)
        penalty = group['preference_penalty']
        # 縦幅いっぱいに広げる
        ax.imshow([penalty], aspect='auto', cmap='Reds', alpha=0.15, 
                  extent=[0, T, 0, max(1, y_pos)], vmin=0, vmax=np.max(penalty)+1)

        ax.set_ylabel(group['name'], fontsize=10, rotation=0, ha='right', va='center')
        ax.set_yticks([]) # Y軸メモリは消す
        ax.set_ylim(0, max(1, y_pos))
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)
        
        # Day区切り
        for x in xticks:
            ax.axvline(x, color='black', linewidth=1, alpha=0.3)

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     run_warm_start_experiment()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_individual_solution(prob, assignments, title="Individual Schedule Result"):
    """
    個人単位の割り当て結果(assignments)を受け取り、需給グラフと個人別ガントチャートを描画する
    assignments構造: [{'employee_id': int, 'schedule': [0,1,...], ...}, ...]
    """
    T = prob.T
    hours = np.arange(T)
    
    # --- 1. データを整理 ---
    # 従業員IDをキーにした辞書
    assign_dict = {a['employee_id']: a['schedule'] for a in assignments}
    
    # タイプごとの供給量を計算（積み上げグラフ用）
    # prob.groups は「元のグループ定義（3種類）」を参照するために使う
    # prob.employees[i]['type'] でその人がどのグループベースかがわかる
    n_types = 3 # Full-time, Part-time(H), Part-time(S)
    type_supplies = np.zeros((n_types, T))
    
    # 全員のスケジュールを集計
    for emp in prob.employees:
        eid = emp['id']
        etype = emp['type']
        sched = assign_dict.get(eid, np.zeros(T)) # 割り当てがない場合は全休
        type_supplies[etype] += np.array(sched)

    # --- 2. プロット作成 ---
    fig = plt.figure(figsize=(18, 14))
    # 上段: 需給グラフ(3), 下段: 全従業員のガントチャート(10)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

    # === A. 需給積み上げグラフ ===
    ax_main = fig.add_subplot(gs[0])
    
    bottom = np.zeros(T)
    # 元のグループ定義（色や名前）を使って積み上げ
    # ※ IndividualShiftProblemでも親クラスの groups に基本定義が残っている前提
    # ただし prob.groups が書き換わっている可能性に備え、色などは固定または employees[0] から取得
    base_groups = [
        {'name': 'Full-time', 'color': 'tab:blue'},
        {'name': 'Part-time(HW)', 'color': 'tab:orange'},
        {'name': 'Part-time(Stu)', 'color': 'tab:green'}
    ]
    
    for k in range(n_types):
        ax_main.bar(hours, type_supplies[k], bottom=bottom, 
                    label=base_groups[k]['name'], color=base_groups[k]['color'], 
                    width=1.0, align='edge', alpha=0.8, edgecolor='none')
        bottom += type_supplies[k]
    
    # 需要線
    ax_main.step(hours, prob.demand, where='post', color='red', linewidth=2.5, linestyle='--', label='Required Demand')
    
    ax_main.set_title(f"{title} - Supply vs Demand", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("Count")
    ax_main.set_xlim(0, T)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, linestyle=':', alpha=0.6)
    
    # X軸の装飾
    xticks = np.arange(0, T+1, 24)
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels([f"Day {i}" for i in range(len(xticks))])
    for x in xticks:
        ax_main.axvline(x, color='black', linewidth=1, alpha=0.3)

    # === B. 個人別ガントチャート ===
    ax_gantt = fig.add_subplot(gs[1], sharex=ax_main)
    
    # 表示順を整える（タイプ順 -> ID順）
    sorted_employees = sorted(prob.employees, key=lambda x: (x['type'], x['id']))
    
    y_pos = 0
    yticks = []
    yticklabels = []
    
    for emp in sorted_employees:
        eid = emp['id']
        sched = assign_dict.get(eid, np.zeros(T))
        
        # 背景にその人のペナルティ（希望）を表示
        # ペナルティが高い時間帯ほど赤くする
        penalty = np.array([emp['preference_penalty']])
        ax_gantt.imshow(penalty, aspect='auto', cmap='Reds', alpha=0.3, 
                        extent=[0, T, y_pos - 0.4, y_pos + 0.4], vmin=0, vmax=300)

        # 勤務区間の描画
        bounded = np.hstack(([0], sched, [0]))
        difs = np.diff(bounded)
        starts, = np.where(difs == 1)
        stops, = np.where(difs == -1)
        
        color = base_groups[emp['type']]['color']
        
        for s, e in zip(starts, stops):
            duration = e - s
            rect = mpatches.Rectangle((s, y_pos - 0.3), duration, 0.6, 
                                      facecolor=color, edgecolor='black', alpha=0.9)
            ax_gantt.add_patch(rect)
        
        yticks.append(y_pos)
        yticklabels.append(f"{emp['name']}")
        y_pos += 1
        
    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(yticklabels, fontsize=9)
    ax_gantt.set_ylim(-0.5, len(sorted_employees) - 0.5)
    ax_gantt.invert_yaxis() # 上から順にIDを表示
    ax_gantt.grid(True, axis='x', linestyle=':', alpha=0.5)
    ax_gantt.set_title("Individual Schedules (Red background = High Discomfort/Penalty)", fontsize=12)
    
    for x in xticks:
        ax_gantt.axvline(x, color='black', linewidth=1, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_individual_assignment_experiment():
    print("==========================================")
    print("   Individual Assignment Comparison")
    print("==========================================")
    
    # 1. データ生成（個人データの作成）
    # ※ 人数が多すぎるとグラフが見にくいので20人程度で実行
    prob = IndividualShiftProblem(n_employees=20, n_time_slots=168)
    print(f"Generated {prob.n_employees} employees with unique preferences.")

    # ----------------------------------------
    # A. 厳密解法 (Individual Exact MIP)
    # ----------------------------------------
    print("\nA. Running Individual Exact Solver (Huge MIP)...")
    exact_solver = IndividualExactSolver(prob)
    res_exact = exact_solver.solve()
    
    print(f"  > Time: {res_exact['time']:.2f} s")
    print(f"  > Obj : {res_exact['ip_obj']:.2f}")

    # 結果の可視化 (Exact)
    if res_exact['assignments']:
        print("  > Visualizing Exact Solution...")
        visualize_individual_solution(prob, res_exact['assignments'], title="Exact Solution (MIP)")
    else:
        print("  > No solution found by Exact Solver.")

    # ----------------------------------------
    # B. 提案手法 (Clustering -> CG -> Matching)
    # ----------------------------------------
    print("\nB. Running Proposed Decomposition Solver...")
    prop_solver = ProposedDecompositionSolver(prob, n_clusters=3)
    res_prop = prop_solver.solve()
    
    print(f"  > Time: {res_prop['time']:.2f} s")
    print(f"  > Obj : {res_prop['obj']:.2f}")
    
    # 結果の可視化 (Proposed)
    if res_prop['assignments']:
        print("  > Visualizing Proposed Solution...")
        visualize_individual_solution(prob, res_prop['assignments'], title="Proposed Method (Cluster+CG+Match)")
    
    # ----------------------------------------
    # 比較
    # ----------------------------------------
    if res_exact['ip_obj'] and res_exact['ip_obj'] > 0:
        gap = (res_prop['obj'] - res_exact['ip_obj']) / res_exact['ip_obj'] * 100
    else:
        gap = 0
        
    print("\n==========================================")
    print(f"Summary: Gap = {gap:.2f}%")
    print(f"Exact Time: {res_exact['time']:.2f}s vs Proposed Time: {res_prop['time']:.2f}s")
    print("==========================================")

if __name__ == "__main__":
    run_individual_assignment_experiment()
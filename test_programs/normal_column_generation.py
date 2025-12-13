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
    def __init__(self, n_time_slots=168):
        self.T = n_time_slots
        self.hours = np.arange(self.T)
        
        np.random.seed(42)
        
        # 需要: 昼と夕方にピークがある一般的な店舗を想定
        day_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
        weekly_base = np.tile(day_pattern, 7) 
        # 土日は需要増
        weekly_base[24*5:] = weekly_base[24*5:] * 1.2 
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_base + noise, 1, None).astype(int)
        
        self.groups = []
        
        # --- グループ定義 ---
        
        # Group 0: 正社員 (Full-time)
        # コスト高, 8時間労働基本, 日中メイン
        g0_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 8 or h > 19: g0_penalty[t] = 200 # 早朝深夜は嫌
        
        self.groups.append({
            'name': 'Full-time (Day)',
            'max_employees': 5,      # 人数少なめ
            'base_cost': 5000,       # 固定費高い
            'hourly_wage': 2000,     # 時給換算(ペナルティ計算用基準)
            'preference_penalty': g0_penalty,
            'min_work': 8,
            'max_work': 9,
            'break_threshold': 6,
            'color': 'tab:blue'
        })

        # Group 1: パート・主婦層 (Part-time Day)
        # コスト安, 短時間, 夕方以降NG
        g1_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 9: g1_penalty[t] = 100
            if h > 16: g1_penalty[t] = 1000 # 夕食準備のため絶対帰りたい
        
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

        # Group 2: 学生バイト (Part-time Night)
        # 修正方針:
        # - 平日: 授業中(9-17)はNG。朝は苦手。
        # - 週末: 「早起きは絶対したくない(昼まで寝る)」「でも深夜まで働くのはOK」という学生らしさを追加
        g2_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            day = t // 24
            
            # --- 平日 (月〜金) ---
            if day < 5: 
                if 9 <= h <= 17: g2_penalty[t] = 500 # 授業中
                if h < 10: g2_penalty[t] += 50       # 朝はちょっと苦手
            
            # --- 週末 (土・日) ---
            else:
                # 週末はもっと寝ていたい (12時前は働きたくない)
                if h < 12: g2_penalty[t] += 300
                
                # (オプション) 日曜の深夜(22時以降)は翌日の学校に響くので嫌がる傾向を入れるなら
                if day == 6 and h >= 22: g2_penalty[t] += 100

        self.groups.append({
            'name': 'Part-time (Student)',
            'max_employees': 20,
            'base_cost': 1000,
            'hourly_wage': 1100,
            'preference_penalty': g2_penalty,
            'min_work': 4,  # 最低勤務時間を少し長めにするのも「こま切れ」防止に有効です
            'max_work': 8,
            'break_threshold': 6,
            'color': 'tab:green'
        })
        
        self.K = len(self.groups)
        self.big_m = 100000

# ==========================================
# 2. Proposed Solver (CG + Multi-day Graph)
# ==========================================
class ColumnGenerationSolver:
    def __init__(self, problem):
        self.prob = problem
        self.columns = []
        self.pool = [] # 今回は columns とほぼ同期させますが、概念として分けます
        self.duals = {'demand': [], 'convex': []}

    def build_initial_columns(self):
        # 初期解: 各グループごとの標準的なシフトを入れる
        # G0: 9-18, G1: 10-15, G2: 18-23
        standard_times = [(9, 18), (10, 15), (18, 23)]
        
        for k in range(self.prob.K):
            group = self.prob.groups[k]
            s_time, e_time = standard_times[k]
            
            for day in range(7):
                start = day*24 + s_time
                end = day*24 + e_time
                duration = end - start
                
                if duration < group['min_work'] or duration > group['max_work']:
                    continue # 設定ミス回避
                
                sched = [0]*self.prob.T
                # コスト = 固定費(1回) + ペナルティ総和
                cost = group['base_cost']
                for t in range(start, end):
                    if t < self.prob.T:
                        sched[t] = 1
                        cost += group['preference_penalty'][t]
                
                self.add_column(k, sched, cost)
        
        # ダミー列 (実行不能回避)
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
        
        active_cols = [c for c in self.columns if c['in_rmp']]
        
        x_vars = {c['id']: pulp.LpVariable(f"x_{c['id']}", lowBound=0, cat=cat) for c in active_cols}
        s_vars = [pulp.LpVariable(f"s_{t}", lowBound=0, cat=cat) for t in range(self.prob.T)]
        
        # (目的関数・制約式の構築部分は変更なし...)
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
            
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # === 追加箇所: 解の値を列情報に保存 ===
        # これがないと、どの列を消していいかわからないため
        if not integer:
            for c in active_cols:
                val = pulp.value(x_vars[c['id']])
                c['val'] = val if val is not None else 0.0
        # ====================================

        if integer:
            assignments = []
            for c in active_cols:
                val = pulp.value(x_vars[c['id']])
                if val is not None and val > 0.1:
                    assignments.append({'column': c, 'count': int(round(val))})
            return assignments
        else:
            self.duals['demand'] = [c.pi for c in d_cons]
            self.duals['convex'] = [c.pi for c in g_cons]
            return pulp.value(model.objective)
        
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

    def solve(self):
        self.build_initial_columns()
        
        max_iter = 400
        # ヘッダーに Active (RMP内の列数) を追加
        print(f"{'Iter':<5} | {'Obj':<10} | {'Active':<8} | {'Total':<8} | {'Pool Check'}")
        print("-" * 65)
        
        for i in range(max_iter):
            obj = self.solve_rmp(integer=False)
            
            # === 追加箇所: クリーンアップ処理 ===
            # 毎回やると、プールからの復活(pricing内)と打ち消し合って振動する可能性があるので
            # 数回に1回、あるいは列数が増えてきたら実行するのが定石
            cleaned = 0
            if i > 0 and i % 5 == 0: # 5回に1回お掃除
                cleaned = self.cleanup_rmp()
            # ==================================

            new_cols = self.pricing()
            
            # RMPに入っている列の数
            active_cols = sum(1 for c in self.columns if c['in_rmp'])
            
            # ログ出力修正
            note = f"Cleaned {cleaned}" if cleaned > 0 else ""
            print(f"{i:<5} | {obj:<10.2f} | {active_cols:<8} | {len(self.columns):<8} | {note}")
            
            if new_cols == 0:
                print("\nOptimal (LP Relaxation) Reached.")
                break
        
        print("\nSolving Integer Problem...")
        return self.solve_rmp(integer=True)
    

    
class CompactExactSolver:
    """
    厳密解法ソルバー (Compact MIP / Network Flow Formulation)
    
    列生成を行わず、Pricingで用いるグラフ構造（ノードとエッジ）を
    そのまま巨大な混合整数計画問題(MIP)として定式化して解く。
    理論上の「最適解」を保証するが、計算時間がかかりメモリを消費する。
    """
    def __init__(self, problem):
        self.prob = problem

    def build_graph_for_group(self, k):
        """Pricingと同じロジックでグラフ構造（エッジリスト）のみを作成"""
        group = self.prob.groups[k]
        G = nx.DiGraph()
        source, sink = 'S', 'E'
        
        # 1. Source -> Start
        for t in range(self.prob.T):
            if t + group['min_work'] > self.prob.T: continue
            # コスト: 固定費 + ペナルティ (双対変数は引かない！)
            c = group['base_cost'] + group['preference_penalty'][t]
            G.add_edge(source, (t, 1, 0), weight=c, type='start', time=t)

        # 2. Transitions
        for t in range(self.prob.T - 1):
            for d in range(1, group['max_work'] + 1):
                for stat in [0, 1]:
                    u = (t, d, stat)
                    if not G.has_node(u) and d > 1: continue

                    cost_next = group['preference_penalty'][t+1]
                    
                    # (A) Work Continuation
                    if d < group['max_work']:
                        # Unrested
                        if stat == 0:
                            if d < group['break_threshold']:
                                G.add_edge(u, (t+1, d+1, 0), weight=cost_next, type='work', time=t+1)
                            if d >= 2:
                                # Break (0 cost, no work)
                                G.add_edge(u, (t+1, d, 1), weight=0, type='break', time=t+1)
                        # Rested
                        else:
                            G.add_edge(u, (t+1, d+1, 1), weight=cost_next, type='work', time=t+1)
                    
                    # (B) Work -> Rest/End
                    if d >= group['min_work']:
                        G.add_edge(u, (t, 'rest'), weight=0, type='rest_start', time=t)
                        G.add_edge(u, sink, weight=0, type='end', time=None)

            # (C) Rest Transitions
            if t > 0:
                G.add_edge((t, 'rest'), (t+1, 'rest'), weight=0, type='rest_cont', time=t+1)
                c_restart = group['preference_penalty'][t+1]
                G.add_edge((t, 'rest'), (t+1, 1, 0), weight=c_restart, type='restart', time=t+1)
        
        # End processing
        for t in range(self.prob.T):
            if G.has_node((t, 'rest')):
                G.add_edge((t, 'rest'), sink, weight=0, type='end', time=None)
                
        return G, source, sink

    def solve(self):
        print("Building Compact MIP Model (This may take time)...")
        start_time = time.time()
        
        model = pulp.LpProblem("CompactMIP", pulp.LpMinimize)
        
        # 変数管理用
        # flow_vars[k][u][v] = グループkがエッジ(u,v)を流れる人数
        flow_vars = {} 
        
        # グループごとにグラフを構築し、フロー変数を定義
        for k in range(self.prob.K):
            G, source, sink = self.build_graph_for_group(k)
            flow_vars[k] = {}
            
            for u, v, data in G.edges(data=True):
                # エッジごとの流量変数（整数）
                var_name = f"f_{k}_{str(u)}_{str(v)}"
                x = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)
                flow_vars[k][(u, v)] = x
                
                # 目的関数にコストを追加 (cost * x)
                model += x * data['weight']

            # フロー保存則 (Flow Conservation)
            # Source, Sink以外の全ノードで 入ってくる量 = 出ていく量
            for node in G.nodes():
                if node == source:
                    # Sourceから出る量 <= max_employees
                    model += pulp.lpSum([flow_vars[k][(source, v)] for v in G.successors(source)]) <= self.prob.groups[k]['max_employees']
                elif node == sink:
                    pass # Sinkは吸収のみなので制約なし（Sourceの制約で決まる）
                else:
                    flow_in = pulp.lpSum([flow_vars[k][(u, node)] for u in G.predecessors(node)])
                    flow_out = pulp.lpSum([flow_vars[k][(node, v)] for v in G.successors(node)])
                    model += flow_in == flow_out

        # 需要制約 (Demand Constraint)
        # 全グループの「働いているエッジ」の合計 >= 需要
        print("Adding demand constraints...")
        for t in range(self.prob.T):
            supply_at_t = []
            for k in range(self.prob.K):
                # この時刻tに「働いている(type=work/start/restart)」エッジを探して足す
                # グラフ再構築は重いので、変数生成時にメタデータを保持しておくのが効率的だが
                # ここでは簡易的にグラフを再取得する（厳密には非効率だが実装優先）
                G, _, _ = self.build_graph_for_group(k)
                for u, v, data in G.edges(data=True):
                    if data.get('time') == t and data.get('type') in ['start', 'work', 'restart']:
                        if (u, v) in flow_vars[k]:
                            supply_at_t.append(flow_vars[k][(u, v)])
            
            if supply_at_t:
                model += pulp.lpSum(supply_at_t) >= self.prob.demand[t]

        print("Solving Compact MIP...")
        # タイムリミットを設定しないと終わらない可能性があるため、適度に設定してもよい
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

if __name__ == "__main__":
    run_visualize_experiment()
import pulp
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

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
        # 変更なし（前回のコードと同じグラフ探索ロジックを使用）
        new_cnt = 0
        if not self.duals['demand']: return 0
        
        for k in range(self.prob.K):
            group = self.prob.groups[k]
            pi = self.duals['demand']
            mu = self.duals['convex'][k]
            G = nx.DiGraph()
            source, sink = 'S', 'E'
            for t in range(self.prob.T):
                if t + group['min_work'] > self.prob.T: continue
                c = (group['base_cost'] + group['preference_penalty'][t]) - pi[t]
                G.add_edge(source, (t, 1, 0), weight=c)
            for t in range(self.prob.T - 1):
                for d in range(1, group['max_work'] + 1):
                    for stat in [0, 1]:
                        curr_node = (t, d, stat)
                        if not G.has_node(curr_node) and d > 1: continue 
                        cost_next = group['preference_penalty'][t+1] - pi[t+1]
                        if d < group['max_work']:
                            if stat == 0:
                                if d < group['break_threshold']:
                                    G.add_edge(curr_node, (t+1, d+1, 0), weight=cost_next)
                                if d >= 2: 
                                    G.add_edge(curr_node, (t+1, d, 1), weight=0)
                            else:
                                G.add_edge(curr_node, (t+1, d+1, 1), weight=cost_next)
                        if d >= group['min_work']:
                            G.add_edge(curr_node, (t, 'rest'), weight=0)
                            G.add_edge(curr_node, sink, weight=0)
                if t > 0: 
                    G.add_edge((t, 'rest'), (t+1, 'rest'), weight=0)
                    c_restart = group['preference_penalty'][t+1] - pi[t+1]
                    G.add_edge((t, 'rest'), (t+1, 1, 0), weight=c_restart)
            for t in range(self.prob.T):
                if G.has_node((t, 'rest')):
                    G.add_edge((t, 'rest'), sink, weight=0)
            try:
                path = nx.bellman_ford_path(G, source, sink, weight='weight')
                rc = nx.bellman_ford_path_length(G, source, sink, weight='weight')
                if rc - mu < -1e-5:
                    sched = [0]*self.prob.T
                    real_cost = group['base_cost']
                    for i in range(1, len(path)-1):
                        u, v = path[i-1], path[i]
                        if isinstance(v, tuple) and len(v) == 3:
                            t = v[0]
                            if isinstance(u, tuple) and len(u)==3 and u[1] == v[1]: pass 
                            else:
                                sched[t] = 1
                                real_cost += group['preference_penalty'][t]
                    self.add_column(k, sched, real_cost)
                    new_cnt += 1
            except: pass
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
        # グラフ構築ロジック（変更なし）
        group = self.prob.groups[k]
        G = nx.DiGraph()
        source, sink = 'S', 'E'
        
        # 1. Source -> Start
        for t in range(self.prob.T):
            if t + group['min_work'] > self.prob.T: continue
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
                        if stat == 0:
                            if d < group['break_threshold']:
                                G.add_edge(u, (t+1, d+1, 0), weight=cost_next, type='work', time=t+1)
                            if d >= 2:
                                # Break
                                G.add_edge(u, (t+1, d, 1), weight=0, type='break', time=t+1)
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
        
        for t in range(self.prob.T):
            if G.has_node((t, 'rest')):
                G.add_edge((t, 'rest'), sink, weight=0, type='end', time=None)
                
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

if __name__ == "__main__":
    run_warm_start_experiment()

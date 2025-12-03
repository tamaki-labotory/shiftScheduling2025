import pulp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ==========================================
# 1. 問題設定 (1ヶ月=4週間)
# ==========================================
class ShiftSchedulingProblem:
    def __init__(self, n_weeks=4):
        # 期間設定: 4週間 = 672時間
        self.n_weeks = n_weeks
        self.T = 168 * n_weeks
        self.hours = np.arange(self.T)
        
        np.random.seed(42)
        
        # --- 需要データの生成 ---
        base_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
        weekly_demand = np.tile(base_pattern, 7 * n_weeks)
        
        # 週末需要増 & ノイズ
        for t in range(self.T):
            day = (t // 24) % 7
            hour = t % 24
            if day == 4 and hour >= 18: weekly_demand[t] *= 1.3
            elif day >= 5: weekly_demand[t] *= 1.2
            
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_demand + noise, 0, None).astype(int)
        
        # --- グループ定義 ---
        self.groups = []
        
        # Group 0: Full-time (週5日)
        g0_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 7 or h > 20: g0_penalty[t] = 100
            
        self.groups.append({
            'id': 0, 'name': 'Full-time', 'max_employees': 8,
            'base_cost': 1500, 'hourly_cost': 2000, 'preference_penalty': g0_penalty,
            'min_work': 8, 'max_work': 10, 'rest_interval': 8, 'color': 'tab:blue',
            'max_days_per_week': 5 
        })

        # Group 1: Part-time (週5日)
        g1_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 9: g1_penalty[t] = 50
            if h > 17: g1_penalty[t] = 5000
            
        self.groups.append({
            'id': 1, 'name': 'Part-time (Day)', 'max_employees': 10,
            'base_cost': 1000, 'hourly_cost': 1400, 'preference_penalty': g1_penalty,
            'min_work': 4, 'max_work': 6, 'rest_interval': 8, 'color': 'tab:orange',
            'max_days_per_week': 5
        })

        # Group 2: Student (週4日)
        g2_penalty = np.zeros(self.T)
        for t in range(self.T):
            d = (t // 24) % 7
            h = t % 24
            if d < 5 and 9 <= h <= 17: g2_penalty[t] = 5000
            
        self.groups.append({
            'id': 2, 'name': 'Student (Night/Wknd)', 'max_employees': 10,
            'base_cost': 1000, 'hourly_cost': 1100, 'preference_penalty': g2_penalty,
            'min_work': 3, 'max_work': 8, 'rest_interval': 8, 'color': 'tab:green',
            'max_days_per_week': 4
        })
        
        self.K = len(self.groups)
        self.big_m = 5000

# ==========================================
# 2. 厳密ソルバー (週次リセット対応)
# ==========================================
class ExactTimeSpaceMIPSolver:
    def __init__(self, problem):
        self.prob = problem

    def build_network(self, group_id):
        """
        Nodes: (t, duration, state, k)
        k: *その週における* 累積勤務回数 (0 ~ max_days)
        """
        group = self.prob.groups[group_id]
        G = nx.DiGraph()
        source, sink = 'Source', 'Sink'
        T = self.prob.T
        min_w, max_w = group['min_work'], group['max_work']
        min_rest = group['rest_interval']
        max_days = group.get('max_days_per_week', 5)
        
        # --- 1. 初期出勤 ---
        for t in range(T - min_w + 1):
            cost = group['base_cost'] + group['hourly_cost'] + group['preference_penalty'][t]
            # 第1週の1回目なので k=1
            G.add_edge(source, (t, 1, 0, 1), weight=cost)

        # --- 2. 時間進行 ---
        # k は 0(未勤務) 〜 max_days までループ
        # ※ 週が変わると k=0 になる可能性があるため 0 からスタート
        for k in range(max_days + 1): 
            for t in range(T - 1):
                
                # 週またぎチェック
                current_week = t // 168
                next_week = (t + 1) // 168
                is_new_week = (next_week > current_week)
                
                # === State 0: WORK ===
                for d in range(1, max_w + 1):
                    u = (t, d, 0, k)
                    # このノードが存在しないならスキップ（グラフサイズ削減）
                    # ※ 動的に生成してもいいが、ここでは全探索ループで簡易化
                    
                    cost_next = group['hourly_cost'] + group['preference_penalty'][t+1]
                    
                    # 次の k を決定 (週が変わればリセット)
                    # 継続勤務の場合、そのシフトは「前の週」のカウントに含まれるため
                    # 新しい週のカウントとしては 0 (まだ新しいシフトは始めていない) 扱いとする
                    next_k_cont = 0 if is_new_week else k
                    
                    # (A) 勤務継続
                    if d < max_w:
                        v_cont = (t+1, d+1, 0, next_k_cont)
                        G.add_edge(u, v_cont, weight=cost_next)
                    
                    # (B) 退勤 -> Rest
                    if d >= min_w:
                        # 休憩に入るので、次の週のカウントは0
                        next_k_rest = 0 if is_new_week else k
                        v_rest = (t+1, 1, 1, next_k_rest)
                        G.add_edge(u, v_rest, weight=0)
                        G.add_edge(u, sink, weight=0)

                # === State 1: REST ===
                for r in range(1, min_rest + 1):
                    u = (t, r, 1, k)
                    
                    # 週が変われば k=0 にリセット
                    next_k_rest = 0 if is_new_week else k
                    
                    # (C) 休息継続
                    next_r = min(r + 1, min_rest)
                    v_cont = (t+1, next_r, 1, next_k_rest)
                    G.add_edge(u, v_cont, weight=0)
                    
                    # (D) 再出勤 (Rest -> Work)
                    # ここで k が増える (週が変わっていれば 0->1, 同じ週なら k->k+1)
                    if r >= min_rest:
                        # 新しい k を計算
                        if is_new_week:
                            target_k = 1 # 新しい週の1回目
                        else:
                            target_k = k + 1 # 同じ週の k+1回目
                        
                        # 上限チェック
                        if target_k <= max_days:
                            cost_work = group['base_cost'] + cost_next
                            v_work = (t+1, 1, 0, target_k)
                            G.add_edge(u, v_work, weight=cost_work)
                    
                    G.add_edge(u, sink, weight=0)

        # 最終処理
        for k in range(max_days + 1):
            for d in range(min_w, max_w + 1):
                if G.has_node((T-1, d, 0, k)): G.add_edge((T-1, d, 0, k), sink, weight=0)
            for r in range(1, min_rest + 1):
                if G.has_node((T-1, r, 1, k)): G.add_edge((T-1, r, 1, k), sink, weight=0)
            
        return G, source, sink

    def solve(self):
        print(f"Building Exact Model for {self.prob.n_weeks} weeks (T={self.prob.T})...")
        start_time = time.time()
        
        model = pulp.LpProblem("Monthly_StaffScheduling", pulp.LpMinimize)
        
        flow_vars = {} 
        work_at_t = [[] for _ in range(self.prob.T)]
        graphs = {}
        objective_terms = []
        
        total_nodes = 0
        
        for k in range(self.prob.K):
            G, source, sink = self.build_network(k)
            graphs[k] = (G, source, sink)
            flow_vars[k] = {}
            total_nodes += len(G.nodes())
            
            for u, v, data in G.edges(data=True):
                var_name = f"x{k}_{hash(str(u))}_{hash(str(v))}"
                x = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)
                flow_vars[k][(u, v)] = x
                
                if data['weight'] > 0:
                    objective_terms.append(data['weight'] * x)
                
                if v != sink and isinstance(v, tuple) and v[2] == 0:
                    t_idx = v[0]
                    work_at_t[t_idx].append(x)
            
            # Flow Conservation
            for node in G.nodes():
                if node == source or node == sink: continue
                in_edges = G.in_edges(node)
                out_edges = G.out_edges(node)
                model += (pulp.lpSum([flow_vars[k][e] for e in in_edges]) == 
                          pulp.lpSum([flow_vars[k][e] for e in out_edges]))
            
            # Max Employees
            source_edges = G.out_edges(source)
            model += pulp.lpSum([flow_vars[k][e] for e in source_edges]) <= self.prob.groups[k]['max_employees']

        # Demand Constraints
        for t in range(self.prob.T):
            s = pulp.LpVariable(f"s_{t}", lowBound=0)
            objective_terms.append(self.prob.big_m * s)
            model += pulp.lpSum(work_at_t[t]) + s >= self.prob.demand[t]

        model += pulp.lpSum(objective_terms)

        print(f"Model Built. Total Nodes across layers: {total_nodes}")
        print("Solving with CBC (This may take 1-2 minutes)...")
        
        # タイムアウト設定 (最大300秒)
        status = model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300))
        
        print(f"Solved in {time.time() - start_time:.2f} s. Status: {pulp.LpStatus[status]}")
        print(f"Objective Value: {pulp.value(model.objective):,.2f}")
        
        return self.reconstruct_solution(model, flow_vars, graphs)

    def reconstruct_solution(self, model, flow_vars, graphs):
        assignments = []
        for k in range(self.prob.K):
            G, source, sink = graphs[k]
            
            active_edges = []
            for u, v in G.edges():
                val = pulp.value(flow_vars[k][(u, v)])
                if val is not None and val > 0.5:
                    count = int(round(val))
                    for _ in range(count):
                        active_edges.append((u, v))
            
            temp_G = nx.MultiDiGraph()
            temp_G.add_edges_from(active_edges)
            
            while True:
                try:
                    path = nx.shortest_path(temp_G, source, sink)
                    schedule = [0] * self.prob.T
                    total_cost = 0
                    
                    # 週ごとの勤務日数を集計するための配列
                    weekly_shifts = [0] * self.prob.n_weeks
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        total_cost += G[u][v]['weight']
                        
                        if v != sink and isinstance(v, tuple):
                            t, d, state, k_idx = v
                            if state == 0: 
                                schedule[t] = 1
                            
                            # 出勤回数の記録 (k_idx はその週の累積回数)
                            # 各週の最大値を記録すれば、その週の総出勤数がわかる
                            week_idx = t // 168
                            if week_idx < self.prob.n_weeks:
                                weekly_shifts[week_idx] = max(weekly_shifts[week_idx], k_idx)
                        
                        keys = list(temp_G[u][v].keys())
                        if keys: temp_G.remove_edge(u, v, key=keys[0])
                    
                    assignments.append({
                        'column': {'group_id': k, 'schedule': schedule, 'cost': total_cost},
                        'count': 1,
                        'weekly_shifts': weekly_shifts # 配列で保存
                    })
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                    break
        return assignments

# ==========================================
# 3. 可視化 (1ヶ月版)
# ==========================================
def visualize_results(prob, assignments):
    T = prob.T
    hours = np.arange(T)
    group_supply = np.zeros((prob.K, T))
    gantt_data = {k: [] for k in range(prob.K)}
    total_cost = 0
    
    for item in assignments:
        col = item['column']
        cnt = item['count']
        gid = col['group_id']
        sched = np.array(col['schedule'])
        total_cost += col['cost'] * cnt
        group_supply[gid] += sched * cnt
        if cnt > 0:
            gantt_data[gid].append(item)

    # 横幅を広く取る
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(prob.K + 1, 1, height_ratios=[3] + [1]*prob.K)
    
    ax_main = fig.add_subplot(gs[0])
    bottom = np.zeros(T)
    for k in range(prob.K):
        grp = prob.groups[k]
        ax_main.bar(hours, group_supply[k], bottom=bottom, label=grp['name'], color=grp['color'], width=1.0, alpha=0.8, align='edge')
        bottom += group_supply[k]
    ax_main.step(hours, prob.demand, where='post', color='red', linewidth=1.5, linestyle='--', label='Demand')
    ax_main.set_title(f"1-Month Schedule (Exact MIP): Total Cost = {total_cost:,.0f}", fontsize=14)
    ax_main.set_xlim(0, T)
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # X軸: 週ごとの区切り線を強調
    xticks = np.arange(0, T+1, 24 * 7) # 1週間ごと
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels([f"Week {i+1}" for i in range(len(xticks))])
    for x in xticks: ax_main.axvline(x, color='k', linewidth=1.5, alpha=0.5)
    
    # 日付ごとの細かい線
    minor_ticks = np.arange(0, T+1, 24)
    for x in minor_ticks: ax_main.axvline(x, color='gray', linewidth=0.5, alpha=0.2)
    
    for k in range(prob.K):
        ax = fig.add_subplot(gs[k+1], sharex=ax_main)
        grp = prob.groups[k]
        items = gantt_data[k]
        y_pos = 0
        
        # ソート
        items.sort(key=lambda x: sum(x['weekly_shifts']), reverse=True)

        for item in items:
            sched = item['column']['schedule']
            weekly_shifts = item['weekly_shifts']
            
            diff = np.diff(np.hstack(([0], sched, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                rect = mpatches.Rectangle((s, y_pos), e-s, 0.8, facecolor=grp['color'], edgecolor='black', alpha=0.6)
                ax.add_patch(rect)
            
            # 各週の勤務日数を表示
            info_text = "Days/Wk: " + ", ".join(map(str, weekly_shifts))
            ax.text(T+10, y_pos+0.4, info_text, va='center', fontsize=8, color='black')
            
            y_pos += 1
        ax.set_ylabel(grp['name'], rotation=0, ha='right', fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(0, max(1, y_pos))
        
        # 週区切り線
        for x in xticks: ax.axvline(x, color='k', linewidth=1.5, alpha=0.5)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== 1-Month Exact Shift Scheduling ===")
    problem = ShiftSchedulingProblem(n_weeks=4)
    solver = ExactTimeSpaceMIPSolver(problem)
    assignments = solver.solve()
    visualize_results(problem, assignments)
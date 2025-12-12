import pulp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ==========================================
# 1. 問題設定: データ生成クラス
# ==========================================
class ShiftSchedulingProblem:
    def __init__(self, n_weeks=4):
        # 期間設定: 1週間 = 168時間
        self.T = 168 * n_weeks
        self.hours = np.arange(self.T)
        
        np.random.seed(42)
        
        # --- 需要データの生成 ---
        # 基本パターン (24h)
        base_pattern = np.array([1,1,1,1,1, 2,3,6,8,10, 11,10,11,12,13, 12,10,8,6, 4,3,2,2,1])
        weekly_demand = np.tile(base_pattern, 7 * n_weeks)
        
        # 週末(金夜〜日)の需要増
        for t in range(self.T):
            day = (t // 24) % 7
            hour = t % 24
            # 金曜夜〜日曜
            if day == 4 and hour >= 18: weekly_demand[t] *= 1.3
            elif day >= 5: weekly_demand[t] *= 1.2
            
        # ノイズ付加
        noise = np.random.randint(-1, 2, self.T)
        self.demand = np.clip(weekly_demand + noise, 0, None).astype(int)
        
        # --- 従業員グループ定義 (K) ---
        self.groups = []
        
        # Group 0: 正社員 (Full-time)
        # 特徴: 長時間勤務(8h~), 休憩必須, インターバル厳守(11h)
        g0_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 7 or h > 20: g0_penalty[t] = 100 # 深夜早朝は嫌がる
            
        self.groups.append({
            'id': 0,
            'name': 'Full-time',
            'max_employees': 8,
            'base_cost': 1500, # 固定給的要素
            'hourly_cost': 2000,
            'preference_penalty': g0_penalty,
            'min_work': 8,
            'max_work': 10,
            'rest_interval': 8, # 勤務間インターバル
            'max_shifts': 5,
            'color': 'tab:blue'
        })

        # Group 1: パートタイム (Part-time A)
        # 特徴: 短時間(4-6h), インターバル短い, コスト安い
        g1_penalty = np.zeros(self.T)
        for t in range(self.T):
            h = t % 24
            if h < 9: g1_penalty[t] = 50
            if h > 17: g1_penalty[t] = 5000 # 夕方以降はNG
            
        self.groups.append({
            'id': 1,
            'name': 'Part-time (Day)',
            'max_employees': 10,
            'base_cost': 1000,
            'hourly_cost': 1400,
            'preference_penalty': g1_penalty,
            'min_work': 4,
            'max_work': 6,
            'rest_interval': 8,
            'max_shifts': 5,
            'color': 'tab:orange'
        })

        # Group 2: 学生バイト (Part-time B)
        # 特徴: 夜勤OK, 週末OK
        g2_penalty = np.zeros(self.T)
        for t in range(self.T):
            d = (t // 24) % 7
            h = t % 24
            if d < 5 and 9 <= h <= 17: g2_penalty[t] = 5000 # 平日昼間は授業
            
        self.groups.append({
            'id': 2,
            'name': 'Student (Night/Wknd)',
            'max_employees': 10,
            'base_cost': 1000,
            'hourly_cost': 1100,
            'preference_penalty': g2_penalty,
            'min_work': 3,
            'max_work': 8,
            'rest_interval': 8, # 若いので回復が早い(仮定)
            'max_shifts': 5,
            'color': 'tab:green'
        })
        
        self.K = len(self.groups)
        self.big_m = 5000  # 需要未達ペナルティ

# ==========================================
# 2. ソルバー: 列生成法 + 時間空間NW + 列プール
# ==========================================
class AdvancedColumnGenerationSolver:
    def __init__(self, problem):
        self.prob = problem
        
        # 列プール (All generated columns)
        # 構造: list of dict {'group_id': k, 'schedule': [0,1,...], 'cost': c, 'active': bool}
        self.pool = [] 
        
        # RMPで現在使用中の列インデックス集合
        self.rmp_indices = []
        
        self.duals = {'demand': None, 'convex': None}

    def initialize(self):
        """初期列の生成（実行可能解確保のため高コストなダミー列を追加）"""
        print("Initializing with dummy columns...")
        for k in range(self.prob.K):
            # 全時間帯勤務というありえない列（Big-Mコスト）
            sched = [1] * self.prob.T
            cost = 1e9
            self.add_column_to_pool(k, sched, cost, initial=True)

    def add_column_to_pool(self, group_id, schedule, cost, initial=False):
        """プールに列を追加し、RMPの使用フラグを立てる"""
        col_idx = len(self.pool)
        self.pool.append({
            'id': col_idx,
            'group_id': group_id,
            'schedule': schedule,
            'cost': cost,
            'active': True # 新規追加時はRMPに入れる
        })
        self.rmp_indices.append(col_idx)

    def solve_rmp(self, integer=False):
        """制限付き主問題(RMP)を解く"""
        model = pulp.LpProblem("StaffScheduling_RMP", pulp.LpMinimize)
        
        # 変数定義
        x_vars = {}
        active_pool = [self.pool[i] for i in self.rmp_indices]
        
        cat = pulp.LpInteger if integer else pulp.LpContinuous
        
        for col in active_pool:
            x_vars[col['id']] = pulp.LpVariable(f"x_{col['id']}", lowBound=0, cat=cat)
            
        # スラック変数（需要未達ペナルティ）
        s_vars = [pulp.LpVariable(f"s_{t}", lowBound=0) for t in range(self.prob.T)]
        
        # 目的関数
        model += pulp.lpSum([col['cost'] * x_vars[col['id']] for col in active_pool]) + \
                 pulp.lpSum([self.prob.big_m * s for s in s_vars])
                 
        # 制約1: 需要制約
        demand_constrs = []
        for t in range(self.prob.T):
            # column['schedule'][t] が 1 のものだけ集計
            expr = pulp.lpSum([col['schedule'][t] * x_vars[col['id']] for col in active_pool]) + s_vars[t] >= self.prob.demand[t]
            name = f"Demand_Constraint_{t}"
            model += expr, name
            demand_constrs.append(name)
            
        # 制約2: グループ人数制約 (凸性制約)
        convex_constrs = []
        for k in range(self.prob.K):
            expr = pulp.lpSum([x_vars[col['id']] for col in active_pool if col['group_id'] == k]) <= self.prob.groups[k]['max_employees']
            name = f"Convex_Constraint_{k}"
            model += expr, name
            convex_constrs.append(name)
            
        # 解く
        solver_msg = 0
        model.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
        
        if integer:
            # 整数解の抽出
            assignments = []
            for col in active_pool:
                val = pulp.value(x_vars[col['id']])
                if val is not None and val > 0.5:
                    assignments.append({'column': col, 'count': int(round(val))})
            return assignments
        else:
            # 双対変数の取得
            if model.status == 1:
                # PuLPから双対変数を取得 (constraints辞書経由)
                self.duals['demand'] = [model.constraints[name].pi for name in demand_constrs]
                self.duals['convex'] = [model.constraints[name].pi for name in convex_constrs]
                return pulp.value(model.objective)
            else:
                return float('inf')

    def build_time_expanded_network(self, group_id, dual_demand):
        """
        1ヶ月対応・週次リセット付きネットワーク構築
        Nodes: (t, duration, state, shift_count)
               shift_count: 「その週における」勤務開始回数 (0 ~ max_shifts)
        """
        group = self.prob.groups[group_id]
        G = nx.DiGraph()
        source, sink = 'Source', 'Sink'
        
        T = self.prob.T
        min_w, max_w = group['min_work'], group['max_work']
        min_rest = group['rest_interval']
        max_shifts = group.get('max_shifts', 5)
        
        # --- 1. Sourceからの遷移 ---
        # 最初の週の1回目の勤務 (k=1)
        for t in range(T - min_w + 1):
            cost = group['base_cost'] + \
                   (group['hourly_cost'] + group['preference_penalty'][t]) - dual_demand[t]
            # 初期状態なので k=1 スタート
            G.add_edge(source, (t, 1, 0, 1), weight=cost)

        # --- 2. タイムステップごとの遷移 ---
        # k: 現在の週の勤務回数 (0回〜max_shifts回)
        # ※ k=0 は「週が変わってリセットされた直後」の状態を表すために必要
        for k in range(0, max_shifts + 1):
            for t in range(T - 1):
                
                # --- 週またぎ判定 ---
                # t+1 が 168の倍数なら、次の時刻は「新しい週の始まり」
                is_new_week = ((t + 1) % 168 == 0)
                
                # 次の時刻の k (ベース)
                # 新しい週なら 0 にリセット、そうでなければ k を維持
                k_next_base = 0 if is_new_week else k
                
                # コスト計算
                cost_next = (group['hourly_cost'] + group['preference_penalty'][t+1]) - dual_demand[t+1]

                # === State 0: WORK (勤務中) ===
                # ルール: 勤務中の週またぎは、勤務開始時点の週カウントに属するとみなすのが一般的だが、
                # ここではシンプルに「週が変わったら回数カウントは0(新規週分)扱い」に移行する実装にします。
                for d in range(1, max_w + 1):
                    u = (t, d, 0, k)
                    
                    # (A) 勤務継続
                    if d < max_w:
                        v_cont = (t+1, d+1, 0, k_next_base) # 新週ならk=0, 同週ならk維持
                        G.add_edge(u, v_cont, weight=cost_next)
                    
                    # (B) 退勤 -> 休息へ
                    if d >= min_w:
                        v_rest = (t+1, 1, 1, k_next_base)
                        G.add_edge(u, v_rest, weight=0)
                        
                        # 期間終了ならSinkへ
                        G.add_edge(u, sink, weight=0)

                # === State 1: REST (休息中) ===
                for r in range(1, min_rest + 1):
                    u = (t, r, 1, k)
                    
                    # (C) 休息継続
                    next_r = min(r + 1, min_rest)
                    v_cont = (t+1, next_r, 1, k_next_base)
                    G.add_edge(u, v_cont, weight=0)
                    
                    # (D) 再出勤 (重要)
                    # 条件: 休息十分 AND ( (新週である) OR (同週内で回数上限未満) )
                    can_start_work = False
                    next_work_k = -1
                    
                    if r >= min_rest:
                        if is_new_week:
                            # 新しい週なので、無条件で1回目(k=1)として出勤可
                            can_start_work = True
                            next_work_k = 1
                        elif k < max_shifts:
                            # 同週内なら、上限未満の場合のみ出勤可(k -> k+1)
                            can_start_work = True
                            next_work_k = k + 1
                    
                    if can_start_work:
                        v_work = (t+1, 1, 0, next_work_k)
                        cost_work = group['base_cost'] + cost_next
                        G.add_edge(u, v_work, weight=cost_work)
                    
                    # Sinkへの接続
                    G.add_edge(u, sink, weight=0)

        # 最終時刻 T-1 の処理
        for k in range(0, max_shifts + 1): # k=0も処理対象
            for d in range(min_w, max_w + 1):
                if G.has_node((T-1, d, 0, k)):
                    G.add_edge((T-1, d, 0, k), sink, weight=0)
            for r in range(1, min_rest + 1):
                if G.has_node((T-1, r, 1, k)):
                    G.add_edge((T-1, r, 1, k), sink, weight=0)
                
        return G, source, sink

    def pricing_step(self):
        """Pricing Problem: Heuristic check -> Exact Search"""
        total_new_cols = 0  # 全体で見つかった列の総数
        
        for k in range(self.prob.K):
            group_new_cols = 0  # ★重要: このグループ内で見つかった列数
            
            group = self.prob.groups[k]
            mu = self.duals['convex'][k]
            pi = self.duals['demand']
            
            # --- (A) Heuristic Pricing (プール探索) ---
            # プール内の既存列の被約費用を計算
            
            # RMPに含まれていない列だけチェック
            inactive_indices = [i for i in range(len(self.pool)) 
                              if self.pool[i]['group_id'] == k and not self.pool[i]['active']]
            
            for idx in inactive_indices:
                col = self.pool[idx]
                sum_pi = sum(pi[t] for t in range(self.prob.T) if col['schedule'][t] == 1)
                rc = col['cost'] - sum_pi - mu
                
                if rc < -1e-5:
                    self.pool[idx]['active'] = True
                    self.rmp_indices.append(idx)
                    group_new_cols += 1
                    total_new_cols += 1
            
            # ★修正点: 「このグループで」列が見つかった場合のみ、このグループのExact探索をスキップ
            # (以前のコードはここで全グループの探索を止めてしまっていたのが原因)
            if group_new_cols > 0:
                continue

            # --- (B) Exact Pricing (グラフ探索) ---
            # プールに良い列がない場合のみ実行
            G, source, sink = self.build_time_expanded_network(k, pi)
            
            try:
                # Bellman-Fordで最短路探索 (負の重み対応)
                path = nx.bellman_ford_path(G, source, sink, weight='weight')
                path_len = nx.bellman_ford_path_length(G, source, sink, weight='weight')
                
                reduced_cost = path_len - mu
                
                if reduced_cost < -1e-4:
                    # === ここが定義されていなかった部分です ===
                    # パスからスケジュール配列(0/1)と実コストを復元する処理
                    
                    schedule = [0] * self.prob.T
                    current_work_cost = 0
                    
                    # パス解析: Source -> (t,d,s) -> ... -> Sink
                    for i in range(len(path) - 1):
                        u = path[i] # 現在ノード
                        v = path[i+1] # 次のノード
                        
                        if u == source:
                            pass
                        elif v == sink:
                            pass
                        else:
                            # ノード形式 v = (時刻t, 継続時間d, 状態state)
                            t, d, state, k_count = v
                            
                            # state 0 (WORK) の場合のみ勤務フラグを立てる
                            if state == 0: 
                                schedule[t] = 1
                                # ★重要: ここで G[u][v]['weight'] を使ってはいけません
                                # 必ず元の group 定義から「時給」と「ペナルティ」だけを足します
                                raw_hourly_cost = group['hourly_cost'] + group['preference_penalty'][t]
                                current_work_cost += raw_hourly_cost
                    
                    # 基本給(Base Cost)の加算ロジック
                    # シフト回数（0->1の立ち上がり回数）× BaseCost
                    # 今回のモデルでは1回勤務なので単純に1回分足す、あるいは不連続勤務を許容するなら回数分足す
                    arr = np.array(schedule)
                    # [0, 1, 1, 0] -> diff -> [1, 0, -1] -> 1がある場所が開始点
                    diff = np.diff(np.hstack(([0], arr)))
                    starts = np.sum(diff == 1)
                    
                    real_cost = current_work_cost + starts * group['base_cost']
                    
                    # === 復元処理ここまで ===

                    # プールに追加
                    self.add_column_to_pool(k, schedule, real_cost)
                    total_new_cols += 1
                    
            except nx.NetworkXNoPath:
                # 負閉路などでパスが見つからない、または到達不能な場合
                pass
                
        return total_new_cols

    def solve(self, max_iter=100):
        start_time = time.time()
        self.initialize()
        
        obj_history = []
        print(f"{'Iter':<5} | {'Obj Value':<12} | {'New Cols':<8} | {'Active':<8} | {'Total':<8} | {'Time (s)':<8}")
        print("-" * 75)
        
        for it in range(max_iter):
            # 1. RMPを解く
            obj = self.solve_rmp(integer=False)
            obj_history.append(obj)
            
            # 2. Aging (今回は簡易化のため実装省略: active=Falseにする処理)
            # 例: 5イテレーションごとに掃除を実行
            if it > 0 and it % 5 == 0:
                removed = self.cleanup_columns(threshold=0.001)
                if removed > 0:
                    print(f"   [Cleanup] Removed {removed} inefficient columns from RMP.")
            
            # 3. Pricing
            new_cols = self.pricing_step()
            
            elapsed = time.time() - start_time
            active_count = len(self.rmp_indices)  # 現在RMP計算に使っている列数
            pool_count = len(self.pool)           # 過去に生成された全列数
            
            print(f"{it:<5} | {obj:<12.2f} | {new_cols:<8} | {active_count:<8} | {pool_count:<8} | {elapsed:<8.2f}")
            
            if new_cols == 0:
                print("Convergence reached (No negative reduced cost columns).")
                break
                
        print("\nSolving Integer Master Problem (Heuristic)...")
        # assignments = self.solve_rmp(integer=True) # 元の重い処理
        assignments = self.solve_approximated_ip()   # ★新しい高速処理
        
        print(f"Optimal Integer Solution Found. Total assignments: {sum(a['count'] for a in assignments)}")
        
        return assignments, obj_history
    
    def cleanup_columns(self, threshold=1e-5):
        """
        有望でない列（被約費用が正の列）をRMPの対象から外す
        ※ poolからは削除せず、rmp_indicesから除外する
        """
        if self.duals['demand'] is None:
            return 0

        pi = self.duals['demand']
        mu = self.duals['convex']
        
        current_rmp_set = set(self.rmp_indices)
        kept_indices = []
        removed_count = 0
        
        # 現在RMPにある列をチェック
        for idx in self.rmp_indices:
            col = self.pool[idx]
            k = col['group_id']
            
            # 被約費用の計算: RC = Cost - (sum(pi) + mu)
            # ※ schedule[t] == 1 の箇所の pi を足す
            sum_pi = sum(pi[t] for t in range(self.prob.T) if col['schedule'][t] == 1)
            rc = col['cost'] - sum_pi - mu[k]
            
            # 判定ロジック:
            # RC が閾値より小さい（負、または0に近い）場合は残す
            # つまり「役に立つ可能性がある列」は保持
            if rc < threshold:
                kept_indices.append(idx)
            else:
                # RCが大きくプラスなら、今は不要なのでRMPから外す
                # ただし、activeフラグは残しておき、Pricingでの再利用に備える実装も可能
                # 今回はシンプルにリストから外すだけにする
                col['active'] = False # フラグも落としておく
                removed_count += 1
                
        self.rmp_indices = kept_indices
        return removed_count
    

    def solve_approximated_ip(self, threshold=1e-5):
        """
        LP緩和解の近傍探索による高速な整数解構築
        (値が0に近い列を削除してMIPを解く)
        """
        print(f"\nConstructing Integer Solution from LP neighborhood (Threshold > {threshold})...")
        
        # 1. まずLP（連続緩和）としてモデルを構築・求解し、変数の値を取得する
        #    (既存のsolve_rmpは目的関数値しか返さないため、ここでフィルタリング用の実行を行う)
        
        model = pulp.LpProblem("Filtering_Phase", pulp.LpMinimize)
        
        # 現在のプールにある全列を変数化
        active_pool = [self.pool[i] for i in self.rmp_indices]
        x_vars = {}
        
        for col in active_pool:
            # 連続変数として定義
            x_vars[col['id']] = pulp.LpVariable(f"x_filter_{col['id']}", lowBound=0, cat=pulp.LpContinuous)
            
        s_vars = [pulp.LpVariable(f"s_filter_{t}", lowBound=0) for t in range(self.prob.T)]
        
        # 目的関数
        model += pulp.lpSum([col['cost'] * x_vars[col['id']] for col in active_pool]) + \
                 pulp.lpSum([self.prob.big_m * s for s in s_vars])
        
        # 制約 (需要)
        for t in range(self.prob.T):
            model += pulp.lpSum([col['schedule'][t] * x_vars[col['id']] for col in active_pool]) + s_vars[t] >= self.prob.demand[t]
            
        # 制約 (凸性/人数)
        for k in range(self.prob.K):
            model += pulp.lpSum([x_vars[col['id']] for col in active_pool if col['group_id'] == k]) <= self.prob.groups[k]['max_employees']
            
        # LPを解く (高速)
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 2. 値が閾値以上の列だけを抽出 (ここが近傍探索の肝)
        promising_indices = []
        kept_count = 0
        
        for col in active_pool:
            val = pulp.value(x_vars[col['id']])
            if val is not None and val > threshold:
                promising_indices.append(col['id'])
                kept_count += 1
                
        print(f"  -> Filtering: Reduced search space from {len(active_pool)} to {kept_count} columns.")
        
        # 3. 抽出した列だけでMIPを解く
        #    既存の solve_rmp は self.rmp_indices を参照するため、一時的に書き換えるトリックを使う
        
        original_indices = self.rmp_indices.copy() # バックアップ
        self.rmp_indices = promising_indices       # フィルタ済みインデックスに差し替え
        
        try:
            # 変数数が激減しているため、整数条件でも高速に解ける
            assignments = self.solve_rmp(integer=True)
        finally:
            # 必ず元に戻す
            self.rmp_indices = original_indices
            
        return assignments

# ==========================================
# 3. 結果の可視化
# ==========================================
def visualize_results(prob, assignments):
    T = prob.T
    hours = np.arange(T)
    
    # グループごとの供給量集計
    group_supply = np.zeros((prob.K, T))
    # ガントチャート用データ
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
            gantt_data[gid].append({'schedule': sched, 'count': cnt})

    # --- Plotting ---
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(prob.K + 1, 1, height_ratios=[3] + [1]*prob.K)
    
    # 1. Supply vs Demand (Stacked Bar)
    ax_main = fig.add_subplot(gs[0])
    bottom = np.zeros(T)
    
    for k in range(prob.K):
        grp = prob.groups[k]
        ax_main.bar(hours, group_supply[k], bottom=bottom, 
                   label=grp['name'], color=grp['color'], 
                   width=1.0, alpha=0.8, align='edge')
        bottom += group_supply[k]
        
    # Demand Line
    ax_main.step(hours, prob.demand, where='post', color='red', linewidth=2, linestyle='--', label='Demand')
    
    ax_main.set_title(f"Optimization Result: Total Cost = {total_cost:,.0f}", fontsize=14)
    ax_main.set_xlim(0, T)
    ax_main.set_ylabel("Number of Staff")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # X軸: 日付表示
    xticks = np.arange(0, T+1, 24)
    xticklabels = [f"Day {i}" for i in range(len(xticks))]
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels(xticklabels)
    for x in xticks: ax_main.axvline(x, color='k', alpha=0.2)
    
    # 2. Gantt Charts per Group
    for k in range(prob.K):
        ax = fig.add_subplot(gs[k+1], sharex=ax_main)
        grp = prob.groups[k]
        items = gantt_data[k]
        
        y_pos = 0
        for item in items:
            sched = item['schedule']
            cnt = item['count']
            
            # 勤務区間の検出 (0->1, 1->0)
            diff = np.diff(np.hstack(([0], sched, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for s, e in zip(starts, ends):
                dur = e - s
                # シフトバー
                rect = mpatches.Rectangle((s, y_pos), dur, 0.8, 
                                        facecolor=grp['color'], edgecolor='black', alpha=0.6)
                ax.add_patch(rect)
                # 人数表示
                if dur > 2:
                    ax.text(s + dur/2, y_pos+0.4, f"x{cnt}", 
                           ha='center', va='center', color='white', fontweight='bold', fontsize=8)
            y_pos += 1
            
        # 背景にペナルティ可視化 (Heatmap)
        penalty = grp['preference_penalty']
        ax.imshow([penalty], aspect='auto', cmap='Reds', alpha=0.2, 
                 extent=[0, T, 0, max(1, y_pos)], vmin=0, vmax=max(penalty.max(), 1))
        
        ax.set_ylabel(grp['name'], rotation=0, ha='right', fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(0, max(1, y_pos))
        for x in xticks: ax.axvline(x, color='k', alpha=0.2)
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    print("=== Column Generation Staff Scheduling ===")
    
    # 1. 問題生成 (1週間分)
    problem = ShiftSchedulingProblem(n_weeks=4)
    
    # 2. ソルバー初期化と実行
    solver = AdvancedColumnGenerationSolver(problem)
    final_assignments, history = solver.solve(max_iter=30)
    
    # 3. 結果可視化
    visualize_results(problem, final_assignments)
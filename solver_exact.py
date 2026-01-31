import pulp
import time
import numpy as np
import pandas as pd
import re
import os
from collections import defaultdict
from problem import GraphBuilder

class ExactMIPSolver:
    def __init__(self, problem):
        self.prob = problem
        self.breakdown = {} # 内訳保存用
        self.stats = {}     # 統計情報保存用

    def parse_cbc_log(self, log_path):
        """
        CBCのログを解析して、(経過時間, 目的関数値) の推移リストを作成する
        """
        trajectory = []
        if not os.path.exists(log_path):
            return trajectory
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Regex: "Integer solution of 1234.5 found after 0.12 seconds"
        pattern_sol = re.compile(r"Integer solution of\s+([-\d\.]+)\s+found.*?\(([\d\.]+)\s+seconds\)")
        matches = pattern_sol.findall(content)
        
        for obj_str, time_str in matches:
            try:
                t = float(time_str)
                obj = float(obj_str)
                trajectory.append((t, obj))
            except ValueError:
                continue
        
        # 時間順にソート
        trajectory.sort(key=lambda x: x[0])
        return trajectory

    def solve(self, time_limit=300, gapRel=0.01):
        start_time = time.time()
        self.stats = {} # Reset stats

        model = pulp.LpProblem("ExactMIP", pulp.LpMinimize)
        flow_vars = {} 
        obj_terms = []
        work_vars_mapping = []
        demand_vars = defaultdict(list)
        
        # --- モデル構築 (変更なし) ---
        for k in range(self.prob.K):
            G, src, sink = GraphBuilder.build_graph(self.prob, k)
            for u, v, d in G.edges(data=True):
                var_name = f"x_{k}_{hash((u,v))}"
                x = pulp.LpVariable(var_name, 0, 1, pulp.LpBinary)
                flow_vars[(k, u, v)] = x
                if d['weight'] > 0: obj_terms.append(d['weight'] * x)
                
                if d.get('type') in ['work_start', 'work_cont']:
                    t = d['time']
                    demand_vars[t].append(x)
                    work_vars_mapping.append((k, t, x))
            
            model += pulp.lpSum([flow_vars[(k, src, v)] for v in G.successors(src)]) == 1
            for n in G.nodes():
                if n == src or n == sink: continue
                vin = pulp.lpSum([flow_vars[(k, u, n)] for u in G.predecessors(n)])
                vout = pulp.lpSum([flow_vars[(k, n, v)] for v in G.successors(n)])
                model += vin == vout
            model += pulp.lpSum([flow_vars[(k, u, sink)] for u in G.predecessors(sink)]) == 1

        slacks = [pulp.LpVariable(f"s_{t}", 0) for t in range(self.prob.T)]
        obj_terms.extend([self.prob.big_m * s for s in slacks])
        model += pulp.lpSum(obj_terms)
        
        for t in range(self.prob.T):
            model += pulp.lpSum(demand_vars[t]) + slacks[t] >= self.prob.demand[t]
            
        # --- ソルバー実行設定 (ログ出力追加) ---
        timestamp = int(time.time())
        log_file = f"exact_mip_log_{timestamp}.txt"
        
        # CBCソルバーの設定
        # threads=1 で決定論的な挙動にし、logPathを指定
        solver = pulp.PULP_CBC_CMD(
            msg=0, 
            timeLimit=time_limit, 
            gapRel=gapRel, 
            logPath=log_file,
            options=['randomSeed 42', 'randomCbcSeed 42', 'threads 1']
        )
        
        model.solve(solver)
        elapsed = time.time() - start_time
        
        # --- ログ解析と時刻記録 ---
        trajectory = self.parse_cbc_log(log_file)
        if os.path.exists(log_file):
            os.remove(log_file)
            
        self.stats['mip_trajectory'] = trajectory
        
        if trajectory:
            # trajectoryは (time_in_solver, obj) のリスト
            # Exactの場合、ソルバー起動までの前処理時間は短いので time_in_solver をそのまま採用するか、
            # 前処理時間を加算するか選べますが、ここではソルバー内時間を採用します。
            self.stats['time_first_sol'] = trajectory[0][0]
            self.stats['time_best_sol'] = trajectory[-1][0]
        else:
            # 解が見つからなかった場合、もしくはログ解析失敗時
            if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusInteger:
                # ログには残らなかったが解はある場合（稀なケース）、終了時刻を入れる
                self.stats['time_first_sol'] = elapsed
                self.stats['time_best_sol'] = elapsed
            else:
                self.stats['time_first_sol'] = None
                self.stats['time_best_sol'] = None

        # --- スケジュール復元 ---
        schedule = np.zeros((self.prob.K, self.prob.T))
        for k, t, x in work_vars_mapping:
            if x.varValue is not None and x.varValue > 0.5:
                schedule[k, t] = 1
        
        # --- コスト内訳計算 (変更なし) ---
        total_base_wage = 0.0
        total_mismatch_cost = 0.0
        total_penalty = 0.0
        for k in range(self.prob.K):
            emp = self.prob.employees[k]
            total_base_wage += np.sum(schedule[k] * emp['hourly_wage'])
            total_mismatch_cost += np.sum(schedule[k] * emp['rho'])
        for t, s_var in enumerate(slacks):
            if s_var.varValue is not None and s_var.varValue > 1e-5:
                total_penalty += s_var.varValue * self.prob.big_m

        self.breakdown = {
            'Base Wage': total_base_wage,
            'Mismatch Cost': total_mismatch_cost,
            'Understaffing Penalty': total_penalty
        }
        
        print(f"[Exact] Result Breakdown:")
        print(f"  Base Wage : {total_base_wage:,.0f}")
        print(f"  Total Obj : {pulp.value(model.objective):,.0f}")

        self.final_schedule = schedule 
        
        # ★返り値をCG法に合わせて4つ (obj, time, stats, schedule) に変更
        return pulp.value(model.objective), elapsed, self.stats, schedule

    def save_pool_to_csv(self, filename):
        if not hasattr(self, 'final_schedule'):
            return

        data = []
        for k in range(self.prob.K):
            emp = self.prob.employees[k]
            row_sched = self.final_schedule[k]
            sched_str = "".join(map(str, map(int, row_sched)))
            wage = np.sum(row_sched * emp['hourly_wage'])
            rho = np.sum(row_sched * emp['rho'])
            cost = wage + rho
            
            data.append({
                'col_id': k,
                'emp_id': k,
                'emp_type': emp['type'],
                'cost': cost,
                'schedule_pattern': sched_str,
                'in_final_mip': 1,
                'is_selected': 1,
                'usage_int': 1,
                'usage_rmp': 0
            })
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
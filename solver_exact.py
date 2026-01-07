import pulp
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from problem import GraphBuilder

class ExactMIPSolver:
    def __init__(self, problem):
        self.prob = problem
        self.breakdown = {} # 内訳保存用

    def solve(self, time_limit=300):
        start_time = time.time()
        model = pulp.LpProblem("ExactMIP", pulp.LpMinimize)
        flow_vars = {} 
        obj_terms = []
        work_vars_mapping = []
        demand_vars = defaultdict(list)
        
        # ネットワーク構築と変数定義
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
            
            # フロー保存制約
            model += pulp.lpSum([flow_vars[(k, src, v)] for v in G.successors(src)]) == 1
            for n in G.nodes():
                if n == src or n == sink: continue
                vin = pulp.lpSum([flow_vars[(k, u, n)] for u in G.predecessors(n)])
                vout = pulp.lpSum([flow_vars[(k, n, v)] for v in G.successors(n)])
                model += vin == vout
            model += pulp.lpSum([flow_vars[(k, u, sink)] for u in G.predecessors(sink)]) == 1

        # 欠員変数 (Slack)
        slacks = [pulp.LpVariable(f"s_{t}", 0) for t in range(self.prob.T)]
        obj_terms.extend([self.prob.big_m * s for s in slacks])
        
        # 目的関数
        model += pulp.lpSum(obj_terms)
        
        # 需要制約
        for t in range(self.prob.T):
            model += pulp.lpSum(demand_vars[t]) + slacks[t] >= self.prob.demand[t]
            
        # ソルバー実行
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        model.solve(solver)
        
        elapsed = time.time() - start_time
        
        # スケジュール復元
        schedule = np.zeros((self.prob.K, self.prob.T))
        for k, t, x in work_vars_mapping:
            if x.varValue is not None and x.varValue > 0.5:
                schedule[k, t] = 1
        
        # === 追加: コスト内訳の計算 ===
        total_base_wage = 0.0
        total_mismatch_cost = 0.0
        total_penalty = 0.0
        
        # 1. 人件費と不一致コスト
        for k in range(self.prob.K):
            emp = self.prob.employees[k]
            # 基本給: sum(work_flag * hourly_wage)
            total_base_wage += np.sum(schedule[k] * emp['hourly_wage'])
            # 不一致: sum(work_flag * rho[t])
            total_mismatch_cost += np.sum(schedule[k] * emp['rho'])
            
        # 2. 欠員ペナルティ
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
        print(f"  Mismatch  : {total_mismatch_cost:,.0f}")
        print(f"  Penalty   : {total_penalty:,.0f}")
        print(f"  Total Obj : {pulp.value(model.objective):,.0f}")

        self.final_schedule = schedule # CSV保存用に保持
        
        return pulp.value(model.objective), elapsed, schedule

    def save_pool_to_csv(self, filename):
        """
        可視化ツール(plot_cost_components.py)との互換性のために、
        求まった最適解を'pool'形式のCSVとして出力する機能を追加。
        """
        if not hasattr(self, 'final_schedule'):
            print("  [Warning] No schedule to save. Run solve() first.")
            return

        data = []
        for k in range(self.prob.K):
            emp = self.prob.employees[k]
            row_sched = self.final_schedule[k]
            sched_str = "".join(map(str, map(int, row_sched)))
            
            # コスト計算
            wage = np.sum(row_sched * emp['hourly_wage'])
            rho = np.sum(row_sched * emp['rho'])
            cost = wage + rho
            
            data.append({
                'col_id': k,  # Dummy ID
                'emp_id': k,
                'emp_type': emp['type'],
                'cost': cost,
                'schedule_pattern': sched_str,
                'in_final_mip': 1,  # 採用されたので1
                'is_selected': 1,   # 採用されたので1
                'usage_int': 1,
                'usage_rmp': 0
            })
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"  -> Exact solution saved to: {filename}")
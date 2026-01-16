import os
import json
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import re

def calculate_and_add_reduced_cost(base_dir, methods):
    """
    指定されたディレクトリ構造内の全pool_wk?.csvに対して
    被約費用(reduced_cost)を算出し、列として追加して上書き保存する。
    (修正: 実行不可能を回避するため、不足変数 delta を導入)
    """
    config_path = os.path.join(base_dir, 'problem_config.json')
    if not os.path.exists(config_path):
        print(f"[Error] Configuration file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"[Error] Failed to load json: {e}")
        return
    
    n_employees = config.get('n_employees')
    demand_history = config.get('demand_history', {})
    T = config.get('T', 168)
    BIG_M = 1000000  # 不足変数のペナルティコスト

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if not os.path.exists(method_dir):
            continue
            
        print(f"\nProcessing method: {method}")
        files = [f for f in os.listdir(method_dir) if f.startswith('pool_wk') and f.endswith('.csv')]
        files.sort()

        for file in files:
            file_path = os.path.join(method_dir, file)
            match = re.search(r'pool_wk(\d+)\.csv', file)
            if not match: continue
            week_str = match.group(1)
            
            if week_str not in demand_history:
                continue
            
            demand = np.array(demand_history[week_str])
            if len(demand) != T:
                demand = demand[:min(len(demand), T)]
            calc_T = len(demand)

            try:
                df = pd.read_csv(file_path)
            except:
                continue
            
            # --- LP構築 (修正版) ---
            N_cols = len(df)
            
            # スケジュール行列
            schedules_list = df['schedule_pattern'].astype(str).apply(lambda x: [int(c) for c in x[:calc_T]]).tolist()
            schedules = np.array(schedules_list) # (N_cols, T)
            
            costs = df['cost'].values
            group_ids = df['emp_id'].values

            # 変数構成: [x_0 ... x_N, delta_0 ... delta_T]
            # 合計変数は N_cols + calc_T 個
            
            # 目的関数 c: コスト + BIG_M * delta
            c = np.concatenate([costs, np.full(calc_T, BIG_M)])

            # 不等式制約: 需要を満たす (不足 delta を許容)
            # Sum(sched * x) + delta >= demand
            # => -Sum(sched * x) - delta <= -demand
            # 行列 A_ub の構築
            # x部分: -schedules.T
            # delta部分: -I (単位行列)
            A_ub = np.hstack([-schedules.T, -np.eye(calc_T)])
            b_ub = -demand

            # 等式制約: 各従業員に1つのパターン
            # Sum(x) = 1 (deltaは関係なし)
            A_eq = np.zeros((n_employees, N_cols + calc_T))
            for k in range(n_employees):
                A_eq[k, :N_cols] = (group_ids == k).astype(float)
            b_eq = np.ones(n_employees)

            # ソルバー実行
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
            
            if not res.success:
                print(f"  [Warning] {file}: Still failed. Status: {res.message}")
                continue

            # --- 被約費用の計算 ---
            y = res.ineqlin.marginals # Demand duals (negative for <= constraint)
            nu = res.eqlin.marginals  # Assignment duals
            
            # RC = Cost - (Contribution to demand) * Dual_demand - (Contribution to assign) * Dual_assign
            # Contribution to demand (in >= form) is +1 per active time
            # But scipy form is <=. 
            # Correct logic with scipy outputs:
            # RC = c_x - A_ub_x.T @ y - A_eq_x.T @ nu
            # A_ub_x = -schedules.T
            # RC = costs - (-schedules.T).T @ y - nu[group_ids]
            #    = costs + schedules @ y - nu[group_ids]
            
            term_demand = schedules @ y
            term_assign = nu[group_ids]
            reduced_costs = costs + term_demand - term_assign
            
            df['reduced_cost'] = reduced_costs
            df.to_csv(file_path, index=False)
            print(f"  -> {file}: Success. (Min RC: {reduced_costs.min():.2f})")

if __name__ == "__main__":
    BASE_DIR = "results_5emp"
    TARGET_METHODS = ["std(100)", "acc(100)", "pruning(100)"]
    calculate_and_add_reduced_cost(BASE_DIR, TARGET_METHODS)
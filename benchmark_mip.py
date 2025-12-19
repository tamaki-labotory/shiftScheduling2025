import os
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
import pulp
import numpy as np

# 既存モジュールのインポート
# ※ 実行環境にこれらのファイルが存在することを前提としています
from problem import ShiftProblemData
from solver_cg import ColumnGenerationSolver
from solver_cg_pruning import ColumnGenerationSolverWithAging

# ファイルがないソルバーについてはimportエラーを回避するためのダミーまたはtry-except
try:
    from solver_cg_lru import ColumnGenerationSolverLRU
except ImportError:
    ColumnGenerationSolverLRU = None

try:
    from solver_cg_smart import ColumnGenerationSolverSmart
except ImportError:
    ColumnGenerationSolverSmart = None

# -----------------------------------------------------------------------------
# Solver設定
# -----------------------------------------------------------------------------
SOLVER_MAPPING = {
    'Std': {'class': ColumnGenerationSolver, 'kwargs': {'use_pool': False}, 'color': 'red', 'linestyle': '--'},
    'Pool': {'class': ColumnGenerationSolver, 'kwargs': {'use_pool': True}, 'color': 'green', 'linestyle': '-'},
    'Aging': {'class': ColumnGenerationSolverWithAging, 'kwargs': {'use_pool': True}, 'color': 'blue', 'linestyle': '-.'},
}

# 追加のソルバーがあればここに追加
if ColumnGenerationSolverLRU:
    SOLVER_MAPPING['LRU'] = {'class': ColumnGenerationSolverLRU, 'kwargs': {'use_pool': True}, 'color': 'orange', 'linestyle': ':'}
if ColumnGenerationSolverSmart:
    SOLVER_MAPPING['Smart'] = {'class': ColumnGenerationSolverSmart, 'kwargs': {'use_pool': True}, 'color': 'purple', 'linestyle': '-'}

# -----------------------------------------------------------------------------
# CBCログ解析関数
# -----------------------------------------------------------------------------
def parse_cbc_log(log_filename):
    """
    CBCソルバーのログファイルを読み込み、(time, objective) のリストを返す。
    """
    history = []
    
    # 正規表現パターン: 
    # CBCのログ形式例: "Cbc0010I Integer solution of 12345 found after 500 iterations and 0 nodes (0.15 seconds)"
    pattern = re.compile(r"Integer solution of\s+([0-9\.\+\-eE]+)\s+found.*?\(\s*([0-9\.]+)\s+seconds\)")
    
    if not os.path.exists(log_filename):
        return history

    with open(log_filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                obj_val = float(match.group(1))
                time_val = float(match.group(2))
                history.append((time_val, obj_val))
    
    return history

# -----------------------------------------------------------------------------
# カスタム実行ロジック
# -----------------------------------------------------------------------------
def run_cg_and_log_mip_process(solver, mip_time_limit=60, mip_log_file="mip.log"):
    """
    1. 列生成(LP緩和)が収束するまで回す
    2. 最後にログ付きでMIPを解く
    3. 時間 vs 目的関数値 の履歴を返す
    """
    
    print(f"  Phase 1: Running Column Generation (LP Relaxation)...")
    
    # --- Step 1: LP緩和（列生成ループ）を手動で回す ---
    # solver.solve() の中身を分解して実行します
    solver.reset_stats()
    solver.initialize_rmp()
    
    prev_obj = float('inf')
    no_improve_iter = 0
    max_iter = 200 # 列生成自体の最大回数
    
    # LPループ
    lp_start = time.time()
    for i in range(max_iter):
        res = solver.solve_rmp(integer=False)
        if res is None: break
        obj, pi, sigma = res
        
        # 収束判定
        improvement = (prev_obj - obj) / abs(prev_obj + 1e-9)
        if improvement < 1e-4:
            no_improve_iter += 1
        else:
            no_improve_iter = 0
        prev_obj = obj
        
        pool_add, graph_add = solver.pricing(pi, sigma)
        
        if (pool_add + graph_add) == 0:
            break
        if no_improve_iter >= 3:
            break
            
    lp_time = time.time() - lp_start
    print(f"  -> CG Converged in {lp_time:.2f}s (Final LP Obj: {prev_obj:.2f}, Pool Size: {len(solver.pool)})")

    # --- Step 2: MIP構築とログ付き実行 ---
    print(f"  Phase 2: Solving MIP with logging (Limit: {mip_time_limit}s)...")
    
    # solve_rmp(integer=True) のロジックを再現しつつ、ログ出力を有効化
    model = pulp.LpProblem("RMP_Final_MIP", pulp.LpMinimize)
    active_cols = [solver.pool[i] for i in solver.rmp_indices]
    
    x = {c['id']: pulp.LpVariable(f"x_{c['id']}", 0, 1, pulp.LpBinary) for c in active_cols}
    delta = [pulp.LpVariable(f"d_{t}", 0) for t in range(solver.prob.T)]
    
    # 目的関数
    model += pulp.lpSum([c['cost']*x[c['id']] for c in active_cols]) + \
             pulp.lpSum([solver.prob.big_m * d for d in delta])
    
    # 制約
    for t in range(solver.prob.T):
        model += pulp.lpSum([c['schedule'][t]*x[c['id']] for c in active_cols]) + delta[t] >= solver.prob.demand[t]
        
    for k in range(solver.prob.K):
        model += pulp.lpSum([x[c['id']] for c in active_cols if c['group_id'] == k]) == 1

    # 既存のログファイルを削除
    if os.path.exists(mip_log_file):
        os.remove(mip_log_file)

    # CBCソルバーをログ出力モードで呼び出し
    # msg=1 で標準出力、logPath でファイル出力
    solver_cmd = pulp.PULP_CBC_CMD(
        msg=1, 
        timeLimit=mip_time_limit, 
        logPath=mip_log_file,
        gapRel=0.0 # 最適解が見つかるまで粘る設定
    )
    
    mip_start_timestamp = time.time()
    model.solve(solver_cmd)
    
    # ログファイルの解析
    history = parse_cbc_log(mip_log_file)
    
    # 最終結果もヒストリーに追加（タイムアウト時など、ログの最後より進んでいる可能性があるため）
    final_obj = pulp.value(model.objective)
    total_elapsed = time.time() - mip_start_timestamp
    
    if final_obj is not None:
        history.append((total_elapsed, final_obj))
    
    # 重複削除とソート
    history = sorted(list(set(history)), key=lambda x: x[0])
    
    return history, lp_time

# -----------------------------------------------------------------------------
# メイン実行部
# -----------------------------------------------------------------------------
def main():
    # パラメータ設定
    n_employees = 20    # 従業員数（難易度調整: 多いほどMIPが重くなる）
    mip_limit = 600      # MIPの計算時間上限(秒)
    week_num = 1        # テストする週
    
    print(f"=== MIP Convergence Benchmark (Employees: {n_employees}, Time Limit: {mip_limit}s) ===")
    
    # 問題データの作成
    prob = ShiftProblemData(n_employees=n_employees)
    prob.generate_new_demand(period=week_num-1)
    
    results = {}
    
    # 各手法を実行
    for name, config in SOLVER_MAPPING.items():
        print(f"\n[{name}] Initializing...")
        SolverClass = config['class']
        kwargs = config['kwargs']
        
        # ソルバーインスタンス化
        solver = SolverClass(prob, **kwargs)
        
        # 実行とログ取得
        log_file = f"temp_cbc_log_{name}.txt"
        history, lp_time = run_cg_and_log_mip_process(solver, mip_time_limit=mip_limit, mip_log_file=log_file)
        
        results[name] = {
            'history': history,
            'lp_time': lp_time
        }
        
        # 後始末
        if os.path.exists(log_file):
            os.remove(log_file)

    # -------------------------------------------------------------------------
    # プロット作成
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for name, data in results.items():
        hist = data['history']
        lp_time = data['lp_time']
        style = SOLVER_MAPPING[name]
        
        if not hist:
            print(f"Warning: No MIP history found for {name}")
            continue
            
        # x, y データの準備
        # ステッププロット用にデータを整形 (前の値を維持する階段状)
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        
        # グラフ描画
        # drawstyle='steps-post' で階段状にプロット（解が見つかった瞬間に値が下がる）
        plt.plot(xs, ys, label=f"{name} (LP Time: {lp_time:.1f}s)", 
                 color=style['color'], linestyle=style['linestyle'], 
                 drawstyle='steps-post', marker='o', markersize=4)

    plt.xlabel("MIP Calculation Time (seconds)")
    plt.ylabel("Objective Value (Cost)")
    plt.title(f"MIP Convergence Process (Employees={n_employees})")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    output_img = f"mip_convergence_{n_employees}emp.png"
    plt.savefig(output_img)
    print(f"\nPlot saved to: {output_img}")
    # plt.show() # ローカル環境ならコメントアウトを外して表示

if __name__ == "__main__":
    main()
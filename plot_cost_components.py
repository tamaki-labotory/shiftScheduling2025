import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from problem import ShiftProblemData

def load_problem_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return ShiftProblemData(config_path=config_path)

def calculate_cost_breakdown(prob, pool_csv_path, week):
    """
    CSVから採用されたスケジュールを読み込み、コストの内訳を計算する
    """
    prob.generate_new_demand(period=week - 1)
    
    if not os.path.exists(pool_csv_path):
        return None

    try:
        df = pd.read_csv(pool_csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read {pool_csv_path}: {e}")
        return None
    
    if 'is_selected' not in df.columns:
        print(f"  [Skip] 'is_selected' column missing in {os.path.basename(pool_csv_path)}.")
        return None
        
    selected_df = df[df['is_selected'] == 1]
    
    if selected_df.empty:
        return {'Wage': 0.0, 'Mismatch Cost': 0.0, 'Understaffing Penalty': 0.0, 'Total Obj': 0.0}

    total_base_wage = 0.0
    total_mismatch_cost = 0.0
    supplied = np.zeros(prob.T)
    
    for _, row in selected_df.iterrows():
        emp_id = int(row['emp_id'])
        sched_str = str(row['schedule_pattern'])
        schedule = np.array([int(c) for c in sched_str])
        
        emp = prob.employees[emp_id]
        base_wage = np.sum(schedule * emp['hourly_wage'])
        total_base_wage += base_wage
        
        rho_cost = np.sum(schedule * emp['rho'])
        total_mismatch_cost += rho_cost
        
        supplied += schedule

    shortage = np.maximum(0, prob.demand - supplied)
    total_penalty = np.sum(shortage) * prob.big_m
    
    return {
        'Base Wage': total_base_wage,
        'Mismatch Cost': total_mismatch_cost,
        'Understaffing Penalty': total_penalty,
        'Total Obj': total_base_wage + total_mismatch_cost + total_penalty
    }

def plot_grouped_stacked_bar(data_dict, output_dir):
    """
    X軸：週
    グループ：手法
    積み上げ：コスト内訳
    これらを1枚のグラフにプロットする
    """
    weeks = sorted(list(set(k[0] for k in data_dict.keys())))
    methods = sorted(list(set(k[1] for k in data_dict.keys())))
    
    if not weeks:
        print("No valid data available to plot.")
        return

    components = ['Base Wage', 'Mismatch Cost', 'Understaffing Penalty']
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] # 緑, オレンジ, 赤
    
    n_weeks = len(weeks)
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=(max(4, n_weeks * 0.6), 7))
    
    # ★変更1: バー全体の幅を広げて隙間を減らす (0.8 -> 0.9)
    total_width = 0.8
    bar_width = total_width / n_methods
    indices = np.arange(n_weeks)
    
    for i, method in enumerate(methods):
        # グループ内でのオフセット計算
        offset = (i - (n_methods - 1) / 2) * bar_width
        x_pos = indices + offset
        
        bottoms = np.zeros(n_weeks)
        
        for j, comp in enumerate(components):
            values = []
            for wk in weeks:
                val = data_dict.get((wk, method), {}).get(comp, 0.0)
                values.append(val)
            
            values = np.array(values)
            label = comp if i == 0 else ""
            
            ax.bar(x_pos, values, width=bar_width, bottom=bottoms, 
                   label=label, color=colors[j], alpha=0.85, edgecolor='black', linewidth=0.5)
            
            bottoms += values
        

    # ★変更2: X軸の表示範囲をバーの端に合わせて「謎のスペース」を消す
    # 左端: 0番目の中心 - (全体幅/2) - 余白少々
    # 右端: 最後の中心 + (全体幅/2) + 余白少々
    padding = 0.1
    left_limit = - (total_width / 2) - padding
    right_limit = (n_weeks - 1) + (total_width / 2) + padding
    ax.set_xlim(left_limit, right_limit)

    ax.set_ylabel("Total Cost")
    ax.set_title("Cost Component Breakdown by Week (Grouped by Method)")

    # ★変更箇所: X軸ラベルと目盛りの設定
    ax.set_xlabel("Week", fontsize=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(weeks, fontsize=12)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    unique_labels = list(by_label.keys())[::-1]
    unique_handles = list(by_label.values())[::-1]
    
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1), title="Cost Components")
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "cost_breakdown_timeline.png")
    plt.savefig(out_file)
    plt.close()
    print(f"  Saved timeline plot: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze cost components from result pools.")
    parser.add_argument("emp", type=str, help="Number of employees (e.g., 20)")
    parser.add_argument("methods", nargs='+', help="List of methods to include (e.g., std acc pruning)")
    parser.add_argument("--start_week", type=int, default=1)
    parser.add_argument("--end_week", type=int, default=5)
    
    args = parser.parse_args()
    
    base_dir = f"results_{args.emp}emp"
    config_path = os.path.join(base_dir, "problem_config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading problem config from: {config_path}")
    try:
        prob = load_problem_config(config_path)
    except Exception as e:
        print(f"Error loading problem config: {e}")
        return

    data_store = {}
    
    print("-" * 60)
    print(f"Processing Weeks {args.start_week} to {args.end_week} for methods: {args.methods}")
    
    for wk in range(args.start_week, args.end_week + 1):
        for method in args.methods:
            pool_file = os.path.join(base_dir, method, f"pool_wk{wk}.csv")
            
            if method == 'exact':
                continue
                
            if os.path.exists(pool_file):
                res = calculate_cost_breakdown(prob, pool_file, wk)
                if res:
                    data_store[(wk, method)] = res
            else:
                pass 

    if not data_store:
        print("No valid data found.")
        print("Please run main.py first to generate results.")
        return

    print("-" * 60)
    print("Generating Timeline Plot...")
    plot_grouped_stacked_bar(data_store, base_dir)
    print("Done.")

if __name__ == "__main__":
    main()
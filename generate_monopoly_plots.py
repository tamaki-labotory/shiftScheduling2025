import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# problem.py が同じディレクトリにあることを前提とします（コスト計算用）
try:
    from problem import ShiftProblemData
except ImportError:
    ShiftProblemData = None

# ==========================================
# データ解析関数
# ==========================================

def parse_metrics(filepath):
    """レポートファイルから指標を抽出"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'Objective_Value': 0.0,
        'RMP': 0.0,
        'MIP': 0.0,
        'Pool_Search': 0.0,
        'Solving_Shortest_Path_Problem': 0.0,
        'Total_Time': 0.0,
        'Gap_Absolute': None,
        'Pool_Hit_Rate': None,
        'Iterations': 0.0,
        'Stagnant_Iterations': 0.0
    }

    # 基本指標の抽出
    patterns = {
        'Objective_Value': r"Objective Value\s*:\s*([\d\.,]+)",
        'RMP': r"RMP Time\s*:\s*([\d\.]+)",
        'MIP': r"MIP Time\s*:\s*([\d\.]+)",
        'Pool_Search': r"Pool Search Time\s*:\s*([\d\.]+)",
        'Solving_Shortest_Path_Problem': r"Graph Search Time\s*:\s*([\d\.]+)",
        'Total_Time': r"Execution Time\s*:\s*([\d\.]+)",
        'Pool_Hit_Rate': r"Pool Hit Rate\s*:\s*([\d\.]+)%",
        'Iterations': r"Iterations\s*:\s*([\d]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val_str = match.group(1).replace(',', '')
            metrics[key] = float(val_str)

    # Total_Time補完 (もしログにTotal Timeがなければ合計値を使う)
    calculated_total = (metrics['RMP'] + metrics['MIP'] + 
                        metrics['Pool_Search'] + metrics['Solving_Shortest_Path_Problem'])
    
    if metrics['Total_Time'] == 0.0:
        metrics['Total_Time'] = calculated_total

    # 絶対ギャップ (Objective - LowerBound)
    lower_bound = None
    match_rmp_lb = re.search(r"RMP Relaxed Value \(.*?\)\s*:\s*([\d\.,]+)", content)
    if match_rmp_lb:
        lower_bound = float(match_rmp_lb.group(1).replace(',', ''))
            
    if lower_bound is not None and metrics['Objective_Value'] > 0:
        metrics['Gap_Absolute'] = metrics['Objective_Value'] - lower_bound

    # 停滞回数 (Stagnation)
    obj_values = []
    lines = content.splitlines()
    in_history_section = False
    
    for line in lines:
        if "Iteration History" in line:
            in_history_section = True
            continue
        if not in_history_section:
            continue
        
        stripped = line.strip()
        if not stripped: continue
        if "Iter" in line or "-----" in line: continue
        
        parts = line.split('|')
        if len(parts) >= 2:
            try:
                val_str = parts[1].strip().replace(',', '')
                val = float(val_str)
                obj_values.append(val)
            except ValueError:
                continue

    stagnant_count = 0
    for i in range(len(obj_values) - 1):
        if abs(obj_values[i] - obj_values[i+1]) < 1e-9:
            stagnant_count += 1
            
    metrics['Stagnant_Iterations'] = float(stagnant_count)

    return metrics

def load_data_from_directory(target_dir):
    """指定ディレクトリ内のレポートファイルを読み込む"""
    pattern = os.path.join(target_dir, "report_wk*.txt")
    files = glob.glob(pattern)
    
    data_list = []
    for filepath in files:
        match = re.search(r"report_wk(\d+)\.txt", os.path.basename(filepath))
        if match:
            wk = int(match.group(1))
            metrics = parse_metrics(filepath)
            if metrics:
                metrics['Week'] = wk
                data_list.append(metrics)
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    return df.sort_values('Week')

# ==========================================
# コスト計算関数
# ==========================================

def calculate_cost_breakdown(prob, pool_csv_path, week):
    """コスト内訳を計算"""
    prob.generate_new_demand(period=week - 1)
    
    if not os.path.exists(pool_csv_path):
        return None

    try:
        df = pd.read_csv(pool_csv_path)
    except Exception:
        return None
    
    if 'is_selected' not in df.columns:
        return None
        
    selected_df = df[df['is_selected'] == 1]
    
    if selected_df.empty:
        return {'Base Wage': 0.0, 'Mismatch Cost': 0.0, 'Understaffing Penalty': 0.0}

    total_base_wage = 0.0
    total_mismatch_cost = 0.0
    supplied = np.zeros(prob.T)
    
    for _, row in selected_df.iterrows():
        emp_id = int(row['emp_id'])
        sched_str = str(row['schedule_pattern'])
        schedule = np.array([int(c) for c in sched_str])
        
        emp = prob.employees[emp_id]
        total_base_wage += np.sum(schedule * emp['hourly_wage'])
        total_mismatch_cost += np.sum(schedule * emp['rho'])
        supplied += schedule

    shortage = np.maximum(0, prob.demand - supplied)
    total_penalty = np.sum(shortage) * prob.big_m
    
    return {
        'Base Wage': total_base_wage,
        'Mismatch Cost': total_mismatch_cost,
        'Understaffing Penalty': total_penalty
    }

# ==========================================
# グラフ描画関数
# ==========================================

def plot_time_stacked(df, output_dir):
    """実行時間の内訳を積み上げ棒グラフで描画"""
    save_dir = os.path.join(output_dir, "plots_metrics")
    os.makedirs(save_dir, exist_ok=True)
    
    weeks = df['Week'].values
    
    # 積み上げる項目と表示ラベルの定義
    components_map = {
        'RMP': 'RMP Solving',
        'Solving_Shortest_Path_Problem': 'Shortest Path Search',
        'Pool_Search': 'Pool Search',
        'MIP': 'MIP'
    }
    
    # Othersの計算 (Total - 上記4つの合計)
    # マイナスにならないようにクリップする
    known_sum = df['RMP'] + df['Solving_Shortest_Path_Problem'] + df['Pool_Search'] + df['MIP']
    others = df['Total_Time'] - known_sum
    others = others.apply(lambda x: max(0, x))
    
    # データフレームにOthersを追加（描画用）
    plot_df = df.copy()
    plot_df['Others'] = others
    
    # 描画順序リスト
    stack_order = ['RMP', 'Solving_Shortest_Path_Problem', 'Pool_Search', 'MIP', 'Others']
    labels = [components_map.get(k, k) for k in stack_order]
    
    # 色の定義 (Tableau 10 like)
    colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#BAB0AC']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(weeks))
    
    for key, label, color in zip(stack_order, labels, colors):
        if key not in plot_df.columns:
            continue
        
        values = plot_df[key].values
        ax.bar(weeks, values, bottom=bottom, label=label, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_title("Execution Time Breakdown", fontsize=16)
    ax.set_xlabel("Week", fontsize=14)
    ax.set_ylabel("Time (s)", fontsize=14)
    ax.set_xticks(weeks)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, "time_breakdown.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved Time Breakdown: {output_path}")

def plot_other_metrics(df, output_dir):
    """時間以外の指標の推移グラフを描画"""
    metrics_keys = [
        'Objective_Value', 'Gap_Absolute', 'Pool_Hit_Rate', 'Iterations'
    ]
    
    metric_titles = {
        'Objective_Value': 'Objective Value',
        'Gap_Absolute': 'Optimality Gap',
        'Pool_Hit_Rate': 'Pool Hit Rate (%)',
        'Iterations': 'Number of Iterations'
    }

    save_dir = os.path.join(output_dir, "plots_metrics")
    os.makedirs(save_dir, exist_ok=True)
    
    weeks = df['Week'].values

    for metric in metrics_keys:
        if metric not in df.columns: continue
        
        # データがすべて0の場合はスキップ（Gapなどはありえるので条件付き）
        if df[metric].sum() == 0 and metric != 'Gap_Absolute':
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        if metric == 'Iterations':
            effective = df['Iterations'] - df['Stagnant_Iterations']
            stagnant = df['Stagnant_Iterations']
            
            ax.bar(weeks, effective, label='Effective Iterations', color='#4E79A7', alpha=0.9)
            ax.bar(weeks, stagnant, bottom=effective, label='Stagnant Iterations', 
                   color='#4E79A7', alpha=0.4, hatch='///', edgecolor='black', linewidth=0.0)
            ax.legend()
        else:
            ax.bar(weeks, df[metric], color='#4E79A7', alpha=0.7, label=metric)
            ax.plot(weeks, df[metric], color='#2B4F76', marker='o', linestyle='-', linewidth=1.5)

        ax.set_title(metric_titles.get(metric, metric), fontsize=14)
        ax.set_xlabel("Week", fontsize=12)
        ax.set_xticks(weeks)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        output_path = os.path.join(save_dir, f"{metric}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def plot_cost_breakdown(cost_data_list, output_dir):
    """コスト内訳の積み上げ棒グラフ"""
    if not cost_data_list: return

    save_dir = os.path.join(output_dir, "plots_cost")
    os.makedirs(save_dir, exist_ok=True)
    
    weeks = [d['Week'] for d in cost_data_list]
    components = ['Base Wage', 'Mismatch Cost', 'Understaffing Penalty']
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.arange(len(weeks))
    bottoms = np.zeros(len(weeks))
    
    for i, comp in enumerate(components):
        vals = np.array([d[comp] for d in cost_data_list])
        ax.bar(indices, vals, bottom=bottoms, label=comp, color=colors[i], alpha=0.85, edgecolor='white')
        bottoms += vals
        
    ax.set_title("Cost Breakdown per Week", fontsize=16)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(weeks)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cost_breakdown.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# メイン処理（GUI）
# ==========================================

def main():
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す

    # スクリプト自身のディレクトリを取得
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    print("ディレクトリ選択ダイアログを開いています...")
    # initialdir を指定してダイアログを開く
    target_dir = filedialog.askdirectory(
        title="Select Experiment Result Directory",
        initialdir=current_script_dir
    )

    if not target_dir:
        print("キャンセルされました。")
        return

    print(f"Selected Directory: {target_dir}")
    
    # 1. データの読み込み
    df = load_data_from_directory(target_dir)
    
    output_base = os.path.join(target_dir, "analysis_results")
    os.makedirs(output_base, exist_ok=True)

    if not df.empty:
        print(f"Found data for {len(df)} weeks.")
        
        # 1-A. 時間の積み上げグラフ
        plot_time_stacked(df, output_base)
        
        # 1-B. その他の指標グラフ
        plot_other_metrics(df, output_base)
        
        # CSVとしても保存
        df.to_csv(os.path.join(output_base, "summary_metrics.csv"), index=False)
    else:
        messagebox.showwarning("No Data", "指定されたディレクトリ内に 'report_wk*.txt' が見つかりませんでした。")
        return

    # 2. コスト内訳の計算とプロット
    if ShiftProblemData is None:
        print("ShiftProblemData not found. Skipping cost breakdown.")
    else:
        config_path = os.path.join(target_dir, "problem_config.json")
        if not os.path.exists(config_path):
             config_path = os.path.join(os.path.dirname(target_dir), "problem_config.json")
        
        if os.path.exists(config_path):
            print("Generating cost breakdown plots...")
            try:
                prob = ShiftProblemData(config_path=config_path)
                
                cost_list = []
                for wk in df['Week']:
                    pool_file = os.path.join(target_dir, f"pool_wk{wk}.csv")
                    res = calculate_cost_breakdown(prob, pool_file, wk)
                    if res:
                        res['Week'] = wk
                        cost_list.append(res)
                
                plot_cost_breakdown(cost_list, output_base)
            except Exception as e:
                print(f"Error during cost calculation: {e}")
        else:
            print("problem_config.json not found. Skipping cost breakdown.")

    print(f"完了しました。結果は以下に保存されています:\n{output_base}")
    messagebox.showinfo("Complete", f"処理が完了しました。\n保存先: {output_base}")

if __name__ == "__main__":
    main()
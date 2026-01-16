import os
import re

# ==========================================================
# 設定: ディレクトリ名と手法名のマッピング
# ==========================================================
BASE_DIR = "results_5emp"

# {LaTeXの列名: 実際のフォルダ名}
METHOD_FOLDERS = {
    "Std": "std(100)",         
    "Acc": "acc(100)",
    "Pruning": "pruning(100)"
}

# 厳密解のフォルダ名
EXACT_FOLDER = "exact"

# 解析対象の週
WEEKS = range(1, 16) 

# ==========================================================
# パーサー関数
# ==========================================================
def parse_report(filepath):
    data = {
        "obj_val": None,
        "gap": None,
        "exec_time": None,
        "iterations": None,
        "hit_rate": None,
        "new_cols": None,
        "pool_search_time": None,
        "graph_search_time": None,
        "rmp_time": None,
        "mip_time": None,
        "mip_vars": None
    }

    if not os.path.exists(filepath):
        return data

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        patterns = {
            "obj_val": r"Objective Value\s*:\s*([0-9,.]+)",
            "gap": r"Integrality Gap\s*:\s*([0-9,.]+)", 
            "exec_time": r"Execution Time\s*:\s*([0-9,.]+)",
            "iterations": r"Iterations\s*:\s*([0-9,]+)",
            "hit_rate": r"Pool Hit Rate\s*:\s*([0-9,.]+)",
            "new_cols": r"Graph Gen \(Newly Created\)\s*:\s*([0-9,]+)",
            "pool_search_time": r"Pool Search Time\s*:\s*([0-9,.]+)",
            "graph_search_time": r"Graph Search Time\s*:\s*([0-9,.]+)",
            "rmp_time": r"RMP Time\s*:\s*([0-9,.]+)",
            "mip_time": r"MIP Time\s*:\s*([0-9,.]+)",
            "mip_vars": r"MIP Decision Variables\s*:\s*([0-9,]+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                val_str = match.group(1).replace(',', '')
                try:
                    if key in ["iterations", "mip_vars", "new_cols"]:
                        data[key] = int(float(val_str))
                    else:
                        data[key] = float(val_str)
                except ValueError:
                    pass
    return data

# ==========================================================
# フォーマット関数
# ==========================================================
def fmt_int(val):
    if val is None: return ""
    return f"{int(val):,}"

def fmt_float(val, precision=2):
    if val is None: return ""
    return f"{val:,.{precision}f}"

def get_week_data(week):
    row_data = {}
    for method, folder in METHOD_FOLDERS.items():
        filepath = os.path.join(BASE_DIR, folder, f"report_wk{week}.txt")
        row_data[method] = parse_report(filepath)
    exact_path = os.path.join(BASE_DIR, EXACT_FOLDER, f"report_wk{week}.txt")
    row_data["Exact"] = parse_report(exact_path)
    return row_data

# ==========================================================
# LaTeX生成関数
# ==========================================================

def generate_table1():
    print("% =========================================================")
    print("% Table 1: Overall Performance (Time, Obj, Ratio, Gap)")
    print("% Items: 4 metrics (Balanced)")
    print("% =========================================================")
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Overall Performance: Execution Time, Objective Value, Ratio, and Gap (N=5)}")
    print(r"    \label{tab:overall_balanced}")
    print(r"    \resizebox{\textwidth}{!}{%")
    print(r"    \begin{tabular}{c|ccc|ccc|ccc|ccc}")
    print(r"        \toprule")
    print(r"        \multirow{2}{*}{\textbf{Wk}} & \multicolumn{3}{c|}{\textbf{Exec Time (s)}} & \multicolumn{3}{c|}{\textbf{Objective Value}} & \multicolumn{3}{c|}{\textbf{Ratio to Opt.}} & \multicolumn{3}{c}{\textbf{Gap (\%)}} \\")
    print(r"         & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru \\")
    print(r"        \midrule\midrule")

    sums = {k: {"time": 0, "obj": 0, "ratio": 0, "gap": 0, "count": 0} for k in METHOD_FOLDERS.keys()}

    for wk in WEEKS:
        d = get_week_data(wk)
        exact_obj = d["Exact"]["obj_val"]
        
        times = []
        objs = []
        ratios = []
        gaps = []

        for m in ["Std", "Acc", "Pruning"]:
            dm = d[m]
            
            # Ratio calculation
            if dm["obj_val"] is not None and exact_obj is not None and exact_obj != 0:
                ratio_val = dm["obj_val"] / exact_obj
                ratios.append(fmt_float(ratio_val, 2))
                sums[m]["ratio"] += ratio_val
            else:
                ratios.append("-")

            times.append(fmt_float(dm["exec_time"]))
            objs.append(fmt_int(dm["obj_val"]))
            gaps.append(fmt_float(dm["gap"]))

            if dm["exec_time"] is not None:
                sums[m]["time"] += dm["exec_time"]
                sums[m]["obj"] += dm["obj_val"] if dm["obj_val"] else 0
                sums[m]["gap"] += dm["gap"] if dm["gap"] else 0
                sums[m]["count"] += 1

        print(f"        {wk} & {' & '.join(times)} & {' & '.join(objs)} & {' & '.join(ratios)} & {' & '.join(gaps)} \\\\")

    # Averages
    avg_times = []
    avg_objs = []
    avg_ratios = []
    avg_gaps = []

    for m in ["Std", "Acc", "Pruning"]:
        c = sums[m]["count"]
        if c > 0:
            avg_times.append(fmt_float(sums[m]["time"] / c, 2))
            avg_objs.append(fmt_int(sums[m]["obj"] / c))
            avg_ratios.append(fmt_float(sums[m]["ratio"] / c, 2))
            avg_gaps.append(fmt_float(sums[m]["gap"] / c, 2))
        else:
            avg_times.append("-")
            avg_objs.append("-")
            avg_ratios.append("-")
            avg_gaps.append("-")

    print(r"        \midrule")
    print(f"        \\textbf{{Avg}} & {' & '.join(avg_times)} & {' & '.join(avg_objs)} & {' & '.join(avg_ratios)} & {' & '.join(avg_gaps)} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}%")
    print(r"    }")
    print(r"\end{table}")
    print("\n")

def generate_table2():
    print("% =========================================================")
    print("% Table 2: Time Breakdown (RMP, MIP, Graph Search, Pool Search)")
    print("% Items: 4 metrics (Balanced)")
    print("% =========================================================")
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Detailed Time Breakdown: RMP, MIP, Graph Search, and Pool Search (N=5)}")
    print(r"    \label{tab:time_breakdown_balanced}")
    print(r"    \resizebox{\textwidth}{!}{%")
    print(r"    \begin{tabular}{c|ccc|ccc|ccc|ccc}")
    print(r"        \toprule")
    print(r"        \multirow{2}{*}{\textbf{Wk}} & \multicolumn{3}{c|}{\textbf{RMP Time (s)}} & \multicolumn{3}{c|}{\textbf{MIP Time (s)}} & \multicolumn{3}{c|}{\textbf{Graph Search (s)}} & \multicolumn{3}{c}{\textbf{Pool Search (s)}} \\")
    print(r"         & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru \\")
    print(r"        \midrule\midrule")

    sums = {k: {"rmp": 0, "mip": 0, "gs": 0, "ps": 0, "count": 0} for k in METHOD_FOLDERS.keys()}

    for wk in WEEKS:
        d = get_week_data(wk)
        
        rmps = []
        mips = []
        gs_times = []
        ps_times = []

        for m in ["Std", "Acc", "Pruning"]:
            dm = d[m]
            rmps.append(fmt_float(dm["rmp_time"]))
            mips.append(fmt_float(dm["mip_time"]))
            gs_times.append(fmt_float(dm["graph_search_time"]))
            ps_times.append(fmt_float(dm["pool_search_time"]))

            if dm["exec_time"] is not None:
                sums[m]["rmp"] += dm["rmp_time"] if dm["rmp_time"] else 0
                sums[m]["mip"] += dm["mip_time"] if dm["mip_time"] else 0
                sums[m]["gs"] += dm["graph_search_time"] if dm["graph_search_time"] else 0
                sums[m]["ps"] += dm["pool_search_time"] if dm["pool_search_time"] else 0
                sums[m]["count"] += 1

        print(f"        {wk} & {' & '.join(rmps)} & {' & '.join(mips)} & {' & '.join(gs_times)} & {' & '.join(ps_times)} \\\\")

    # Averages
    avg_rmp = []
    avg_mip = []
    avg_gs = []
    avg_ps = []

    for m in ["Std", "Acc", "Pruning"]:
        c = sums[m]["count"]
        if c > 0:
            avg_rmp.append(fmt_float(sums[m]["rmp"] / c, 2))
            avg_mip.append(fmt_float(sums[m]["mip"] / c, 2))
            avg_gs.append(fmt_float(sums[m]["gs"] / c, 2))
            avg_ps.append(fmt_float(sums[m]["ps"] / c, 2))
        else:
            avg_rmp.append("-")
            avg_mip.append("-")
            avg_gs.append("-")
            avg_ps.append("-")

    print(r"        \midrule")
    print(f"        \\textbf{{Avg}} & {' & '.join(avg_rmp)} & {' & '.join(avg_mip)} & {' & '.join(avg_gs)} & {' & '.join(avg_ps)} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}%")
    print(r"    }")
    print(r"\end{table}")
    print("\n")

def generate_table3():
    print("% =========================================================")
    print("% Table 3: Algorithm Stats (Iterations, Hit Rate, New Cols, MIP Vars)")
    print("% Items: 4 metrics (Balanced)")
    print("% =========================================================")
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \caption{Algorithm Statistics: Iterations, Pool Hit Rate, Generated Columns, and MIP Variables (N=5)}")
    print(r"    \label{tab:stats_balanced}")
    print(r"    \resizebox{\textwidth}{!}{%")
    print(r"    \begin{tabular}{c|ccc|ccc|ccc|ccc}")
    print(r"        \toprule")
    print(r"        \multirow{2}{*}{\textbf{Wk}} & \multicolumn{3}{c|}{\textbf{Iterations}} & \multicolumn{3}{c|}{\textbf{Pool Hit Rate (\%)}} & \multicolumn{3}{c|}{\textbf{New Columns}} & \multicolumn{3}{c}{\textbf{MIP Decision Vars}} \\")
    print(r"         & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru & Std & Acc & Pru \\")
    print(r"        \midrule\midrule")

    sums = {k: {"iter": 0, "hit": 0, "col": 0, "vars": 0, "count": 0} for k in METHOD_FOLDERS.keys()}

    for wk in WEEKS:
        d = get_week_data(wk)
        
        iters = []
        hits = []
        cols = []
        vars_ = []

        for m in ["Std", "Acc", "Pruning"]:
            dm = d[m]
            iters.append(fmt_int(dm["iterations"]))
            hits.append(fmt_float(dm["hit_rate"], 1))
            cols.append(fmt_int(dm["new_cols"]))
            vars_.append(fmt_int(dm["mip_vars"]))

            if dm["exec_time"] is not None:
                sums[m]["iter"] += dm["iterations"] if dm["iterations"] else 0
                sums[m]["hit"] += dm["hit_rate"] if dm["hit_rate"] else 0
                sums[m]["col"] += dm["new_cols"] if dm["new_cols"] else 0
                sums[m]["vars"] += dm["mip_vars"] if dm["mip_vars"] else 0
                sums[m]["count"] += 1

        print(f"        {wk} & {' & '.join(iters)} & {' & '.join(hits)} & {' & '.join(cols)} & {' & '.join(vars_)} \\\\")

    # Averages
    avg_iters = []
    avg_hits = []
    avg_cols = []
    avg_vars = []

    for m in ["Std", "Acc", "Pruning"]:
        c = sums[m]["count"]
        if c > 0:
            avg_iters.append(fmt_float(sums[m]["iter"] / c, 1))
            avg_hits.append(fmt_float(sums[m]["hit"] / c, 1))
            avg_cols.append(fmt_int(sums[m]["col"] / c))
            avg_vars.append(fmt_int(sums[m]["vars"] / c))
        else:
            avg_iters.append("-")
            avg_hits.append("-")
            avg_cols.append("-")
            avg_vars.append("-")

    print(r"        \midrule")
    print(f"        \\textbf{{Avg}} & {' & '.join(avg_iters)} & {' & '.join(avg_hits)} & {' & '.join(avg_cols)} & {' & '.join(avg_vars)} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}%")
    print(r"    }")
    print(r"\end{table}")
    print("\n")

if __name__ == "__main__":
    generate_table1()
    generate_table2()
    generate_table3()
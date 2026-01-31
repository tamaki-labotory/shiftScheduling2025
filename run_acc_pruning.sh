#!/bin/bash

# ==========================================
# 設定
# ==========================================
EMP_NUM=5
DEFAULT_OUT_DIR="results_${EMP_NUM}emp"
CONFIG_BACKUP="master_problem_config.json"

# 実行するメソッドのリスト
target_methods=("acc" "pruning")

echo "=== 実験準備: マスタ設定ファイルの確認 ==="

# master_problem_config.json が存在するか確認
if [ ! -f "$CONFIG_BACKUP" ]; then
    echo "Error: $CONFIG_BACKUP not found! Run std experiment first."
    exit 1
fi
echo "Using existing master config: $CONFIG_BACKUP"

# 念のため一時フォルダが残っていたら削除（中断した残骸の可能性があるため）
if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi


# ==========================================
# 実験1: 列生成過程の打ち切り基準 (Patience)
# ==========================================
echo ""
echo "=== Starting Experiment 1 (ACC & Pruning): CG Termination Criteria ==="

patience_values=(5 10 15 100)

for p in "${patience_values[@]}"; do
    for method in "${target_methods[@]}"; do
        
        # 最終的な保存先フォルダ名
        FINAL_DIR="results_exp1_pat_${p}_${method}"
        
        # ★スキップ判定: すでに結果フォルダがある場合は何もしない
        if [ -d "$FINAL_DIR" ]; then
            echo "[SKIP] $FINAL_DIR already exists."
            continue
        fi

        echo "Running Exp1 ($method) with Patience = $p ..."
        
        # 1. 掃除 & 準備
        if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi
        mkdir -p "$DEFAULT_OUT_DIR"
        cp "$CONFIG_BACKUP" "$DEFAULT_OUT_DIR/problem_config.json"
        
        # 2. 実行 (-u でログをリアルタイム表示)
        python -u main.py \
            --employees $EMP_NUM \
            --methods $method \
            --weeks 0 \
            --patience $p \
            --mip_rc_threshold inf \
            --mip_gap 0.01 \
            --time_limit 3600
            
        # 3. 結果保存
        mv "$DEFAULT_OUT_DIR" "$FINAL_DIR"
        
        echo "Finished Exp1 ($method, Patience=$p). Saved to $FINAL_DIR"
        sleep 5
    done
done


# # ==========================================
# # 実験2: IP構築時の列選定基準 (Reduced Cost Threshold)
# # ==========================================
# echo ""
# echo "=== Starting Experiment 2 (ACC & Pruning): MIP Column Selection ==="

# rc_values=(1 1000 100000 inf)

# for rc in "${rc_values[@]}"; do
#     for method in "${target_methods[@]}"; do
        
#         # 最終的な保存先フォルダ名
#         FINAL_DIR="results_exp2_rc_${rc}_${method}"

#         # ★スキップ判定
#         if [ -d "$FINAL_DIR" ]; then
#             echo "[SKIP] $FINAL_DIR already exists."
#             continue
#         fi

#         echo "Running Exp2 ($method) with RC Threshold = $rc ..."
        
#         # 1. 掃除 & 準備
#         if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi
#         mkdir -p "$DEFAULT_OUT_DIR"
#         cp "$CONFIG_BACKUP" "$DEFAULT_OUT_DIR/problem_config.json"
        
#         # 2. 実行
#         # ★重要: mip_gap を 0.01 に緩和して、現実的な時間で終わるように修正
#         python -u main.py \
#             --employees $EMP_NUM \
#             --methods $method \
#             --weeks 0 \
#             --patience 100 \
#             --mip_rc_threshold $rc \
#             --mip_gap 0.01 \
#             --time_limit 3600

#         # 3. 結果保存
#         mv "$DEFAULT_OUT_DIR" "$FINAL_DIR"
        
#         echo "Finished Exp2 ($method, RC=$rc). Saved to $FINAL_DIR"
        
#         sleep 5
#     done
    
#     # パラメータ区切りの休憩
#     sleep 10
# done

echo ""
echo "=== All ACC & Pruning Experiments Completed ==="
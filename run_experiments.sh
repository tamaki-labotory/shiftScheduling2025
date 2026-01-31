#!/bin/bash

# ==========================================
# 設定
# ==========================================
EMP_NUM=5  
# マスタ作成用の週数
INIT_WEEKS=15

# ★修正: exact をリストに追加
BASE_ARGS="--employees $EMP_NUM --methods exact std acc pruning"

# 結果が出力されるデフォルトのディレクトリ名
DEFAULT_OUT_DIR="results_${EMP_NUM}emp"

# 設定ファイルを保存する名前
CONFIG_BACKUP="master_problem_config.json"

# ==========================================
# 0. マスタ設定ファイルの作成
# ==========================================
echo "=== 準備: マスタ設定ファイルの作成 ==="

if [ -d "$DEFAULT_OUT_DIR" ]; then
    rm -rf "$DEFAULT_OUT_DIR"
fi

# 15週分のシナリオを生成・保存 (exactは重いのでマスタ生成時はstdのみなど軽量にする手もあるが、ここでは統一)
# 時間がかかりすぎるのを防ぐため、マスタ生成時は std のみを使用する
echo "Generating master config with $EMP_NUM employees for $INIT_WEEKS weeks..."
python main.py --employees $EMP_NUM --methods std --weeks $INIT_WEEKS --patience 100 --mip_rc_threshold inf --mip_gap 0.0 --time_limit 600

# 設定ファイルをバックアップ
mv "$DEFAULT_OUT_DIR/problem_config.json" "./$CONFIG_BACKUP"
echo "Master config saved to $CONFIG_BACKUP"

rm -rf "$DEFAULT_OUT_DIR"

# ==========================================
# ★追加: ウォームアップ（Cold Start対策）
# ==========================================
echo ""
echo "=== WARM-UP: Priming OS Cache for Solver ==="
echo "Running a dummy simulation to load 'cbc' binary into RAM..."

python main.py $BASE_ARGS --weeks 1 --time_limit 10 --mip_gap 0.5 > /dev/null 2>&1

if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi

echo "Warm-up complete. Starting actual experiments..."
sleep 2

# ==========================================
# 実験1: 列生成過程の打ち切り基準 (Patience)
# ==========================================
echo ""
echo "=== Starting Experiment 1: CG Termination Criteria ==="

patience_values=(5 10 15 100)

for p in "${patience_values[@]}"; do
    echo "Running Exp1 with Patience = $p ..."
    
    if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi
    mkdir -p "$DEFAULT_OUT_DIR"
    
    cp "$CONFIG_BACKUP" "$DEFAULT_OUT_DIR/problem_config.json"
    
    python main.py $BASE_ARGS \
        --weeks 0 \
        --patience $p \
        --mip_rc_threshold inf \
        --mip_gap 0.01 \
        --time_limit 3600
        
    mv "$DEFAULT_OUT_DIR" "results_exp1_pat_${p}"
    
    echo "Finished Exp1 (Patience=$p). Saved to results_exp1_pat_${p}"
    sleep 2
done

# ==========================================
# 実験2: IP構築時の列選定基準 (Reduced Cost Threshold)
# ==========================================
echo ""
echo "=== Starting Experiment 2: MIP Column Selection ==="

rc_values=(1 1000 100000 10000000 inf)

for rc in "${rc_values[@]}"; do
    echo "Running Exp2 with RC Threshold = $rc ..."
    
    if [ -d "$DEFAULT_OUT_DIR" ]; then rm -rf "$DEFAULT_OUT_DIR"; fi
    mkdir -p "$DEFAULT_OUT_DIR"
    
    cp "$CONFIG_BACKUP" "$DEFAULT_OUT_DIR/problem_config.json"
    
    python main.py $BASE_ARGS \
        --weeks 0 \
        --patience 100 \
        --mip_rc_threshold $rc \
        --mip_gap 0.0 \
        --time_limit 3600

    mv "$DEFAULT_OUT_DIR" "results_exp2_rc_${rc}"
    
    echo "Finished Exp2 (RC=$rc). Saved to results_exp2_rc_${rc}"
    sleep 2
done

echo ""
echo "=== All Experiments Completed ==="
#!/bin/bash

# ==========================================
# 実験設定: 20名, 15週間, 3手法比較
# ==========================================
# 従業員数
EMP_NUM=20
# シミュレーション週数
WEEKS=15
# 比較手法 (Exact, Neighbor, Pruning)
# METHODS="std"
# METHODS="exact pruning neighbor"
METHODS="pruning neighbor"
# 出力ディレクトリ
OUT_DIR="results_comparison_20emp"

# 実験パラメータ
# patience: 改善が停滞した場合の打ち切り回数 (実験設定: 3)
# time_limit: 全体の制限時間 (実験設定: 3600秒)
# rc_threshold: MIP構築時の列選定用 (Neighbor等は内部ロジック優先だが念のため指定)
PATIENCE=3
TIME_LIMIT=3600

echo "=========================================="
echo " Starting Benchmark Experiment"
echo " Employees : $EMP_NUM"
echo " Weeks     : $WEEKS"
echo " Methods   : $METHODS"
echo " Patience  : $PATIENCE"
echo "=========================================="



# Pythonスクリプトの実行 (simpleモード)
python main.py simple \
    --employees $EMP_NUM \
    --weeks $WEEKS \
    --methods $METHODS \
    --patience $PATIENCE \
    --rc_threshold 1e10 \
    --time_limit $TIME_LIMIT

# 結果ディレクトリの場所を移動（main.pyの出力先が固定の場合の対応）
# main.pyの simple モードは "results_{EMP}emp" に出力するため、リネームする
DEFAULT_OUT="results_${EMP_NUM}emp"

if [ -d "$DEFAULT_OUT" ]; then
    echo "Renaming output directory to $OUT_DIR..."
    mv "$DEFAULT_OUT" "$OUT_DIR"
fi

echo ""
echo "Experiment Completed."
echo "Results are saved in: $OUT_DIR"
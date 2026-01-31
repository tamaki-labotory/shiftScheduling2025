#!/bin/bash

# ==========================================
# Final Benchmark Script (Safety First)
# Employees: 20, Weeks: 15
# Order: Safe Methods (All Gaps) -> Exact (All Gaps)
# ==========================================

# --- Configuration ---
EMP_NUM=20
WEEKS=15
TIME_LIMIT=3600
PATIENCE=3
TOL=1e-8
MIP_RC_THRESHOLD=100000

# Group 1: Safe Methods (Won't crash)
SAFE_METHODS=("std" "acc" "pruning" "neighbor")

# Group 2: Risky Method (Might crash)
EXACT_METHOD=("exact")

# Output Directories
OUT_DIR_5="results_nightly_gap5pct_${EMP_NUM}emp"
OUT_DIR_0="results_nightly_gap0pct_${EMP_NUM}emp"
TEMP_DIR="results_${EMP_NUM}emp"

echo "=========================================================="
echo "STARTING SAFETY-PRIORITY BENCHMARK"
echo "Employees: $EMP_NUM"
echo "Weeks    : $WEEKS"
echo "Strategy : Run all SAFE methods first, then EXACT last."
echo "=========================================================="
echo ""

# Function to clean temp dir
clean_temp() {
    if [ -d "$TEMP_DIR" ]; then rm -rf "$TEMP_DIR"; fi
}

# ==========================================================
# PART 1: Safe Methods (Gap 5%)
# ==========================================================
MIP_GAP=0.05
TARGET_DIR=$OUT_DIR_5

echo ">>> [1/4] Safe Methods | Gap 5%"
if [ -d "$TARGET_DIR" ]; then rm -rf "$TARGET_DIR"; fi
mkdir -p "$TARGET_DIR"
clean_temp

for method in "${SAFE_METHODS[@]}"; do
    echo "--- Running $method (Gap 5%) ---"
    python -u main.py --employees $EMP_NUM --weeks $WEEKS --methods $method \
        --time_limit $TIME_LIMIT --patience $PATIENCE --tol $TOL \
        --mip_rc_threshold $MIP_RC_THRESHOLD --mip_gap $MIP_GAP
    
    if [ -d "$TEMP_DIR/$method" ]; then
        mv "$TEMP_DIR/$method" "$TARGET_DIR/"
        cp "$TEMP_DIR/problem_config.json" "$TARGET_DIR/" 2>/dev/null
    fi
    clean_temp
    sleep 2
done

# ==========================================================
# PART 2: Safe Methods (Gap 0%)
# ==========================================================
MIP_GAP=0.0
TARGET_DIR=$OUT_DIR_0

echo ""
echo ">>> [2/4] Safe Methods | Gap 0%"
if [ -d "$TARGET_DIR" ]; then rm -rf "$TARGET_DIR"; fi
mkdir -p "$TARGET_DIR"
clean_temp

for method in "${SAFE_METHODS[@]}"; do
    echo "--- Running $method (Gap 0%) ---"
    python -u main.py --employees $EMP_NUM --weeks $WEEKS --methods $method \
        --time_limit $TIME_LIMIT --patience $PATIENCE --tol $TOL \
        --mip_rc_threshold $MIP_RC_THRESHOLD --mip_gap $MIP_GAP
    
    if [ -d "$TEMP_DIR/$method" ]; then
        mv "$TEMP_DIR/$method" "$TARGET_DIR/"
        cp "$TEMP_DIR/problem_config.json" "$TARGET_DIR/" 2>/dev/null
    fi
    clean_temp
    sleep 5
done

# ==========================================================
# PART 3: Exact (Gap 5%) - Risky
# ==========================================================
MIP_GAP=0.05
TARGET_DIR=$OUT_DIR_5

echo ""
echo ">>> [3/4] EXACT Method | Gap 5% (Risky)"
# Do NOT delete TARGET_DIR here (we append to it)
clean_temp

for method in "${EXACT_METHOD[@]}"; do
    echo "--- Running $method (Gap 5%) ---"
    python -u main.py --employees $EMP_NUM --weeks $WEEKS --methods $method \
        --time_limit $TIME_LIMIT --patience $PATIENCE --tol $TOL \
        --mip_rc_threshold $MIP_RC_THRESHOLD --mip_gap $MIP_GAP
    
    if [ -d "$TEMP_DIR/$method" ]; then
        mv "$TEMP_DIR/$method" "$TARGET_DIR/"
    fi
    clean_temp
    sleep 5
done

# ==========================================================
# PART 4: Exact (Gap 0%) - Very Risky
# ==========================================================
MIP_GAP=0.0
TARGET_DIR=$OUT_DIR_0

echo ""
echo ">>> [4/4] EXACT Method | Gap 0% (Very Risky)"
# Do NOT delete TARGET_DIR here (we append to it)
clean_temp

for method in "${EXACT_METHOD[@]}"; do
    echo "--- Running $method (Gap 0%) ---"
    python -u main.py --employees $EMP_NUM --weeks $WEEKS --methods $method \
        --time_limit $TIME_LIMIT --patience $PATIENCE --tol $TOL \
        --mip_rc_threshold $MIP_RC_THRESHOLD --mip_gap $MIP_GAP
    
    if [ -d "$TEMP_DIR/$method" ]; then
        mv "$TEMP_DIR/$method" "$TARGET_DIR/"
    fi
    clean_temp
done

echo ""
echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "Check results in:"
echo "  - $OUT_DIR_5"
echo "  - $OUT_DIR_0"
echo "=========================================================="
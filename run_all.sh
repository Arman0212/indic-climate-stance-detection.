#!/usr/bin/env bash
# ==============================================================================
# End-to-End Pipeline Execution Script
# Runs: Preprocess -> Train Baseline -> Evaluate -> Predict on sample data
# ==============================================================================

set -euo pipefail

# Determine repository root directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Select Python environment: prioritize local .venv if available
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "Error: Python 3 not found in .venv or PATH." >&2
    exit 1
fi

echo "========================================================================"
echo " Starting Stance Detection Pipeline with Python: $PYTHON"
echo "========================================================================"

# Step 1: Preprocess sample comments
echo ""
echo "[Step 1/4] Preprocessing comments..."
"$PYTHON" src/preprocess.py \
    --input "data/sample/sample_comments.csv" \
    --output "data/processed/sample_preprocessed.csv" \
    --config "config.yaml"

# Step 2: Train baseline TF-IDF + Logistic Regression model (Stratified 5-Fold CV)
echo ""
echo "[Step 2/4] Training baseline model with Stratified 5-Fold CV..."
"$PYTHON" src/train_baseline.py \
    --input "data/processed/sample_preprocessed.csv" \
    --folds-output "data/processed/folds.csv" \
    --output-dir "outputs" \
    --config "config.yaml"

# Step 3: Evaluate model predictions and generate confusion matrix
echo ""
echo "[Step 3/4] Evaluating out-of-fold performance..."
"$PYTHON" src/evaluate.py \
    --input "outputs/oof_predictions.csv" \
    --output-dir "outputs" \
    --config "config.yaml"

# Step 4: Run inference to generate final predictions
echo ""
echo "[Step 4/4] Generating predictions on test sample..."
"$PYTHON" src/predict.py \
    --input "data/sample/sample_comments.csv" \
    --output "outputs/predictions.csv" \
    --model-path "outputs/baseline_model.joblib" \
    --config "config.yaml"

echo ""
echo "========================================================================"
echo " Pipeline completed successfully!"
echo " Predictions saved to: outputs/predictions.csv"
echo " Confusion matrix saved to: outputs/confusion_matrix.png"
echo " Benchmark log saved to: outputs/results_log.csv"
echo "========================================================================"

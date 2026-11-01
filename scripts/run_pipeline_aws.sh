#!/usr/bin/env bash
# ==============================================================================
# CV3D AWS Pipeline Runner
# Usage: ./run_pipeline_aws.sh <OBJECT_ID> [--full-train]
# Examples:
#   ./run_pipeline_aws.sh box_001
#   ./run_pipeline_aws.sh basket_001 --full-train
# ==============================================================================
set -euo pipefail

OBJECT_ID="${1:-box_001}"
MODE="${2:---smoke-test}"

PROJECT_DIR="$HOME/cv3d"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

export QT_QPA_PLATFORM="offscreen"

DATA_DIR="$PROJECT_DIR/data/$OBJECT_ID"
PROCESSED_DIR="$PROJECT_DIR/processed/$OBJECT_ID/joint_sfm"
OUTPUTS_DIR="$PROJECT_DIR/outputs/$OBJECT_ID"

echo "=========================================================="
echo "🎯 Running Pipeline for: $OBJECT_ID (Mode: $MODE)"
echo "=========================================================="

# 1. Unpack archive if not already extracted
if [ ! -d "$DATA_DIR/images" ]; then
    ARCHIVE_PATH="$PROJECT_DIR/dataset_${OBJECT_ID}.tar"
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "❌ Error: Archive $ARCHIVE_PATH not found! Please upload it to $PROJECT_DIR/"
        exit 1
    fi
    echo "📦 Extracting $ARCHIVE_PATH..."
    tar -xf "$ARCHIVE_PATH" -C "$PROJECT_DIR/data"
fi

echo "🖼️ Verifying images and masks:"
echo "Images: $(ls "$DATA_DIR/images"/*.png | wc -l)"
echo "Masks:  $(ls "$DATA_DIR/masks"/*.png | wc -l)"

# 2. Joint COLMAP Reconstruction
echo "=========================================================="
echo "⚙️ [1/6] Running Joint COLMAP Reconstruction..."
echo "=========================================================="
mkdir -p "$PROCESSED_DIR"

ns-process-data images \
    --verbose \
    --data "$DATA_DIR/images" \
    --output-dir "$PROCESSED_DIR" \
    --camera-type perspective \
    --matching-method vocab_tree \
    --num-downscales 2 \
    --skip-image-processing

# 3. Audit Registration Rates
echo "=========================================================="
echo "📊 [2/6] Auditing Registration Rates..."
echo "=========================================================="
python "$PROJECT_DIR/scripts/audit_registration.py" \
    --input-images "$DATA_DIR/images" \
    --transforms "$PROCESSED_DIR/transforms.json" \
    --output-csv "$PROCESSED_DIR/registration_by_condition.csv"

# 4. Attach Pre-Computed High-Fidelity Masks
echo "=========================================================="
echo "🎭 [3/6] Attaching Pre-Computed Masks..."
echo "=========================================================="
python "$PROJECT_DIR/scripts/attach_masks.py" \
    --dataset "$PROCESSED_DIR" \
    --source-masks "$DATA_DIR/masks"

# 5. Generate Condition Splits & Evaluation Subsets
echo "=========================================================="
echo "✂️ [4/6] Generating Condition Splits..."
echo "=========================================================="
for cond in orbit_ spiral_ blur_1_50_ blur_1_25_; do
    python "$PROJECT_DIR/scripts/create_condition_splits.py" \
        --dataset "$PROCESSED_DIR" \
        --condition-prefix "$cond" \
        --holdout-prefix holdout_ \
        --ratios 1.0,0.5,0.25,0.10
done

# 6. Validate Dataset
echo "=========================================================="
echo "🔍 [5/6] Validating Processed Dataset..."
echo "=========================================================="
python "$PROJECT_DIR/scripts/validate_dataset.py" \
    --dataset "$PROCESSED_DIR" \
    --transforms transforms_orbit_views_100.json \
    --require-masks

# 7. Training (Smoke Test or Full)
echo "=========================================================="
echo "🚀 [6/6] Executing Model Training & Evaluation..."
echo "=========================================================="
mkdir -p "$OUTPUTS_DIR"

if [ "$MODE" = "--full-train" ]; then
    ITERS=30000
    echo "🔥 Mode: FULL TRAINING ($ITERS iterations)"
    for method in nerfacto splatfacto; do
        for cond in orbit spiral blur_1_50 blur_1_25; do
            SPLIT_FILE="transforms_${cond}_views_100.json"
            EXP_NAME="${OBJECT_ID}_${cond}_100_${method}"
            echo "Training $method on $SPLIT_FILE..."
            ns-train "$method" \
                --data "$PROCESSED_DIR/$SPLIT_FILE" \
                --experiment-name "$EXP_NAME" \
                --output-dir "$OUTPUTS_DIR" \
                --max-num-iterations "$ITERS" \
                --vis tensorboard \
                nerfstudio-data --downscale-factor 2
        done
    done
else
    ITERS=500
    echo "⚡ Mode: SMOKE TEST ($ITERS iterations on orbit_100)"
    ns-train nerfacto \
        --data "$PROCESSED_DIR/transforms_orbit_views_100.json" \
        --experiment-name "${OBJECT_ID}_smoke_nerfacto" \
        --output-dir "$OUTPUTS_DIR" \
        --max-num-iterations "$ITERS" \
        --vis tensorboard \
        nerfstudio-data --downscale-factor 2

    ns-train splatfacto \
        --data "$PROCESSED_DIR/transforms_orbit_views_100.json" \
        --experiment-name "${OBJECT_ID}_smoke_splatfacto" \
        --output-dir "$OUTPUTS_DIR" \
        --max-num-iterations "$ITERS" \
        --vis tensorboard \
        nerfstudio-data --downscale-factor 2
fi

echo "=========================================================="
echo "🎉 Run complete for $OBJECT_ID!"
echo "Reconstruction: $PROCESSED_DIR"
echo "Outputs & Models: $OUTPUTS_DIR"
echo "=========================================================="

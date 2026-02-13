#!/bin/bash
set -e

# ============================================================
# Train K dynamics models with different seeds for ensemble
# ============================================================
# Usage:
#   ./train_ensemble.sh <train_hdf5> <test_hdf5> <output_root> [--parallel] [--num_models K]
#
# Examples:
#   # Sequential (1 GPU):
#   ./train_ensemble.sh data/train.hdf5 data/val.hdf5 results/ensemble
#
#   # Parallel on multi-GPU machine (each model on a different GPU):
#   ./train_ensemble.sh data/train.hdf5 data/val.hdf5 results/ensemble --parallel
#
#   # 5 models in parallel:
#   ./train_ensemble.sh data/train.hdf5 data/val.hdf5 results/ensemble --parallel --num_models 5
# ============================================================

TRAIN_HDF5=${1:?"Usage: ./train_ensemble.sh <train_hdf5> <test_hdf5> <output_root> [--parallel] [--num_models K]"}
TEST_HDF5=${2:?"Missing test_hdf5 path"}
OUTPUT_ROOT=${3:?"Missing output root directory"}
shift 3

# Parse optional flags
PARALLEL=false
NUM_MODELS=5
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=true; shift ;;
        --num_models) NUM_MODELS=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Generate seed list
SEEDS=()
for i in $(seq 1 $NUM_MODELS); do
    SEEDS+=($((i * 111)))  # 111, 222, 333, 444, 555, ...
done

NUM_EPOCHS=6000
BATCH_SIZE=16
ACTION_DIM=7
PROPRIO_DIM=15
ACTION_CHUNK=16

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Detected ${NUM_GPUS} GPU(s)"
echo "Training ${NUM_MODELS} models with seeds: ${SEEDS[*]}"
echo "Mode: $(if $PARALLEL; then echo "PARALLEL"; else echo "SEQUENTIAL"; fi)"

mkdir -p "$OUTPUT_ROOT"

if $PARALLEL; then
    # Launch all models in parallel
    # Works on single GPU (CUDA time-slices, ~4GB VRAM each) or multi-GPU (round-robin)
    PIDS=()
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu=$((i % NUM_GPUS))
        exp_dir="${OUTPUT_ROOT}/seed_${seed}"

        echo "[GPU ${gpu}] Launching seed=${seed} -> ${exp_dir}"

        CUDA_VISIBLE_DEVICES=$gpu python train_dynaguide.py \
            --exp_dir "$exp_dir" \
            --train_hdf5 "$TRAIN_HDF5" \
            --test_hdf5 "$TEST_HDF5" \
            --cameras third_person \
            --action_dim $ACTION_DIM \
            --proprio_key proprio \
            --proprio_dim $PROPRIO_DIM \
            --num_epochs $NUM_EPOCHS \
            --action_chunk_length $ACTION_CHUNK \
            --batch_size $BATCH_SIZE \
            --seed $seed \
            --noised &

        PIDS+=($!)
    done

    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    TOTAL_VRAM_GB=$((TOTAL_VRAM_MB / 1024))
    ESTIMATED_GB=$((NUM_MODELS * 7))
    echo "All ${NUM_MODELS} training jobs launched. Waiting for completion..."
    echo "  Estimated VRAM: ~${ESTIMATED_GB}GB needed / ${TOTAL_VRAM_GB}GB available"
    if [ $ESTIMATED_GB -gt $TOTAL_VRAM_GB ]; then
        echo "  WARNING: May OOM. Consider fewer --num_models or removing --parallel"
    fi

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
else
    # Sequential training on GPU 0
    for seed in "${SEEDS[@]}"; do
        exp_dir="${OUTPUT_ROOT}/seed_${seed}"
        echo "=========================================="
        echo "Training seed=${seed} -> ${exp_dir}"
        echo "=========================================="

        python train_dynaguide.py \
            --exp_dir "$exp_dir" \
            --train_hdf5 "$TRAIN_HDF5" \
            --test_hdf5 "$TEST_HDF5" \
            --cameras third_person \
            --action_dim $ACTION_DIM \
            --proprio_key proprio \
            --proprio_dim $PROPRIO_DIM \
            --num_epochs $NUM_EPOCHS \
            --action_chunk_length $ACTION_CHUNK \
            --batch_size $BATCH_SIZE \
            --seed $seed \
            --noised

        echo "Finished seed=${seed}"
    done
fi

echo ""
echo "All ${NUM_MODELS} models trained. Checkpoints:"
CKPT_ARGS=""
for seed in "${SEEDS[@]}"; do
    ckpt="${OUTPUT_ROOT}/seed_${seed}/${NUM_EPOCHS}.pth"
    echo "  ${ckpt}"
    CKPT_ARGS="${CKPT_ARGS} ${ckpt}"
done
echo ""
echo "Next step: run the analysis"
echo "  python analyze_ensemble_disagreement.py \\"
echo "    --checkpoints${CKPT_ARGS} \\"
echo "    --test_hdf5 ${TEST_HDF5} \\"
echo "    --output_dir ${OUTPUT_ROOT}/analysis"

#!/bin/bash
set -e

# ============================================================
# Smoke Test: Full Pipeline Check (1-5 minutes)
# ============================================================
# Creates a tiny synthetic HDF5 dataset, trains 3 small ensemble
# models for 3 epochs each, then runs the analysis script.
# Verifies the entire pipeline works before committing to a
# multi-day training run.
#
# Usage:
#   ./smoke_test.sh
#
# What it checks:
#   [1] Synthetic data generation
#   [2] Model instantiation + forward/backward pass
#   [3] Seeded training with --noised flag
#   [4] Checkpoint saving/loading
#   [5] Parallel training (3 models simultaneously)
#   [6] Ensemble analysis script (scatter plot generation)
#
# Expected: completes in 1-5 minutes with "SMOKE TEST PASSED"
# ============================================================

SMOKE_DIR="smoke_test_output"
NUM_MODELS=3
NUM_EPOCHS=3
BATCH_SIZE=4
SEEDS=(42 123 999)

cleanup() {
    echo ""
    echo "Cleaning up smoke test artifacts..."
    rm -rf "$SMOKE_DIR"
}

fail() {
    echo ""
    echo "============================================"
    echo "  SMOKE TEST FAILED: $1"
    echo "============================================"
    exit 1
}

echo "============================================"
echo "  DynaGuide Ensemble Pipeline Smoke Test"
echo "============================================"
echo ""

# Step 0: Check dependencies
echo "[0/6] Checking dependencies..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || fail "PyTorch not installed"
python -c "import diffusers" || fail "diffusers not installed"
python -c "from core.dynamics_models import FinalStatePredictionDino; print('  DynaGuide core: OK')" || fail "DynaGuide not installed (run pip install -e .)"
echo ""

# Step 1: Generate tiny synthetic dataset
echo "[1/6] Generating synthetic HDF5 dataset..."
mkdir -p "$SMOKE_DIR"
python -c "
import h5py
import numpy as np
import json

def make_dataset(path, num_demos=30):
    with h5py.File(path, 'w') as f:
        grp = f.create_group('data')
        grp.attrs['env_args'] = json.dumps({'env_name': 'smoke_test'})
        total = 0
        behaviors = ['switch_on', 'switch_off', 'button_on', 'drawer_open',
                      'door_left', 'red_displace', 'blue_lift', 'pink_displace']
        for i in range(num_demos):
            length = np.random.randint(20, 40)
            demo = grp.create_group(f'demo_{i}')
            demo.create_dataset('obs/third_person', data=np.random.randint(0, 255, (length, 200, 200, 3), dtype=np.uint8))
            demo.create_dataset('obs/proprio', data=np.random.randn(length, 15).astype(np.float32))
            demo.create_dataset('actions', data=np.random.randn(length, 7).astype(np.float32) * 0.1)
            demo.create_dataset('rewards', data=np.zeros(length))
            dones = np.zeros(length)
            dones[-1] = 1
            demo.create_dataset('dones', data=dones)
            demo.attrs['num_samples'] = length
            demo.attrs['behavior'] = behaviors[i % len(behaviors)]
            total += length
        grp.attrs['total'] = total

make_dataset('${SMOKE_DIR}/train.hdf5', num_demos=30)
make_dataset('${SMOKE_DIR}/val.hdf5', num_demos=20)
print('  Created train.hdf5 (30 demos) and val.hdf5 (20 demos)')
" || fail "Synthetic data generation"
echo ""

# Step 2: Train models in parallel
echo "[2/6] Training ${NUM_MODELS} models in parallel (${NUM_EPOCHS} epochs each)..."
echo "       This tests: seeding, forward/backward pass, noise augmentation, checkpointing"
echo ""

PIDS=()
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    exp_dir="${SMOKE_DIR}/seed_${seed}"
    mkdir -p "$exp_dir"

    echo "  Launching seed=${seed}..."
    python train_dynaguide.py \
        --exp_dir "${exp_dir}/" \
        --train_hdf5 "${SMOKE_DIR}/train.hdf5" \
        --test_hdf5 "${SMOKE_DIR}/val.hdf5" \
        --cameras third_person \
        --action_dim 7 \
        --proprio_key proprio \
        --proprio_dim 15 \
        --num_epochs $NUM_EPOCHS \
        --action_chunk_length 16 \
        --batch_size $BATCH_SIZE \
        --seed $seed \
        --noised \
        > "${exp_dir}/train.log" 2>&1 &
    PIDS+=($!)
done

echo "  Waiting for all ${NUM_MODELS} training jobs..."

ALL_OK=true
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    seed=${SEEDS[$i]}
    if wait "$pid"; then
        echo "  seed=${seed}: OK"
    else
        echo "  seed=${seed}: FAILED (see ${SMOKE_DIR}/seed_${seed}/train.log)"
        ALL_OK=false
    fi
done

if ! $ALL_OK; then
    echo ""
    echo "Training logs:"
    for seed in "${SEEDS[@]}"; do
        echo "--- seed=${seed} (last 20 lines) ---"
        tail -20 "${SMOKE_DIR}/seed_${seed}/train.log"
    done
    fail "One or more training jobs failed"
fi
echo ""

# Step 3: Verify checkpoints exist
echo "[3/6] Verifying checkpoints..."
for seed in "${SEEDS[@]}"; do
    ckpt="${SMOKE_DIR}/seed_${seed}/${NUM_EPOCHS}.pth"
    if [ ! -f "$ckpt" ]; then
        # Also check for 0.pth since epochs are 0-indexed and we save at i%100==0
        ckpt0="${SMOKE_DIR}/seed_${seed}/0.pth"
        if [ ! -f "$ckpt0" ]; then
            fail "Missing checkpoint for seed=${seed}"
        fi
    fi
    echo "  seed=${seed}: checkpoint found"
done
echo ""

# Step 4: Verify models produce different outputs (seeds actually work)
echo "[4/6] Verifying seed diversity (models should differ)..."
python -c "
import torch
import numpy as np
from core.dynamics_models import FinalStatePredictionDino

seeds = [${SEEDS[0]}, ${SEEDS[1]}, ${SEEDS[2]}]
outputs = []
for seed in seeds:
    # Find the latest checkpoint
    import glob
    ckpts = sorted(glob.glob(f'${SMOKE_DIR}/seed_{seed}/*.pth'))
    if not ckpts:
        raise RuntimeError(f'No checkpoint found for seed={seed}')
    ckpt = ckpts[-1]

    model = FinalStatePredictionDino(7, 16, cameras=['third_person'], reconstruction=True, proprio='proprio', proprio_dim=15)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    # Extract a sample of weights to compare
    sample_weight = list(model.parameters())[10].data.flatten()[:100].numpy()
    outputs.append(sample_weight)
    print(f'  seed={seed}: loaded {ckpt}, weight sample mean={sample_weight.mean():.6f}')

# Check pairwise differences
diff_01 = np.abs(outputs[0] - outputs[1]).mean()
diff_02 = np.abs(outputs[0] - outputs[2]).mean()
diff_12 = np.abs(outputs[1] - outputs[2]).mean()
print(f'  Weight diffs: {diff_01:.6f}, {diff_02:.6f}, {diff_12:.6f}')
if diff_01 < 1e-8 and diff_02 < 1e-8:
    raise RuntimeError('Models are identical — seeds not working!')
print('  Models are different: OK')
" || fail "Seed diversity check"
echo ""

# Step 5: Run ensemble analysis
echo "[5/6] Running ensemble analysis..."
# Collect all checkpoints (use the last available one per seed)
CKPT_ARGS=""
for seed in "${SEEDS[@]}"; do
    # Get the latest .pth file
    latest=$(ls -t "${SMOKE_DIR}/seed_${seed}"/*.pth 2>/dev/null | head -1)
    CKPT_ARGS="${CKPT_ARGS} ${latest}"
done

DEVICE="cuda"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEVICE="cpu"

python analyze_ensemble_disagreement.py \
    --checkpoints $CKPT_ARGS \
    --test_hdf5 "${SMOKE_DIR}/val.hdf5" \
    --output_dir "${SMOKE_DIR}/analysis" \
    --max_samples 20 \
    --device "$DEVICE" \
    || fail "Ensemble analysis script"
echo ""

# Step 6: Verify outputs
echo "[6/6] Verifying analysis outputs..."
for f in disagreement_vs_error.png category_boxplots.png ensemble_results.npz; do
    if [ ! -f "${SMOKE_DIR}/analysis/${f}" ]; then
        fail "Missing output: ${f}"
    fi
    echo "  ${f}: OK"
done
echo ""

# Summary
echo "============================================"
echo "  SMOKE TEST PASSED"
echo "============================================"
echo ""
echo "All pipeline stages verified:"
echo "  [1] Data loading"
echo "  [2] Parallel model training with seeding"
echo "  [3] Checkpoint save/load"
echo "  [4] Seed diversity (models differ)"
echo "  [5] Ensemble disagreement analysis"
echo "  [6] Plot generation"
echo ""
echo "Output in: ${SMOKE_DIR}/"
echo "  - Training logs: ${SMOKE_DIR}/seed_*/train.log"
echo "  - Plots: ${SMOKE_DIR}/analysis/"
echo ""

echo "To clean up: rm -rf ${SMOKE_DIR}"
if [[ "${1:-}" == "--cleanup" ]]; then
    cleanup
fi

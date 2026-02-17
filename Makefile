# DynaGuide Ensemble Pipeline
# Usage: make <target> [VAR=value]
#
# Examples:
#   make smoke-test
#   make convert SPLIT=training TASK=CalvinDD_train CALVIN_DIR=/workspace/calvin/dataset/task_D_D
#   make train TRAIN_HDF5=dataset/CalvinDD_train/data.hdf5 TEST_HDF5=dataset/CalvinDD_test/labeled_test_set.hdf5
#   make analyze TEST_HDF5=dataset/CalvinDD_val/data.hdf5
#   make figures NPZ=results/ensemble/analysis/ensemble_results.npz
#   make eval-baseline AGENT=<base_policy> GUIDANCE=<dynamics_model> EXP_CONFIG=<config.json>
#   make eval-adaptive AGENT=<base_policy> GUIDANCE=<dynamics_model> EXP_CONFIG=<config.json>
#   make tensorboard

# ============================================================
# Configuration (override with make target VAR=value)
# ============================================================
CALVIN_DIR     ?= /workspace/calvin/dataset/task_D_D
SPLIT          ?= training
TASK           ?= CalvinDD_train
TRAIN_HDF5     ?= dataset/CalvinDebug_train/data.hdf5
TEST_HDF5      ?= dataset/CalvinDebug_test/labeled_test_set.hdf5
ANALYSIS_HDF5  ?= dataset/CalvinDebug_val/data.hdf5
OUTPUT_ROOT    ?= results/ensemble
ANALYSIS_DIR   ?= $(OUTPUT_ROOT)/analysis
NUM_MODELS     ?= 5
NUM_EPOCHS     ?= 6000
PATIENCE       ?= 20
BATCH_SIZE     ?= 16
CHECKPOINT     ?= 200
TB_PORT        ?= 6006
LOCAL_DEST     ?= ~/Desktop

# Evaluation / Adaptive Guidance
AGENT          ?= pretrained/base_policy.pth
GUIDANCE       ?= pretrained/dynamics_model.pth
EXP_CONFIG     ?= calvin_exp_configs_examples/switch_on.json
SCALE          ?= 1.5
ALPHA          ?= 30
SS             ?= 4
N_ROLLOUTS     ?= 50
HORIZON        ?= 400
ADAPTIVE_BETA  ?= 1.0
EVAL_DIR       ?= results/eval
FIGURES_DIR    ?= $(ANALYSIS_DIR)/figures
NPZ            ?= $(ANALYSIS_DIR)/ensemble_results.npz
PRETRAINED_DIR ?= pretrained

.PHONY: help install smoke-test convert split-val verify-labels train \
        analyze figures eval-baseline eval-adaptive eval-compare \
        download-plot download-pretrained tensorboard status kill \
        clean clean-checkpoints

# ============================================================
# Help
# ============================================================
help:
	@echo "DynaGuide Ensemble Pipeline"
	@echo "==========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make smoke-test       Run full pipeline smoke test"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make convert          Convert raw CALVIN to HDF5"
	@echo "    CALVIN_DIR=...        Path to CALVIN split (default: $(CALVIN_DIR))"
	@echo "    SPLIT=training        Subfolder: training or validation"
	@echo "    TASK=CalvinDD_train   Output name under dataset/"
	@echo "  make convert-all      Convert both training and validation for task_D_D"
	@echo "  make split-val        Split validation HDF5 by behavior category"
	@echo "    TASK=CalvinDD_val     Source HDF5 task name"
	@echo "  make verify-labels    Print behavior labels in test set"
	@echo ""
	@echo "Training:"
	@echo "  make train            Train ensemble (parallel, with early stopping)"
	@echo "    TRAIN_HDF5=...        Training HDF5 path"
	@echo "    TEST_HDF5=...         Validation HDF5 path"
	@echo "    OUTPUT_ROOT=...       Output directory (default: $(OUTPUT_ROOT))"
	@echo "    NUM_MODELS=5          Number of ensemble members"
	@echo "    NUM_EPOCHS=6000       Max epochs"
	@echo "    PATIENCE=20           Early stopping patience"
	@echo "  make status            Check training progress (GPU usage + latest logs)"
	@echo "  make tensorboard       Start TensorBoard on port $(TB_PORT)"
	@echo "  make kill              Kill all training processes"
	@echo ""
	@echo "Analysis:"
	@echo "  make analyze           Run ensemble disagreement analysis"
	@echo "    ANALYSIS_HDF5=...     HDF5 to analyze against"
	@echo "    CHECKPOINT=best       Which checkpoint: best, latest, or epoch number"
	@echo "  make figures           Generate uncertainty calibration plots from .npz"
	@echo "    NPZ=...               Path to ensemble_results.npz"
	@echo "  make download-plot     SCP scatter plot to local machine"
	@echo ""
	@echo "Evaluation (Full Pipeline):"
	@echo "  make download-pretrained  Download author's pretrained models"
	@echo "  make install-calvin       Install CALVIN env + robomimic"
	@echo "  make eval-baseline     Run DynaGuide with fixed scale (original)"
	@echo "    AGENT=...             Base policy checkpoint"
	@echo "    GUIDANCE=...          Dynamics model checkpoint"
	@echo "    EXP_CONFIG=...        Experiment config JSON"
	@echo "    SCALE=1.5             Fixed guidance scale"
	@echo "  make eval-adaptive     Run DynaGuide with adaptive scale (ours)"
	@echo "    ADAPTIVE_BETA=1.0     Sensitivity to disagreement"
	@echo "  make eval-compare      Run both baseline and adaptive, print comparison"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-checkpoints Remove intermediate checkpoints (keep best + final)"
	@echo "  make clean             Remove all results"
	@echo ""
	@echo "Full D Pipeline (copy-paste):"
	@echo "  make convert-all"
	@echo "  make split-val TASK=CalvinDD_val"
	@echo "  make verify-labels TEST_HDF5=dataset/CalvinDD_test/labeled_test_set.hdf5"
	@echo "  make train TRAIN_HDF5=dataset/CalvinDD_train/data.hdf5 TEST_HDF5=dataset/CalvinDD_test/labeled_test_set.hdf5"
	@echo "  make analyze ANALYSIS_HDF5=dataset/CalvinDD_val/data.hdf5 CHECKPOINT=best"
	@echo "  make figures"
	@echo ""
	@echo "Full Eval Pipeline (copy-paste):"
	@echo "  make download-pretrained"
	@echo "  make install-calvin"
	@echo "  make eval-compare EXP_CONFIG=calvin_exp_configs_examples/switch_on.json"

# ============================================================
# Setup
# ============================================================
install:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install h5py matplotlib diffusers tensorboard tqdm einops imageio scipy
	python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
	@echo "Done. DINOv2 pre-cached."

smoke-test:
	bash smoke_test.sh

# ============================================================
# Data Preparation
# ============================================================
convert:
	cd data_processing_calvin && python calvin_to_labeled_hdf5.py \
		--original_dir $(CALVIN_DIR)/$(SPLIT)/ \
		--task_name $(TASK)

convert-all:
	@echo "=== Converting training split ==="
	cd data_processing_calvin && python calvin_to_labeled_hdf5.py \
		--original_dir $(CALVIN_DIR)/training/ \
		--task_name CalvinDD_train
	@echo ""
	@echo "=== Converting validation split ==="
	cd data_processing_calvin && python calvin_to_labeled_hdf5.py \
		--original_dir $(CALVIN_DIR)/validation/ \
		--task_name CalvinDD_val

split-val:
	cd data_processing_calvin && python split_behavioral_validation_datasets_calvin.py \
		--original_dir ../dataset/$(TASK)/data.hdf5 \
		--task_name $(TASK)_test

verify-labels:
	@python -c "\
	import h5py; \
	f = h5py.File('$(TEST_HDF5)', 'r'); \
	cats = {}; \
	[cats.update({f['data'][d].attrs.get('behavior','NONE'): cats.get(f['data'][d].attrs.get('behavior','NONE'), 0) + 1}) for d in f['data']]; \
	print('Behavior counts:'); \
	[print(f'  {k}: {v}') for k, v in sorted(cats.items())]; \
	f.close()"

# ============================================================
# Training
# ============================================================
train:
	bash train_ensemble.sh \
		$(TRAIN_HDF5) $(TEST_HDF5) $(OUTPUT_ROOT) \
		--parallel --num_models $(NUM_MODELS) --num_epochs $(NUM_EPOCHS) --patience $(PATIENCE)

status:
	@echo "=== GPU Usage ==="
	@nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "No GPU processes"
	@echo ""
	@echo "=== Latest Training Logs ==="
	@for d in $(OUTPUT_ROOT)/seed_*; do \
		seed=$$(basename $$d); \
		latest_ckpt=$$(ls -t $$d/*.pth 2>/dev/null | head -1); \
		echo "$$seed: latest checkpoint = $${latest_ckpt:-none}"; \
	done
	@echo ""
	@echo "=== Recent Output (seed_111) ==="
	@tail -5 $(OUTPUT_ROOT)/seed_111/train.log 2>/dev/null || \
		ps aux | grep "seed 111" | grep -v grep | head -1 || echo "Not running"

tensorboard:
	tensorboard --logdir $(OUTPUT_ROOT) --bind_all --port $(TB_PORT) &
	@echo "TensorBoard running at http://localhost:$(TB_PORT)"

kill:
	pkill -f train_dynaguide.py || echo "No training processes found"

# ============================================================
# Analysis
# ============================================================
analyze:
	$(eval CKPTS := $(shell \
		if [ "$(CHECKPOINT)" = "best" ]; then \
			for d in $(OUTPUT_ROOT)/seed_*; do ls $$d/best_val_epoch-*.pth 2>/dev/null | head -1; done; \
		elif [ "$(CHECKPOINT)" = "latest" ]; then \
			for d in $(OUTPUT_ROOT)/seed_*; do ls -t $$d/*.pth 2>/dev/null | head -1; done; \
		else \
			for d in $(OUTPUT_ROOT)/seed_*; do echo $$d/$(CHECKPOINT).pth; done; \
		fi))
	python analyze_ensemble_disagreement.py \
		--checkpoints $(CKPTS) \
		--test_hdf5 $(ANALYSIS_HDF5) \
		--output_dir $(ANALYSIS_DIR)
	@echo ""
	@echo "Results in $(ANALYSIS_DIR)/"
	@ls $(ANALYSIS_DIR)/

figures:
	python plot_uncertainty_figures.py \
		--npz $(NPZ) \
		--output_dir $(FIGURES_DIR)

download-plot:
	@echo "Run this on your LOCAL machine:"
	@echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
	@echo "    root@<POD_IP>:$(PWD)/$(ANALYSIS_DIR)/*.png \\"
	@echo "    $(LOCAL_DEST)/"
	@echo ""
	@echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
	@echo "    root@<POD_IP>:$(PWD)/$(FIGURES_DIR)/*.png \\"
	@echo "    $(LOCAL_DEST)/"

# ============================================================
# Evaluation: Full Pipeline (Baseline vs Adaptive)
# ============================================================

# Resolve ensemble checkpoint paths
ENSEMBLE_CKPTS = $(shell for d in $(OUTPUT_ROOT)/seed_*; do ls $$d/best_val_epoch-*.pth 2>/dev/null | head -1; done)

download-pretrained:
	@mkdir -p $(PRETRAINED_DIR)
	@echo "Download these manually from Google Drive (requires browser auth):"
	@echo "  1. Base policy:          https://drive.google.com/file/d/1lcI_PBgFIYDsoK4T4qO7SGJJgI2lw0Kd"
	@echo "  2. Dynamics model:       https://drive.google.com/file/d/1DeOnoDacXjBHgy1DGoRJpYIR8SyDy5fJ"
	@echo "  3. Guidance (switch_on): https://drive.google.com/file/d/1wtEGnG87Y-imqD2MygcqbJwp_RGi7YNB"
	@echo ""
	@echo "Place them in $(PRETRAINED_DIR)/"
	@echo "Then update AGENT and GUIDANCE in your make commands."

install-calvin:
	cd robomimic && pip install -e .
	cd calvin && bash install.sh
	@echo "CALVIN environment and robomimic installed."

eval-baseline:
	@mkdir -p $(EVAL_DIR)/baseline
	python run_dynaguide.py \
		--video_path $(EVAL_DIR)/baseline/rollout.mp4 \
		--dataset_path $(EVAL_DIR)/baseline/rollout.hdf5 \
		--dataset_obs \
		--json_path $(EVAL_DIR)/baseline/results.json \
		--horizon $(HORIZON) \
		--n_rollouts $(N_ROLLOUTS) \
		--agent $(AGENT) \
		--output_folder $(EVAL_DIR)/baseline \
		--video_skip 2 \
		--exp_setup_config $(EXP_CONFIG) \
		--guidance $(GUIDANCE) \
		--camera_names third_person \
		--scale $(SCALE) --ss $(SS) --alpha $(ALPHA) \
		--save_frames
	@echo ""
	@echo "Baseline results:"
	@cat $(EVAL_DIR)/baseline/results.json

eval-adaptive:
	@mkdir -p $(EVAL_DIR)/adaptive
	python run_dynaguide.py \
		--video_path $(EVAL_DIR)/adaptive/rollout.mp4 \
		--dataset_path $(EVAL_DIR)/adaptive/rollout.hdf5 \
		--dataset_obs \
		--json_path $(EVAL_DIR)/adaptive/results.json \
		--horizon $(HORIZON) \
		--n_rollouts $(N_ROLLOUTS) \
		--agent $(AGENT) \
		--output_folder $(EVAL_DIR)/adaptive \
		--video_skip 2 \
		--exp_setup_config $(EXP_CONFIG) \
		--guidance $(GUIDANCE) \
		--camera_names third_person \
		--scale $(SCALE) --ss $(SS) --alpha $(ALPHA) \
		--ensemble_paths $(ENSEMBLE_CKPTS) \
		--adaptive_beta $(ADAPTIVE_BETA) \
		--save_frames
	@echo ""
	@echo "Adaptive results:"
	@cat $(EVAL_DIR)/adaptive/results.json

eval-compare:
	@echo "=== Running Baseline (fixed scale=$(SCALE)) ==="
	$(MAKE) eval-baseline EVAL_DIR=$(EVAL_DIR)
	@echo ""
	@echo "=== Running Adaptive (beta=$(ADAPTIVE_BETA)) ==="
	$(MAKE) eval-adaptive EVAL_DIR=$(EVAL_DIR)
	@echo ""
	@echo "============================================================"
	@echo "COMPARISON"
	@echo "============================================================"
	@echo "Baseline:"
	@cat $(EVAL_DIR)/baseline/results.json
	@echo ""
	@echo "Adaptive:"
	@cat $(EVAL_DIR)/adaptive/results.json
	@echo "============================================================"

# ============================================================
# Cleanup
# ============================================================
clean-checkpoints:
	@echo "Removing intermediate checkpoints (keeping best + final)..."
	@for d in $(OUTPUT_ROOT)/seed_*; do \
		find $$d -name "*.pth" ! -name "best_val_epoch-*" ! -name "$$(ls -t $$d/*.pth | head -1 | xargs basename)" -delete; \
	done
	@echo "Done."

clean:
	rm -rf $(OUTPUT_ROOT)
	rm -rf smoke_test_output
	@echo "Cleaned."

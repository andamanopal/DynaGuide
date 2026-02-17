# DynaGuide Ensemble Pipeline
# Usage: make <target> [VAR=value]
#
# Examples:
#   make smoke-test
#   make convert SPLIT=training TASK=CalvinDD_train CALVIN_DIR=/workspace/calvin/dataset/task_D_D
#   make train TRAIN_HDF5=dataset/CalvinDD_train/data.hdf5 TEST_HDF5=dataset/CalvinDD_test/labeled_test_set.hdf5
#   make analyze TEST_HDF5=dataset/CalvinDD_val/data.hdf5
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

.PHONY: help install smoke-test convert split-val verify-labels train \
        analyze download-plot tensorboard status kill clean clean-checkpoints

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
	@echo "  make download-plot     SCP scatter plot to local machine"
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
	@echo "  make analyze ANALYSIS_HDF5=dataset/CalvinDD_val/data.hdf5"

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

download-plot:
	@echo "Run this on your LOCAL machine:"
	@echo "  scp -P 22128 -i ~/.ssh/id_ed25519 \\"
	@echo "    root@69.30.85.139:$(PWD)/$(ANALYSIS_DIR)/disagreement_vs_error.png \\"
	@echo "    $(LOCAL_DEST)/"

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

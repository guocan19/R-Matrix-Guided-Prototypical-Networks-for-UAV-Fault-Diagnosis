# R-Matrix-Guided-Prototypical-Networks-for-UAV-Fault-Diagnosis


## Overview

This repository implements R-matrix guided prototypical networks for UAV sensor and actuator fault diagnosis under severe class overlap. Two strategies are investigated:

| Method | Mechanism | Key Parameter |
|--------|-----------|---------------|
| **RGuided** | R-weighted cross-entropy: samples from high-overlap classes receive larger loss weights | α = 2.0 |
| **RCL-ProtoNet** | R-matrix guided sample-to-prototype contrastive loss: selectively pushes apart embeddings of class pairs with R > 0.5 | λ_c = 0.3, m_c = 0.5 |

Both methods outperform the standard prototypical network (ProtoNet) and traditional baselines (CNN, Transformer, BiLSTM, GRU, ResNet) on the RflyMAD benchmark.

## Environment

- Python 3.10+
- PyTorch 2.1+ with CUDA
- Key packages: numpy, scikit-learn, matplotlib, pandas

```bash
conda create -n rflymad python=3.10
conda activate rflymad
pip install torch numpy scikit-learn matplotlib pandas
```

## Project Structure

```
├── scripts/
│   ├── main.py                  # Main entry: experiments, sensitivity, multi-seed
├── src/
│   ├── pipeline_orchestrator.py # Full pipeline: load→preprocess→split→train→eval
│   ├── config.py                # Paths and directory utilities
│   ├── models/
│   │   ├── Prototypical/
│   │   │   └── Prototypical_model.py  # Encoder, ProtoNet, 3 loss classes, Wrapper
│   │   ├── CNN/CNN_model.py
│   │   ├── ResNet/ResNet_model.py
│   │   ├── Transformer/Transformer_model.py
│   │   └── RNN/RNN_model.py          # BiLSTM & GRU
│   ├── analysis/
│   │   └── class_overlap_analyzer.py  # R-matrix, ImR, ImR_aug computation
│   ├── data/
│   │   ├── data_adapter.py       # Log data loading and preprocessing
│   │   ├── data_preprocessor.py  # Standardization, feature selection
│   │   └── sequence_builder.py   # Sliding window segmentation
│   ├── evaluation/
│   │   └── model_evaluator.py    # Metrics, confusion matrix, ROC/PR curves
│   └── features/
│       └── label_processor.py    # Fault ID → class index mapping
├── tests/
│   ├── test_model.py             # Evaluate a single model on test set
│   └── test_all_models.py        # Batch evaluation + PR curve comparison
├── data/                         # RflyMAD dataset (log_*/ and TestCase_*/)
├── outputs/
│   ├── models/                   # Trained model checkpoints
│   ├── figures/                  # Generated figures
│   └── test_results/             # Test evaluation reports
└── README.md
```

## Quick Start

### 1. Prepare Data

Place the RflyMAD dataset under `data/`. Expected structure:
```
data/
├── log_0_2023-8-23-13-38-44/
│   ├── *_actuator_controls_0_0.csv
│   ├── *_actuator_outputs_0.csv
│   ├── *_vehicle_land_detected_0.csv
│   ├── *_rfly_ctrl_lxl_0.csv
│   ├── *_sensor_combined_0.csv
│   ├── *_vehicle_attitude_0.csv
│   └── *_vehicle_local_position_0.csv
├── ... (75 training log directories)
├── TestCase_1_2000000000/
├── ... (26 test case directories)
```

### 2. Run Ablation Experiments

```bash
python scripts/main.py
```

Runs the three core methods (**ProtoNet → RGuided → RCL-ProtoNet**). Each experiment trains independently with its own train/val split. Output goes to `outputs/models/` and `outputs/figures/`.

### 3. Evaluate on Test Set

```bash
# Single model
python tests/test_model.py --model RGuided --test_data data

# Batch evaluation (all models in outputs/models/)
python tests/test_all_models.py
```

### 4. Hyperparameter Sensitivity

```bash
# RCL-ProtoNet: λ_c ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0}
python scripts/main.py --sensitivity

# RGuided: α ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0}
python scripts/main.py --rg-sensitivity
```

### 5. Multi-Seed Statistical Validation

```bash
python scripts/main.py --multiseed
```

Tests ProtoNet, RGuided, and RCL-ProtoNet over 5 random seeds. Reports mean ± std.

### 6. Generate Framework Diagram

```bash
python scripts/plot_framework.py
# Output: outputs/figures/framework.svg
```

## Experiment Configurations

All experiments are defined in `scripts/main.py > main()`. Key configurations:

```python
# ProtoNet (CE only baseline)
('ProtoNet', 'prototypical', {
    'model': {},
    'overlap_analysis_enabled': True,
})

# RGuided (R-weighted CE, α=2.0)
('RGuided', 'prototypical', {
    'model': {'use_r_loss': True, 'r_alpha': 2.0},
    'overlap_analysis_enabled': True,
})

# RCL-ProtoNet (contrastive loss, λ_c=0.3, m_c=0.5)
('RCL-ProtoNet', 'prototypical', {
    'model': {
        'use_contrastive_loss': True,
        'lambda_c': 0.3, 'contrastive_margin': 0.5,
    },
    'overlap_analysis_enabled': True,
})
```

Shared hyperparameters (in `get_base_config()`): `lr=3e-4`, `embedding_dim=256`, `dropout=0.5`, `l2_reg=1e-4`, `epochs=120`, `batch_size=64`, `window_size=256`, `stride=32`.

## Key Design Decisions

- **No `lambda_reg` or `lambda_m`**: These prototype regularization and margin parameters are removed from all loss classes (always contributed zero gradient).
- **Independent splits**: Each experiment runs a full independent pipeline (no shared cache for main experiments). Sensitivity analyses use shared caches internally for fair λ_c/α comparison.
- **Three loss classes** in `Prototypical_model.py`:
  - `PrototypicalLoss`: Pure cross-entropy
  - `RGuidedPrototypicalLoss`: R-weighted cross-entropy
  - `ContrastivePrototypicalLoss`: CE + sample-to-prototype contrastive


## Citation

```bibtex
@article{guo2026class,
  title={Class Overlap Matters: R-Matrix Guided Prototypical Networks for UAV Fault Diagnosis},
  author={Guo, Can and Xiang, Min},
  journal={Mechanical Systems and Signal Processing},
  year={2026},
  note={Under review}
}
```

## License

This project is for research purposes. Contact the authors for usage permissions.

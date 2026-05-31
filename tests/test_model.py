"""
python tests/test_model.py --list
python tests/test_model.py --model resnet
"""
import matplotlib
#matplotlib.use('Agg')
import sys
import os
import argparse
import json
import pickle
import glob
import logging
from pathlib import Path
import numpy as np
# ========== Path Setup (must be before src imports) ==========
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import get_test_save_dir
from src.data.data_preprocessor import DataPreprocessor
from src.data.sequence_builder import SequenceBuilder
from src.features.label_processor import GlobalLabelMapper
from src.evaluation.model_evaluator import ModelEvaluator
from sklearn.preprocessing import StandardScaler
import src.config as cfg

# Force correct output directory
cfg.PROJECT_ROOT = PROJECT_ROOT
cfg.RESULT_DIR = str(PROJECT_ROOT / 'outputs')
cfg.MODELS_DIR = str(PROJECT_ROOT / 'outputs' / 'models')
RESULT_DIR = cfg.RESULT_DIR
MODELS_DIR = cfg.MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 1. Path Resolution ====================

def find_model_files(model_identifier: str) -> dict:
    """Auto-locate model-related files.
    Args:
        model_identifier: model directory name, e.g., '1_CNN', '2_ResNet', 'C_RCL-ProtoNet'
    """
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    # Direct match
    matched_dir = os.path.join(MODELS_DIR, model_identifier)

    if not os.path.exists(matched_dir):
        # Try case-insensitive match
        matched_dir = None
        for name in os.listdir(MODELS_DIR):
            full_path = os.path.join(MODELS_DIR, name)
            if os.path.isdir(full_path) and name.lower() == model_identifier.lower():
                matched_dir = full_path
                break

        if matched_dir is None:
            available = list_models()
            raise FileNotFoundError(
                f"Model '{model_identifier}' not found. Available: {available}"
            )

    files = {}

    # Find model file
    for ext in ['*.pth', '*.keras', '*.h5']:
        matches = glob.glob(os.path.join(matched_dir, ext))
        if matches:
            files['model'] = matches[0]
            break

    if 'model' not in files:
        raise FileNotFoundError(f"No model file in: {matched_dir}")

    # Find info file
    info_matches = glob.glob(os.path.join(matched_dir, '*_info.pkl'))
    if not info_matches:
        raise FileNotFoundError(f"No info file in: {matched_dir}")
    files['info'] = info_matches[0]

    # Config file
    config_path = os.path.join(matched_dir, 'config.json')
    if os.path.exists(config_path):
        files['config'] = config_path

    # Label mapper
    mapper_path = os.path.join(RESULT_DIR, 'label_mapper.pkl')
    if os.path.exists(mapper_path):
        files['mapper'] = mapper_path

    logger.info(f"Model: {model_identifier}")
    logger.info(f"  Model file: {os.path.basename(files['model'])}")
    logger.info(f"  Info file: {os.path.basename(files['info'])}")
    return files

# ==================== 2. Model Loading ====================
def load_model_resources(model_identifier: str):
    """Load model resources.
    Args:
        model_identifier: e.g., '1_CNN', 'C_RCL-ProtoNet'
    """
    files = find_model_files(model_identifier)

    # Infer model_type from filename/directory name
    model_file = os.path.basename(files['model'])
    dir_name = model_identifier.lower()
    if 'cnn' in model_file.lower() or 'cnn' in dir_name:
        model_type = 'cnn'
    elif 'bilstm' in model_file.lower() or 'bilstm' in dir_name:
        model_type = 'bilstm'
    elif 'gru' in model_file.lower() or 'gru' in dir_name:
        model_type = 'gru'
    elif 'resnet' in model_file.lower() or 'resnet' in dir_name:
        model_type = 'resnet'
    elif 'proto' in model_file.lower() or 'proto' in dir_name:
        model_type = 'prototypical'
    elif 'transformer' in model_file.lower() or 'transformer' in dir_name:
        model_type = 'transformer'
    else:
        _mtype = cfg.get('model', {}).get('type', 'unknown') if cfg else 'unknown'
        model_type = _mtype if _mtype else 'unknown'

    with open(files['info'], 'rb') as f:
        info = pickle.load(f)

    config = {}
    if files.get('config'):
        with open(files['config'], 'r', encoding='utf-8') as f:
            config = json.load(f)

    # Load label mapper
    mapper = GlobalLabelMapper()
    if files.get('mapper'):
        mapper.load(files['mapper'])
    elif 'global_mapping' in info:
        gm = info['global_mapping']
        if isinstance(gm, dict):
            mapper.fault_id_to_tech = gm.get('fault_id_to_tech', {0: 0})
            mapper.tech_to_fault_id = gm.get('tech_to_fault_id', {0: 0})
            mapper.name_mapping = gm.get('name_mapping', {0: "Normal"})
            mapper.num_classes = gm.get('num_classes', 1)
            mapper.is_fitted = True

    # Preprocessor
    preprocessor = DataPreprocessor({})
    if info.get('scaler_mean') is not None:
        preprocessor.scaler = StandardScaler()
        preprocessor.scaler.mean_ = np.array(info['scaler_mean'])
        preprocessor.scaler.scale_ = np.array(info['scaler_scale'])
    if info.get('selected_indices') is not None:
        preprocessor.selected_indices = np.array(info['selected_indices'])

    # Create model
    input_shape = info.get('input_shape')
    if input_shape is None:
        window_size = config.get('window_size', 128)
        n_features = len(info.get('selected_indices', [])) or 22
        input_shape = (window_size, n_features)

    num_classes = info.get('num_classes', mapper.num_classes)

    model = _create_model(model_type, input_shape, num_classes, info, config)
    model.load_model(files['model'])

    logger.info(f"Model loaded: {model_identifier} ({num_classes} classes)")
    return model, info, config, mapper, preprocessor, model_type


def _create_model(model_type, input_shape, num_classes, info, config=None):
    """Create model instance."""
    if config is None:
        config = {}
    model_cfg = config.get('model', {})
    if model_type == 'cnn':
        from src.models.CNN.CNN_model import CNN
        return CNN(input_shape=input_shape, num_classes=num_classes, learning_rate=1e-4)
    elif model_type == 'resnet':
        from src.models.ResNet.ResNet_model import ResNet
        return ResNet(input_shape=input_shape, num_classes=num_classes, learning_rate=1e-4)
    elif model_type == 'prototypical':
        from src.models.Prototypical.Prototypical_model import PrototypicalNetworkWrapper
        return PrototypicalNetworkWrapper(
            input_shape=input_shape, num_classes=num_classes,
            embedding_dim=info.get('embedding_dim', 128),
            use_contrastive_loss=model_cfg.get('use_contrastive_loss', False),
            use_r_loss=model_cfg.get('use_r_loss', False),
            use_r_init=model_cfg.get('use_r_init', False),
            use_focal_loss=model_cfg.get('use_focal_loss', False),
            focal_gamma=model_cfg.get('focal_gamma', 2.0),
            lambda_c=model_cfg.get('lambda_c', 0.1),
            contrastive_margin=model_cfg.get('contrastive_margin', 0.5),
            r_alpha=model_cfg.get('r_alpha', 2.0),
            r_push_strength=model_cfg.get('r_push_strength', 0.1),
        )
    elif model_type == 'transformer':
        from src.models.Transformer.Transformer_model import TransformerModel
        return TransformerModel(input_shape=input_shape, num_classes=num_classes, learning_rate=1e-4)
    elif model_type in ('bilstm', 'gru'):
        from src.models.RNN.RNN_model import RNNWrapper
        return RNNWrapper(
            input_shape=input_shape, num_classes=num_classes,
            model_type=model_type,
            hidden_dim=info.get('hidden_dim', 128),
            num_layers=info.get('num_layers', 2),
            dropout_rate=info.get('dropout_rate', 0.3),
            bidirectional=info.get('bidirectional', True))
    raise ValueError(f"Unknown model: {model_type}")


# ==================== 3. Data Preparation ====================
def prepare_test_data(test_source, mapper, preprocessor, config, info):
    """Prepare test data."""
    if not os.path.exists(test_source):
        raise FileNotFoundError(f"Test data not found: {test_source}")

    logger.info(f"Loading from: {test_source}")

    from src.data.data_adapter import LogDataAdapter

    adapter = LogDataAdapter(sample_period=config.get('sample_period', 0.02))
    raw_data = adapter.load(test_source)

    if not raw_data:
        raise ValueError(f"No test data loaded from: {test_source}")

    logger.info(f"Loaded {len(raw_data)} files")

    # Label mapping
    for d in raw_data:
        d.raw_labels = mapper.transform(d.raw_labels)

    # Filter
    valid = [d for d in raw_data if preprocessor.validate(d)]
    logger.info(f"Valid: {len(valid)} files")

    if not valid:
        raise ValueError("No valid data after filtering")

    # Standardize
    if preprocessor.scaler:
        for d in valid:
            d.features = preprocessor.scaler.transform(d.features)

    # Step 1: Feature selection
    selected_indices = info.get('selected_indices')
    if selected_indices is None:
        raise ValueError("No selected_indices found in model info")

    logger.info(f"Step 1 - Feature selection: 31 -> {len(selected_indices)} features")
    for d in valid:
        d.features = d.features[:, selected_indices]

    # Step 2: Dimensionality reduction
    dim_reduction_method = info.get('dim_reduction_method')
    dr_model = info.get('dr_model')

    if dim_reduction_method and dr_model is not None:
        logger.info(f"Step 2 - Dimensionality reduction: {dim_reduction_method}")
        from src.features.dim_reducer import DimReducer

        dim_reducer = DimReducer(method=dim_reduction_method, n_components=5)
        dim_reducer.model = dr_model

        for i, d in enumerate(valid):
            n_samples = d.features.shape[0]
            n_features = d.features.shape[1]
            features_3d = d.features.reshape(n_samples, 1, n_features)
            features_reduced = dim_reducer.transform(features_3d)
            d.features = features_reduced.reshape(n_samples, -1)

    final_n_features = valid[0].features.shape[1]
    logger.info(f"Final feature dimension: {final_n_features}")

    # Build sequences
    window_size = config.get('window_size', 128)
    stride = config.get('stride', 2)  # Note: stride=2

    builder = SequenceBuilder(
        window_size=window_size,
        stride=stride,
        min_fault_ratio=config.get('min_fault_ratio', 0.3)
    )

    seqs, labels = [], []
    for d in valid:
        seq_data = builder.build(d)
        if len(seq_data.sequences) > 0:
            seqs.append(seq_data.sequences)
            labels.append(seq_data.labels)

    if not seqs:
        raise ValueError("No sequences built")

    X_test = np.vstack(seqs)  # (n_samples, window_size, n_features)
    y_test = np.concatenate(labels)

    logger.info(f"X_test shape before any transpose: {X_test.shape}")

    return X_test, y_test

# ==================== 4. Evaluation ====================

def evaluate(model, X_test, y_test, mapper, config, model_type, use_rejection):
    """Evaluate model."""
    evaluator = ModelEvaluator(mapper)

    # Get class names
    num_classes = mapper.num_classes
    class_names = [mapper.name_mapping.get(i, f'Class_{i}') for i in range(num_classes)]

    # Evaluate
    metrics, y_pred = evaluator.evaluate(model, X_test, y_test)

    # Create test results directory
    test_vis_dir = get_test_save_dir(model_type)

    # 1. Print evaluation results
    print(f"\n{'=' * 60}")
    print(f"Test Results - {model_type.upper()}")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"{'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 70)
    for cls in range(num_classes):
        cls_str = str(cls)
        if cls_str in metrics['classification_report']:
            m = metrics['classification_report'][cls_str]
            name = mapper.name_mapping.get(cls, f'C{cls}')[:25]
            support_val = m['support']
            print(f"  {name:<25s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                  f"{m['f1-score']:>10.3f} {int(support_val):>10}")

    # 2. Plot confusion matrix using ModelEvaluator
    cm_path = os.path.join(test_vis_dir, f'confusion_matrix_test.svg')
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix - Test Set ({model_type.upper()})",
        save_path=cm_path
    )

    # 3. Plot ROC and PR curves
    try:
        y_proba = model.predict_proba(X_test)
        roc_path = os.path.join(test_vis_dir, f'roc_curve_test.svg')
        pr_path = os.path.join(test_vis_dir, f'pr_curve_test.svg')
        evaluator.plot_roc_curve(y_test, y_proba, model_name=model_type.upper(), save_path=roc_path)
        evaluator.plot_pr_curve(y_test, y_proba, model_name=model_type.upper(), save_path=pr_path)
    except (AttributeError, NotImplementedError):
        logger.warning("Cannot plot ROC/PR curves")

    # 4. Save detailed report
    report_txt_path = os.path.join(test_vis_dir, f'evaluation_report_test.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model Evaluation Report - {model_type.upper()} (Test Set)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Precision (macro): {metrics.get('precision_macro', 0):.4f}\n")
        f.write(f"Recall (macro): {metrics.get('recall_macro', 0):.4f}\n")
        f.write("\nPer-Class Performance:\n" + "-" * 50 + "\n")
        for cls in range(num_classes):
            cls_str = str(cls)
            if cls_str in metrics['classification_report']:
                m = metrics['classification_report'][cls_str]
                name = mapper.name_mapping.get(cls, f'Class_{cls}')
                f.write(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                        f"F1={m['f1-score']:.3f} Support={int(m['support'])}\n")

    results = {'metrics': metrics, 'y_pred': y_pred}

    logger.info(f"All test results saved to: {test_vis_dir}")
    return results

# ==================== 5. Utilities ====================

def list_models():
    """List all available models."""
    if not os.path.exists(MODELS_DIR):
        return []

    available = []
    for name in sorted(os.listdir(MODELS_DIR)):
        d = os.path.join(MODELS_DIR, name)
        if os.path.isdir(d):
            files = os.listdir(d)
            has_model = any(f.endswith(('.pth', '.keras', '.h5')) for f in files)
            has_info = any(f.endswith('_info.pkl') for f in files)
            if has_model and has_info:
                available.append(name)
    return available

# ==================== Main ====================
def run_test(model_identifier: str, test_data: str = None, use_rejection: bool = True):
    """Test a single model."""
    global current_test_data  # used in evaluate()

    if test_data is None:
        # Auto-select first TestCase directory
        data_dir = str(PROJECT_ROOT / 'data')
        test_dirs = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if d.startswith('TestCase') and os.path.isdir(os.path.join(data_dir, d))
        ])
        if test_dirs:
            test_data = test_dirs[0]
            logger.info(f"Auto-selected test data: {test_data}")
        else:
            raise FileNotFoundError(f"No TestCase directory found in: {data_dir}")

    logger.info(f"Test data: {test_data}")
    current_test_data = test_data  # store for evaluate()

    model, info, config, mapper, preprocessor, model_type = load_model_resources(model_identifier)
    X_test, y_test = prepare_test_data(test_data, mapper, preprocessor, config, info)
    return evaluate(model, X_test, y_test, mapper, config, model_identifier, use_rejection)


def main():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--model', type=str, help='Model directory name (e.g., C_RCL-ProtoNet)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--test_data', type=str, default=None, help='Test data path')
    parser.add_argument('--no_rejection', action='store_true', help='Disable rejection')

    args = parser.parse_args()

    if args.list:
        models = list_models()
        print(f"Available: {models}" if models else "No models found")
        return

    if not args.model:
        models = list_models()
        if models:
            print(f"Available: {models}")
        print("Usage: python tests/test_model.py --model <type>")
        return

    run_test(args.model, args.test_data, not args.no_rejection)


if __name__ == '__main__':
    main()
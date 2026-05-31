"""
python tests/test_all_models.py
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from test_model import list_models, load_model_resources, prepare_test_data
from src.evaluation.model_evaluator import ModelEvaluator
from src.config import get_test_save_dir, TEST_COMPARISON_DIR, MODELS_DIR
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_on_test_set(model_identifier):
    """Load model and evaluate on test set."""
    model, info, config, mapper, preprocessor, model_type = load_model_resources(model_identifier)

    test_source = str(Path(__file__).parent.parent / 'data')
    X_test, y_test = prepare_test_data(test_source, mapper, preprocessor, config, info)

    evaluator = ModelEvaluator(mapper)
    metrics, y_pred = evaluator.evaluate(model, X_test, y_test)

    # Compute probability predictions for PR/ROC curves
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    return {
        'model_identifier': model_identifier,
        'model_type': model_type,
        'mapper': mapper,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_true': y_test,
        'X_test': X_test,
        'y_proba': y_proba,
    }

def save_model_results(results):
    """Save complete results for a single model."""
    model_identifier = results['model_identifier']
    model_type = results['model_type']
    m = results['metrics']
    r = {}
    mapper = results['mapper']

    # Use unified path function
    model_test_dir = get_test_save_dir(model_identifier)

    # ========== Plot dataset distribution ==========
    try:
        evaluator = ModelEvaluator(mapper)

        # Read label distribution from metadata.json
        model_dir = os.path.join(MODELS_DIR, model_identifier)
        metadata_path = os.path.join(model_dir, 'metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Read from label_distribution
            label_dist = metadata.get('label_distribution', {})
            train_dist = label_dist.get('train', {})
            val_dist = label_dist.get('val', {})

            # Method 1: use saved label arrays if available
            y_train = None
            y_val = None

            if 'labels' in train_dist and 'labels' in val_dist:
                y_train = np.array(train_dist['labels'])
                y_val = np.array(val_dist['labels'])
                logger.info(f"Using saved labels: train={len(y_train)}, val={len(y_val)}")
            else:
                # Method 2: reconstruct labels from counts
                train_counts = train_dist.get('counts', {})
                val_counts = val_dist.get('counts', {})

                y_train = []
                for cls, count in train_counts.items():
                    y_train.extend([int(cls)] * count)

                y_val = []
                for cls, count in val_counts.items():
                    y_val.extend([int(cls)] * count)

                y_train = np.array(y_train)
                y_val = np.array(y_val)
                logger.info(f"Reconstructed labels from counts: train={len(y_train)}, val={len(y_val)}")

            if y_train is not None and y_val is not None and len(y_train) > 0 and len(y_val) > 0:
                y_test = results['y_true']

                # Plot distribution
                dist_path = os.path.join(model_test_dir, 'dataset_distribution.svg')
                evaluator.plot_dataset_distribution(
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    save_path=dist_path
                )
                logger.info(f"Dataset distribution plot saved to: {dist_path}")
            else:
                logger.warning("Could not reconstruct training/validation labels from metadata")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
    except Exception as e:
        logger.warning(f"Could not plot dataset distribution: {e}")

    # 1. Per-class metrics CSV
    per_class_data = []
    for cls in range(mapper.num_classes):
        cls_str = str(cls)
        if cls_str in m['classification_report']:
            report = m['classification_report'][cls_str]
            per_class_data.append({
                'Class': cls,
                'Class_Name': mapper.name_mapping.get(cls, f'Class_{cls}'),
                'Precision': f"{report['precision']:.4f}",
                'Recall': f"{report['recall']:.4f}",
                'F1_Score': f"{report['f1-score']:.4f}",
                'Support': int(report['support'])
            })

    df_per_class = pd.DataFrame(per_class_data)
    df_per_class.to_csv(os.path.join(model_test_dir, 'per_class_metrics.csv'), index=False)

    # 2. Overall metrics JSON
    overall_metrics = {
        'model': model_identifier,
        'model_type': model_type.upper(),
        'accuracy': float(m['accuracy']),
        'macro_f1': float(m['f1_macro']),
        'weighted_f1': float(m['f1_weighted']),
    }
    with open(os.path.join(model_test_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    # 2.5. Text evaluation report
    report_path = os.path.join(model_test_dir, 'evaluation_report_test.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model Evaluation Report - {model_identifier} (Test Set)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {m['accuracy']:.4f}\n")
        f.write(f"Macro F1: {m['f1_macro']:.4f}\n")
        f.write(f"Weighted F1: {m['f1_weighted']:.4f}\n")
        if r:
            f.write(f"Acceptance Rate: {r.get('acceptance_rate', 0):.4f}\n")
            f.write(f"Accepted Accuracy: {r.get('accepted_accuracy', 0):.4f}\n")
        f.write("\nPer-Class Performance:\n" + "-" * 50 + "\n")
        for cls in range(mapper.num_classes):
            cls_str = str(cls)
            if cls_str in m['classification_report']:
                rep = m['classification_report'][cls_str]
                name = mapper.name_mapping.get(cls, f'Class_{cls}')
                f.write(f"  {name}: P={rep['precision']:.3f} R={rep['recall']:.3f} "
                        f"F1={rep['f1-score']:.3f} Support={int(rep['support'])}\n")
    logger.info(f"Evaluation report saved to: {report_path}")

    # 3. Confusion matrix
    evaluator = ModelEvaluator(mapper)
    evaluator.plot_confusion_matrix(
        results['y_true'], results['y_pred'],
        title=f"Confusion Matrix - Test Set ({model_identifier})",
        save_path=os.path.join(model_test_dir, 'confusion_matrix_test.svg')
    )

    # 4. ROC and PR curves
    try:
        y_proba = results.get('y_proba')
        if y_proba is None:
            model = results.get('model')  # may need external injection
            if model:
                y_proba = model.predict_proba(results['X_test'])
        if y_proba is not None:
            evaluator.plot_roc_curve(
                results['y_true'], y_proba,
                model_name=model_identifier,
                save_path=os.path.join(model_test_dir, 'roc_curve_test.svg')
            )
            evaluator.plot_pr_curve(
                results['y_true'], y_proba,
                model_name=model_identifier,
                save_path=os.path.join(model_test_dir, 'pr_curve_test.svg')
            )
    except Exception:
        pass

    # 5. Rejected sample distribution
    if r and r.get('rejected_classes'):
        rejected_data = []
        for cls, count in sorted(r['rejected_classes'].items(), key=lambda x: -x[1]):
            rejected_data.append({
                'Class': int(cls),
                'Class_Name': mapper.name_mapping.get(int(cls), f'Class_{cls}'),
                'Rejected_Count': count
            })
        df_rejected = pd.DataFrame(rejected_data)
        df_rejected.to_csv(os.path.join(model_test_dir, 'rejected_distribution.csv'), index=False)

    return overall_metrics, per_class_data

def plot_model_comparison(summary_df, save_dir):
    """Plot model comparison charts."""
    import matplotlib.pyplot as plt

    models_list = summary_df['Model'].tolist()
    x = np.arange(len(models_list))
    width = 0.3

    # ========== Chart 1: Basic metrics comparison ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    acc_values = [float(v) for v in summary_df['Accuracy']]
    f1_values = [float(v) for v in summary_df['Macro F1']]

    bars_acc = ax1.bar(x - width / 2, acc_values, width, label='Accuracy', color='#2E86AB', edgecolor='white')
    bars_f1 = ax1.bar(x + width / 2, f1_values, width, label='Macro F1', color='#A23B72', edgecolor='white')

    # Annotate values on bars
    for bar in bars_acc:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars_f1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models_list, fontsize=14, rotation=30, ha='right')
    ax1.set_ylabel('Score', fontsize=14)
    #ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.set_ylim(0, max(max(acc_values), max(f1_values)) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path1 = os.path.join(save_dir, 'model_comparison_metrics.svg')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Performance comparison plot saved to: {save_path1}")

def check_data_distribution():
    """Check data distribution consistency between train and test sets."""
    from collections import Counter
    from test_model import load_model_resources, prepare_test_data
    from pathlib import Path

    # Load one model to get preprocessing parameters
    model_id = 'RCL-ProtoNet'
    model, info, config, mapper, preprocessor, model_type = load_model_resources(model_id)

    # Load test data
    test_source = str(Path(__file__).parent.parent / 'data')
    X_test, y_test = prepare_test_data(test_source, mapper, preprocessor, config, info)

    print("\n" + "=" * 70)
    print("DATA DISTRIBUTION CHECK")
    print("=" * 70)

    # 1. Sample count comparison
    y_train_counts = config.get('training_distribution', {})
    y_test_counts = dict(Counter(y_test))

    print(f"\n1. Sample Count")
    print(f'   Train total: {sum(y_train_counts.values()) if y_train_counts else "N/A"}')
    print(f'   Test total: {len(y_test)}')

    # 2. Class distribution comparison
    print(f"\n2. Class Distribution")
    print(f"   {'Class':<8s} {'Train':>10s} {'Test':>10s} {'Diff':>10s}")
    print(f"   {'-' * 40}")

    all_classes = sorted(set(list(y_train_counts.keys()) + list(y_test_counts.keys())))
    for cls in all_classes:
        train_pct = y_train_counts.get(cls, 0) / sum(y_train_counts.values()) if y_train_counts else 0
        test_pct = y_test_counts.get(cls, 0) / len(y_test)
        diff = abs(train_pct - test_pct)
        flag = "⚠️" if diff > 0.1 else "  "
        print(f"   {cls:<8d} {train_pct:>10.2%} {test_pct:>10.2%} {diff:>10.2%} {flag}")

    # 3. Feature statistics
    print(f"\n3. Feature Statistics (standardized)")
    print(f'   Test X mean: {X_test.mean():.4f}')
    print(f'   Test X std:  {X_test.std():.4f}')
    print(f'   Test X min:  {X_test.min():.4f}')
    print(f'   Test X max:  {X_test.max():.4f}')

    # Note: train statistics not directly saved; check scaler params
    if info.get('scaler_mean') is not None:
        print(f"\n   Train scaler mean (first 5): {info['scaler_mean'][:5]}")
        print(f"   Train scaler std  (first 5): {info['scaler_scale'][:5]}")

    # 4. Window parameter check
    print(f"\n4. 序列参数")
    print(f"   Train window_size: {config.get('window_size', 'N/A')}")
    print(f"   Train stride: {config.get('stride', 'N/A')}")
    print(f"   Test window_size: {config.get('window_size', 'N/A')}")
    print(f"   Test stride: {config.get('stride', 'N/A')}")
    print(f'   Test X shape: {X_test.shape}')


def plot_pr_comparison_overlap(results_dict, save_dir):
    """
    Plot PR curve comparison for high-overlap classes
    Method progression: ProtoNet -> RGuided -> RCL-ProtoNet
    """

    protonet_key = None
    protonet_focal_key = None
    rguided_key = None
    rcl_key = None

    for name in results_dict.keys():
        name_lower = name.lower()
        if 'protonet+focal' in name_lower:
            protonet_focal_key = name
        elif 'protonet' in name_lower and 'rcl' not in name_lower and 'rguided' not in name_lower:
            protonet_key = name
        if 'rguided' in name_lower and 'rcl' not in name_lower:
            rguided_key = name
        if 'rcl' in name_lower:
            rcl_key = name

    missing = []
    if protonet_key is None:
        missing.append('ProtoNet')
    if protonet_focal_key is None:
        missing.append('ProtoNet+Focal')
    if rguided_key is None:
        missing.append('RGuided')
    if rcl_key is None:
        missing.append('RCL-ProtoNet')

    if missing:
        print(f"  Skip PR comparison: {', '.join(missing)} not found. Available: {list(results_dict.keys())}")
        return

    print(f"  PR comparison: {protonet_key} vs {protonet_focal_key} vs {rguided_key} vs {rcl_key}")

    protonet = results_dict[protonet_key]
    protonet_focal = results_dict[protonet_focal_key]
    rguided = results_dict[rguided_key]
    rcl = results_dict[rcl_key]
    mapper = protonet['mapper']
    y_true = protonet['y_true']

    y_true_bin = label_binarize(y_true, classes=range(mapper.num_classes))

    y_proba_protonet = protonet.get('y_proba')
    y_proba_protonet_focal = protonet_focal.get('y_proba')
    y_proba_rguided = rguided.get('y_proba')
    y_proba_rcl = rcl.get('y_proba')

    if any(p is None for p in [y_proba_protonet, y_proba_protonet_focal, y_proba_rguided, y_proba_rcl]):
        print("  Skip PR comparison: missing probability predictions")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ['Class 3 (Gyroscope)', 'Class 5 (Magnetometer)', 'Class 6 (GPS)']

    # Method progression: no guidance -> R-weighted -> contrastive
    configs = [
        (y_proba_protonet,        '#ff7f0e', '-.', 'ProtoNet'),
        (y_proba_protonet_focal,  '#2ca02c', '-',  'ProtoNet$_{Focal}$'),
        (y_proba_rguided,         '#d62728', '--', 'RGuided'),
        (y_proba_rcl,             '#1f77b4', '-',  'RCL-ProtoNet'),
    ]

    for idx, cls in enumerate([3, 5, 6]):
        ax = axes[idx]
        for y_proba, color, ls, label in configs:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, cls], y_proba[:, cls])
            ap = average_precision_score(y_true_bin[:, cls], y_proba[:, cls])
            n = len(recall)
            p_aligned, r_aligned = precision[:n], recall
            if label == 'RCL-ProtoNet':
                mark_every = max(1, n // 6)
                ax.plot(r_aligned, p_aligned, color=color, linestyle=ls, lw=2,
                        marker='o', markevery=mark_every, markersize=6,
                        markeredgewidth=0.5, markeredgecolor='white',
                        label=f'{label} (AP={ap:.3f})')
            else:
                ax.plot(r_aligned, p_aligned, color=color, linestyle=ls, lw=2,
                        label=f'{label} (AP={ap:.3f})')

        ax.set_xlabel('Recall', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title(class_names[idx], fontsize=18, fontweight='bold')
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pr_curves_overlap.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"PR comparison saved to: {save_path}")

def main():
    models = list_models()

    if not models:
        print("No trained models found.")
        return

    test_order = [
        'CNN',
        'Transformer',
        'ResNet',
        'BiLSTM',
        'GRU',
        'ProtoNet',
        'RGuided',
        'RCL-ProtoNet',
        'ProtoNet+Focal',
    ]

    test_models = [m for m in test_order]

    print(f"\n{'=' * 80}")
    print(f"Testing {len(test_models)} models: {test_models}")
    print(f"{'=' * 80}")

    summary = []
    comparison_dir = TEST_COMPARISON_DIR
    os.makedirs(comparison_dir, exist_ok=True)

    # ========== Data distribution check ==========
    check_data_distribution()
    all_result_dict = {}

    for model_id in test_models:
        print(f"\n{'-' * 60}")
        print(f"Testing: {model_id}")
        print(f"{'-' * 60}")

        try:
            results = test_model_on_test_set(model_id)

            try:
                model = load_model_resources(model_id)[0]
                results['y_proba'] = model.predict_proba(results['X_test']) if hasattr(model, 'predict_proba') else None
            except:
                results['y_proba'] = None

            all_result_dict[model_id] = results
            overall, per_class = save_model_results(results)

            m = results['metrics']
            r = results.get('rejection', {})

            summary.append({
                'Model': model_id,
                'Accuracy': f"{m['accuracy']:.4f}",
                'Macro F1': f"{m['f1_macro']:.4f}",
                'Accept Rate': f"{r.get('acceptance_rate', 0):.1%}" if r else 'N/A',
                'Accepted Acc': f"{r.get('accepted_accuracy', 0):.4f}" if r else 'N/A',
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    plot_pr_comparison_overlap(all_result_dict, comparison_dir)


    # Summary
    if summary:
        df = pd.DataFrame(summary)
        csv_path = os.path.join(comparison_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)

        plot_model_comparison(df, str(comparison_dir))

        print(f"\n{'=' * 80}")
        print("MODEL COMPARISON SUMMARY (Test Set)")
        print(f"{'=' * 80}")
        print(df.to_string(index=False))
        print(f"\nComparison results saved to: {comparison_dir}")


if __name__ == '__main__':
    main()
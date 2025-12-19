"""
metrics.py - Comprehensive evaluation metrics for classification
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import json
import os
from typing import List, Dict, Union
import pandas as pd


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_logits: torch.Tensor,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_logits: Model logits for all classes
        class_names: List of class names
    
    Returns:
        Dictionary containing all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = torch.softmax(y_logits, dim=1).numpy()
    
    num_classes = len(class_names)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC and Average Precision (for multi-class)
    if num_classes == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
        avg_precision = average_precision_score(y_true, y_probs[:, 1])
    else:
        # Multi-class classification (one-vs-rest)
        try:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            roc_auc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
            avg_precision = average_precision_score(y_true_bin, y_probs, average='macro')
        except:
            roc_auc = 0.0
            avg_precision = 0.0
    
    # Compile metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
    
    return metrics


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    class_names: List[str],
    save_path: str,
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize counts
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")


def generate_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    class_names: List[str],
    save_path: str = None
) -> str:
    """
    Generate detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save report
    
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(report)
    print("="*70)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Classification report saved to {save_path}")
    
    return report


def plot_per_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    save_path: str
):
    """
    Plot per-class precision, recall, and F1 scores
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: List of class names
        save_path: Path to save figure
    """
    # Extract per-class metrics
    precisions = [metrics.get(f'precision_{name}', 0) for name in class_names]
    recalls = [metrics.get(f'recall_{name}', 0) for name in class_names]
    f1_scores = [metrics.get(f'f1_{name}', 0) for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x, recalls, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1_scores, width, label='F1 Score', color='#e74c3c')
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics plot saved to {save_path}")


def save_metrics_json(metrics: Dict[str, float], save_path: str):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {save_path}")


def generate_metrics_summary(
    metrics: Dict[str, float],
    class_names: List[str]
) -> pd.DataFrame:
    """
    Generate summary DataFrame of metrics
    
    Args:
        metrics: Dictionary containing all metrics
        class_names: List of class names
    
    Returns:
        DataFrame with metrics summary
    """
    # Overall metrics
    overall_metrics = {
        'Metric': [
            'Accuracy', 'Precision (Macro)', 'Precision (Weighted)',
            'Recall (Macro)', 'Recall (Weighted)', 'F1 (Macro)',
            'F1 (Weighted)', 'MCC', 'Cohen\'s Kappa', 'ROC-AUC',
            'Avg Precision'
        ],
        'Score': [
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['precision_weighted'],
            metrics['recall_macro'],
            metrics['recall_weighted'],
            metrics['f1_macro'],
            metrics['f1_weighted'],
            metrics['mcc'],
            metrics['kappa'],
            metrics['roc_auc'],
            metrics['avg_precision']
        ]
    }
    
    df_overall = pd.DataFrame(overall_metrics)
    
    # Per-class metrics
    per_class_data = {
        'Class': class_names,
        'Precision': [metrics.get(f'precision_{name}', 0) for name in class_names],
        'Recall': [metrics.get(f'recall_{name}', 0) for name in class_names],
        'F1 Score': [metrics.get(f'f1_{name}', 0) for name in class_names]
    }
    
    df_per_class = pd.DataFrame(per_class_data)
    
    print("\n" + "="*70)
    print("Overall Metrics")
    print("="*70)
    print(df_overall.to_string(index=False))
    
    print("\n" + "="*70)
    print("Per-Class Metrics")
    print("="*70)
    print(df_per_class.to_string(index=False))
    print("="*70 + "\n")
    
    return df_overall, df_per_class


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics module...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=20, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, 20)
    
    # Create dummy logits
    y_logits = torch.randn(n_samples, n_classes)
    
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_logits, class_names)
    
    # Generate summary
    df_overall, df_per_class = generate_metrics_summary(metrics, class_names)
    
    # Test plotting functions
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        os.path.join(temp_dir, 'confusion_matrix.png')
    )
    
    generate_classification_report(
        y_true, y_pred, class_names,
        os.path.join(temp_dir, 'classification_report.txt')
    )
    
    plot_per_class_metrics(
        metrics, class_names,
        os.path.join(temp_dir, 'per_class_metrics.png')
    )
    
    save_metrics_json(
        metrics,
        os.path.join(temp_dir, 'metrics.json')
    )
    
    print(f"\n✓ All test outputs saved to: {temp_dir}")
    print("✓ Metrics module test completed!")
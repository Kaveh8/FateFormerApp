import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from models import SingleTransformer
from utils.helpers import create_multimodal_model


def compute_confusion_matrices(id, model_config, fold_results, dataset, device):
    """
    Get confusion matrices for each fold and aggregate them.
    Args:
        id (str): Model ID.
        model_config (dict): Model configuration.
        fold_results (list): List of dictionaries containing fold results.
        cls_valid_loader (torch.utils.data.DataLoader): Validation data loader.
        device (str): Device to use.
    Returns:
        list: List of confusion matrices for each fold and the aggregated confusion
            matrix.
    """
    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
            raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    # Initialize an empty confusion matrix for aggregation
    agg_cm = np.zeros((2, 2), dtype=int)
    cms = []

    for i, fold in enumerate(fold_results, 1):
        model_path = fold['best_model_path']
        state_dict = torch.load(model_path)
        val_subset = Subset(dataset, fold['val_idx'])
        cls_valid_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id, **model_config).to(device)
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, bi, y in cls_valid_loader:
                if isinstance(inputs, list):
                    rna= inputs[0].to(device)
                    atac = inputs[1].to(device)
                    flux = inputs[2].to(device)
                    inputs = (rna, atac, flux)
                else:
                    inputs = inputs.to(device)
                bi, y = bi.to(device), y.to(device)

                preds, _ = model(inputs, bi)
                preds = preds.cpu().numpy()
                val_preds.append(preds)
                val_labels.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds).ravel()
        val_labels = np.concatenate(val_labels).ravel()
        
        binary_preds = (val_preds >= 0.5).astype(int)
        # print(f"Fold {i} Confusion Matrix:", val_preds)
        cm = confusion_matrix(val_labels, binary_preds)
        agg_cm += cm
        cms.append(cm)

    cms.append(agg_cm)
    return cms


def compute_metrics_from_confusion_matrix(cm):
    """
    Compute classification metrics from a confusion matrix.
    Args:
        cm (np.array): Confusion matrix.
    Returns:
        dict: Dictionary containing classification metrics.
    """
    # in cm results of 5 folds are saved in a list. compute this metrics for each fold
    # then return the average of them and the std
    metrics_list = []
    for fold_cm in cm[:-1]:  # Exclude the aggregated confusion matrix
        tn, fp, fn, tp = fold_cm.ravel()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        metrics_list.append({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
        })

    avg_metrics = {
        'precision': np.mean([m['precision'] for m in metrics_list]),
        'recall': np.mean([m['recall'] for m in metrics_list]),
        'f1': np.mean([m['f1'] for m in metrics_list]),
        'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
    }

    std_metrics = {
        'precision': np.std([m['precision'] for m in metrics_list]),
        'recall': np.std([m['recall'] for m in metrics_list]),
        'f1': np.std([m['f1'] for m in metrics_list]),
        'accuracy': np.std([m['accuracy'] for m in metrics_list]),
    }

    return {
        'average': avg_metrics,
        'std': std_metrics,
    }
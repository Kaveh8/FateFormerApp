import torch
from utils.helpers import create_multimodal_model
from models import SingleTransformer
from utils.helpers import get_all_modalities_available_samples
from data import create_dataset
import shap

def filter_ds(dataset, indices):
    rna = dataset.rna_data[indices]
    atac = dataset.atac_data[indices]
    flux = dataset.flux_data[indices]
    new_ds = create_dataset.MultiModalDataset((rna, atac, flux), 
                                              dataset.batch_no[indices], 
                                              dataset.labels[indices])
    return new_ds

def get_background_data(id, dataset, samples=100, return_other_samples=False):
    """
    Get background data with balanced samples from each label
    Args:
        dataset: MultiModalDataset object
        samples: Number of samples to get
        return_other_samples: If True, return other samples as well
    Returns:
        new_ds: MultiModalDataset object with background samples
        background_indices: Indices of background samples
        other_ds: MultiModalDataset object with other samples
        other_indices: Indices of other samples
    """
    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    
    if id == 'Multi':
        dataset = get_all_modalities_available_samples(dataset)
        labels = dataset.labels

        # get a balance of samples between labels
        samples_per_label = samples // len(torch.unique(labels))
        background_indices = []
        for label in torch.unique(labels):
            label_indices = torch.where(labels == label)[0]
            background_indices.extend(label_indices[:samples_per_label])
        background_indices = torch.tensor(background_indices)
        background_rna = dataset.rna_data[background_indices]
        background_atac = dataset.atac_data[background_indices]
        background_flux = dataset.flux_data[background_indices]
        bg_ds = create_dataset.MultiModalDataset((background_rna, background_atac, background_flux), 
                                                dataset.batch_no[background_indices], 
                                                dataset.labels[background_indices])
        if return_other_samples:
            # create a new dataset of other samples
            other_indices = torch.tensor([i for i in range(len(labels)) if i not in background_indices])
            other_rna = dataset.rna_data[other_indices]
            other_atac = dataset.atac_data[other_indices]
            other_flux = dataset.flux_data[other_indices]
            other_ds = create_dataset.MultiModalDataset((other_rna, other_atac, other_flux), 
                                                        dataset.batch_no[other_indices], 
                                                        dataset.labels[other_indices])
            return bg_ds, background_indices, other_ds, other_indices
        return bg_ds, background_indices
    else:
        raise ValueError("Not Implemented")
    
class ShapWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        inputs, b = x[:,:-2], x[:,-1].squeeze(-1).long()
        inputs = (inputs[:,:944].long(), inputs[:,944:944+883].float(), inputs[:,944+883:].float())
        preds, _ = self.model(inputs, b)
        preds = torch.sigmoid(preds)
        # print(preds.shape)
        return preds
    
def compute_shap_values(id, fold_results, dataset, model_config, device):
    
    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    
    all_shap_values = []

    if id == 'Multi':
        bg_ds, bg_idx, other_ds, other_idx = get_background_data(id, dataset, samples=50, return_other_samples=True)
        print("total background samples: ", len(bg_idx), "total test samples: ", len(other_idx))
   
    for fold in fold_results:
        val_idx = fold['val_idx']
        # filter val_idx if is in indices
        val_idx = [i for i in val_idx if i in other_idx]

        if len(val_idx) == 0:
            print('No samples of the specified type in the validation set. Skipping...')
            continue
        else:
            print(f'fold {fold["fold"]} -> {len(val_idx)} samples')

        val_ds = filter_ds(dataset, val_idx)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id=id, **model_config).to(device)

        model_path = fold['best_model_path']
        model.load_state_dict(torch.load(model_path))
        model.eval()
        wrapped_model = ShapWrapper(model).to(device)

        bg_x = torch.cat([bg_ds.rna_data, bg_ds.atac_data, bg_ds.flux_data], dim=1).to(device)
        bg_b = bg_ds.batch_no.to(device)
        bgx = torch.cat([bg_x, bg_b[...,None]], dim=-1)
        explainer = shap.GradientExplainer(wrapped_model, bgx)

        inputs, batch_indices = (val_ds.rna_data, val_ds.atac_data, val_ds.flux_data), val_ds.batch_no

        inputs = torch.cat([inputs[0], inputs[1], inputs[2]], dim=1).to(device)
        batch_indices = batch_indices.to(device)
        bgv = torch.cat([inputs, batch_indices[...,None]], dim=-1)
        shap_values = explainer(bgv)
        all_shap_values.append(shap_values)
    
    return all_shap_values
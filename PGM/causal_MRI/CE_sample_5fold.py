### This cf_pa function only return Volume score, not return the MRI counterfatual
import sys
sys.path.append('models/pgm')
sys.path.append('..')
# sys.path.append('../morphomnist')
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from typing import Dict, IO, Optional, Tuple, List
from collections import OrderedDict
import pandas as pd 


from pgm.pgm_trainsetup import * ##(setup_dataloaders etc)
from pgm.pgm_flow import FlowPGM
# from recon_volume import Generate_CF_MRI

## Put data on device
def preprocess(batch, device="cuda:0"):
    batch_new = {}
    for k, v in batch.items():
        try:
            batch_new[k] = batch[k].float().to(device) 
        except:
            continue
    return batch_new

### Take a batch of pa, generate its cf_pa
@torch.no_grad()
def cf_pa(
    pgm: nn.Module, 
    ckpt,
    loader: Dict[str, DataLoader],
    do_pa: Optional[str] = 'diagnosis', ## None, Random is a generative model
    AE_path:[str] = 'path'
) -> Tuple[Tensor, Tensor, Tensor]:
    pgm.load_state_dict(ckpt['model_state_dict'])
    pgm.eval()
    dag_vars = list(pgm.variables.keys())
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    ### take data
    #### 1,000, sex, Age ,diagnosis, svol, Frontal_raw, Insula_raw, Parietal_raw
    ## tensor([[ 1.0000,  0.0000,  2.0328,  0.0000, -1.6151, -1.0800, -1.1767, -1.6311]])
    # Metadata = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata")
    # Residual = torch.load("/home/wpeng/results/Meta_DPM/latents/Residual")

    # loader = tqdm(enumerate(dataloader), total=len(
    #     dataloader), miniters=len(dataloader)//100, mininterval=5)

    step = 1
    pa_np_list = []
    cf_pa_np_list = []
    for bch in loader['test']:
        batch = bch
        bs = batch['age'].shape[0]
        batch = preprocess(batch)
        
        pa_orig = batch['pa']
        # print(f'Original PA shape {pa_orig}')
        pa = {k: v for k, v in batch.items() if k != 'x'}
        pa_np = {key: value.cpu().numpy().flatten()[0] for key, value in pa.items()}

        print(f'we have pa {pa}')
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}
        # do[do_pa] = 0 if pa[do_pa] == 1 else 1

        # print(pa[do_pa])
        do[do_pa] = 1 - pa[do_pa] ## this is more for change the diagnosis

        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
        print(f'After, we have cf_pa {cf_pa}')

        step = step + 1
        cf_pa_np = {key: value.cpu().numpy().flatten()[0] for key, value in cf_pa.items()}

        pa_np_list.append(pa_np)
        cf_pa_np_list.append(cf_pa_np)


    return pa_np_list, cf_pa_np_list

#### Apply to the decoder



### Genate samples by randomly intervention
@torch.no_grad()
def cf_pa_gen(
    pgm: nn.Module, 
    loader: Dict[str, DataLoader],
    do_pa: Optional[str] = None, ## Random is a generative model
) -> Tuple[Tensor, Tensor, Tensor]:
    pgm.eval()
    dag_vars = list(pgm.variables.keys())
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    ### take data
    #### 1,000, sex, Age ,diagnosis, svol, Frontal_raw, Insula_raw, Parietal_raw
    ## tensor([[ 1.0000,  0.0000,  2.0328,  0.0000, -1.6151, -1.0800, -1.1767, -1.6311]])
    # Metadata = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata")
    # Residual = torch.load("/home/wpeng/results/Meta_DPM/latents/Residual")

    # loader = tqdm(enumerate(dataloader), total=len(
    #     dataloader), miniters=len(dataloader)//100, mininterval=5)

    for batch in loader['test']:
        bs = batch['age'].shape[0]
        pa_orig = batch['pa']
        print(pa_orig.shape)
        pa = {k: v for k, v in batch.items() if k != 'x'}
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}
        if do_pa is not None:
            idx = torch.randperm(train_set[do_pa].shape[0])
            do[do_pa] = train_set[do_pa].clone()[idx][:bs]
        else: # random interventions
            while not do:
                for k in dag_vars:
                    if torch.rand(1) > 0.5:  # coin flip to intervene on pa_k
                        idx = torch.randperm(train_set[k].shape[0])
                        do[k] = train_set[k].clone()[idx][:bs]
        do = preprocess(do)
        # infer counterfactual parents
        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)
        
      
    return cf_pa



from scipy.stats import ttest_ind

## regress out head size, etc
# def regress_out(x, y):
#     # Convert x to a column tensor (needed for matrix multiplication)
#     x = x.unsqueeze(1)
#     # Perform linear regression
#     coefficients = torch.linalg.lstsq(x, y).solution
#     # Calculate residuals (subtract predicted values from actual values)
#     residuals = y - torch.matmul(x, coefficients)

#     return residuals

import numpy as np
def regress_out(x, y):
    # Check if x is a NumPy array
    if isinstance(x, np.ndarray):
        # Ensure x is a 2D array (column vector)
        x = np.atleast_2d(x).transpose()
         # Check if dimensions are compatible for linear regression
        if len(x) != len(y):
            print(f'len(x) ={len(x)}   len(y) { len(y)}')
            raise ValueError("Incompatible dimensions: length of x and y must be the same.")
        
        
        # Fit linear regression model
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        
        # Calculate predicted values
        predicted_values = np.dot(x, coefficients)
        
        # Calculate residuals
        residuals = y - predicted_values
    
    # Check if x is a PyTorch tensor
    elif isinstance(x, torch.Tensor):
        # Convert x to a column tensor (needed for matrix multiplication)
        x = x.unsqueeze(1)
        
        # Perform linear regression
        coefficients = torch.linalg.lstsq(x, y).solution
        
        # Calculate residuals (subtract predicted values from actual values)
        residuals = y - torch.matmul(x, coefficients)
    
    else:
        raise ValueError("Unsupported input type. Must be either NumPy array or PyTorch tensor.")
    
    return residuals

# Example usage:
# Assuming ssvol and syn_data are NumPy arrays or PyTorch tensors
# md_residual = regress_out(ssvol, syn_data)


def pvalues_realdata():
    Metadata = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata")

    svol = Metadata[:,4]

    md_residual = regress_out(svol, Metadata)
    # Create boolean masks for part 1: contrl and part 2
    column_3 = Metadata[:, 3]
    mask_part1 = column_3 == 0
    mask_part2 = column_3 == 1

    # Divide the tensor into two parts based on the masks
    contrl = md_residual[mask_part1][:408]
    heavd = md_residual[mask_part2]

    # Perform t-test
    t_statistic, p_value_ttest = ttest_ind(contrl.numpy(), heavd.numpy())
    print(f'Original {p_value_ttest}')

    return pvalue

import torch



def cohens_d(group1, group2):
    # Convert tensors to numpy arrays

    # Calculate means
    mean1 = torch.mean(group1)
    mean2 = torch.mean(group2)
    
    # Calculate standard deviations
    std1 = torch.std(group1, unbiased=True)
    std2 = torch.std(group2, unbiased=True)
    
    # Calculate pooled standard deviation
    pooled_std = torch.sqrt(((len(group1) - 1) * std1 ** 2 + (len(group2) - 1) * std2 ** 2) / (len(group1) + len(group2) - 2))
    
    # Calculate Cohen's d
    cohen_d = (mean1 - mean2) / pooled_std
    
    return cohen_d

def cohens_d(group1, group2):
    # Calculate means for each group
    mean_group1 = torch.mean(group1, dim=0)
    mean_group2 = torch.mean(group2, dim=0)
    
    # Calculate pooled standard deviation
    std_pooled = torch.sqrt(((len(group1) - 1) * torch.std(group1, dim=0, unbiased=True) ** 2 + (len(group2) - 1) * torch.std(group2, dim=0, unbiased=True) ** 2) / (len(group1) + len(group2) - 2))
    
    # Calculate Cohen's d for each pair of groups
    cohen_d = (mean_group1 - mean_group2) / std_pooled
    
    return cohen_d


def compute_pvalue():

    # [1.00, sex, Age ,diagnosis ,svol, Frontal_raw, Insula_raw, Parietal_raw ]
    Metadata = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata")

    svol = Metadata[:,4]
    diag = Metadata[:, 3]
    # print(svol)
    Metadata = Metadata[:,5:]
    mask_ctrl = diag == 0
    mask_hd = diag == 1

    ##826data Metadata
    syn_data_svol = pd.read_csv("./structural/atropos_suptent.csv")
    syn_data = pd.read_csv("./structural/atropos_parc116_to_6rois.csv")
    # Print the column names of the DataFrame
    # print("Column Names:", syn_data_svol.columns)

    syn_data = syn_data[['sri24_parc6_frontal_gm','sri24_parc6_insula_gm','sri24_parc6_parietal_gm']]
    ###Normalize the data
    syn_data_svol = syn_data_svol[['sri24_suptent_supratentorium_volume']]
    mean_values = syn_data_svol.mean()
    std_values = syn_data_svol.std()

    # Define normalization function
    def normalize_column(column):
        return (column - mean_values[column.name]) / std_values[column.name]

    # Apply normalization function to all columns
    syn_data_svol = syn_data_svol.apply(normalize_column)
    ssvol = syn_data_svol['sri24_suptent_supratentorium_volume'].to_numpy()
    ssvol = torch.from_numpy(ssvol)
    # print(ssvol)
    # Compute mean and standard deviation for each column
    mean_values = syn_data.mean()
    std_values = syn_data.std()
    # Apply normalization function to all columns
    syn_data = syn_data.apply(normalize_column)
    syn_data = syn_data.to_numpy()
    syn_data = torch.from_numpy(syn_data)

    svol = torch.cat((svol,ssvol))
    Metadata = torch.cat((Metadata, syn_data))


    md_residual = regress_out(svol, Metadata)
    md_real = md_residual[:826]
    md_syn = md_residual[826:]

    contrl = md_real[mask_ctrl][:408]
    heavd = md_real[mask_hd]

    # Generate a random group of indices
    random_indices = np.random.choice(len(contrl), size=280, replace=False)

    contrl = contrl[random_indices]
    heavd = heavd[random_indices]

    # Perform t-test
    t_statistic, p_value_ttest = ttest_ind(contrl, md_syn)
    cd = cohens_d(contrl, md_syn)
    print(f'Pvalue {p_value_ttest}, cohend {cd}')



batch = 1
# dataloaders = setup_dataloaders(batch, 0)##dataset 1

## Model
pgm = FlowPGM(None)
pgm.cuda()

# ema = EMA(model, beta=0.999)
# ema.cuda()

# file = open(f'./eval.txt', 'a')
cf_pa_np_lists =None
for i in range(5):
    dataloaders = setup_dataloaders(batch, f_id=i, Test=True)##dataset 1
    pgm_path = f'../../checkpoints/checkpoint{i}.pt'
    AE_path = 'logs/test/saved_models/vqgan_ema_500000.th'

    pgm_ckpt = torch.load(pgm_path)
    pa_np_list, cf_pa_np_list = cf_pa(pgm, pgm_ckpt, dataloaders,AE_path=AE_path)
    if cf_pa_np_lists is None:
        cf_pa_np_lists = cf_pa_np_list
    else:
        cf_pa_np_lists = cf_pa_np_lists+cf_pa_np_list

# # Convert list of dictionaries to a DataFrame
df = pd.DataFrame(cf_pa_np_lists)

# Save DataFrame to CSV
df.to_csv('pa_data_synthseg_fr5fold.csv', index=False)

# Convert list of dictionaries to a DataFrame
df = pd.DataFrame(cf_pa_np_list)

# Save DataFrame to CSV
df.to_csv('cf_pa_data_synthseg.csv', index=False)
print(f'Printed two tables')
# file.close()

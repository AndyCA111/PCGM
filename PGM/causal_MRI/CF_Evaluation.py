### This cf_pa function will the MRI counterfatual, not only the Volume score
import sys
sys.path.append('models/pgm')
sys.path.append('..')
# sys.path.append('../morphomnist')
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from recon_volume import Generate_CF_MRI
from typing import Dict, IO, Optional, Tuple, List
from collections import OrderedDict
import pandas as pd 


from pgm_trainsetup import * ##(setup_dataloaders etc)
from pgm_flow import FlowPGM
from recon_volume import Generate_CF_MRI

## PUT data on device
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
    do_pa: Optional[str] = 'age', ## None, Random is a generative model
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
    for bch in loader['test']:
        batch, sex = bch
        bs = batch['age'].shape[0]
        batch = preprocess(batch)
        print(sex)
        name = sex['fname']
        sex = preprocess(sex)
        
        pa_orig = batch['pa']
        print(f'Original PA shape {pa_orig.shape}')
        pa = {k: v for k, v in batch.items() if k != 'x'}
        # randomly intervene on a single parent do(pa_k), pa_k ~ p(pa_k)
        do = {}
        do[do_pa] = 1 + pa[do_pa] ## this is more for change the diagnosis

        cf_pa = pgm.counterfactual(obs=pa, intervention=do, num_particles=1)

        ## Convert dict to tensor
        # Drop key 'b' from my_dict1
        # cf_pa.pop('pa', None)
        # sex.pop('fname', None)
        # Define the desired order [1.00, sex, Age ,diagnosis ,svol, Frontal_raw, Insula_raw, Parietal_raw ]
        order = ['sex', 'age', 'diagnosis', 'svol', 'frontal', 'insula', 'parietal']

        # Merge the dictionaries while preserving the specified order
        tensor_dict = {**sex, **cf_pa}  # Merge dictionaries
        tensor_dict = {key: torch.tensor(value) for key, value in tensor_dict.items()}
        merged_dict = OrderedDict()
        for key in order:
            merged_dict[key] = tensor_dict[key]
        # If you want to stack the tensors along a new dimension, you can use torch.stack
        cf_pa = torch.cat(list(merged_dict.values()),-1)

        ### Generate_CF_MRI(AE_path, residusal, cf_meta, name):
        Generate_CF_MRI(AE_path, batch['x'], cf_pa, step, name)
        step = step + 1
        
    return 

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




@torch.no_grad()
def cf_epoch(
    pgm: nn.Module, 
    predictor: nn.Module, 
    dataloaders: Dict[str, DataLoader],
    do_pa: Optional[str] = None, ## Random is a generative model
    te_cf: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    gen.eval()
    pgm.eval()
    predictor.eval()
    dag_vars = list(pgm.variables.keys())
    preds = {k: [] for k in dag_vars}
    targets = {k: [] for k in dag_vars}
    x_counterfactuals = []
    train_set = copy.deepcopy(dataloaders['train'].dataset.samples)
    loader = tqdm(enumerate(dataloaders['test']), total=len(
        dataloaders['test']), mininterval=0.1)

    for _, batch in loader:
        bs = batch['x'].shape[0]
        batch = preprocess(batch)
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
        _pa = vae_preprocess({k: v.clone() for k, v in pa.items()})
        _cf_pa = vae_preprocess({k: v.clone() for k, v in cf_pa.items()})
        # abduct exogenous noise z
        t_z = t_u = 0.1  # sampling temp

        z = vae.abduct(batch['x'], parents=_pa, t=t_z)
        if vae.cond_prior:
            z = [z[i]['z'] for i in range(len(z))]
        # forward vae with observed parents
        rec_loc, rec_scale = vae.forward_latents(z, parents=_pa)
        # abduct exogenous noise u
        u = (batch['x'] - rec_loc) / rec_scale.clamp(min=1e-12)
        if vae.cond_prior and te_cf:  # g(z*, pa*)
            # infer counterfactual mediator z*
            cf_z = vae.abduct(x=batch['x'], parents=_pa, cf_parents=_cf_pa, alpha=0.65)
            cf_loc, cf_scale = vae.forward_latents(cf_z, parents=_cf_pa)
        else:  # g(z, pa*)
            cf_loc, cf_scale = vae.forward_latents(z, parents=_cf_pa)
        cf_scale = cf_scale * t_u
        cfs = {'x':  torch.clamp(cf_loc + cf_scale * u, min=-1, max=1)}
        cfs.update(cf_pa)
        x_counterfactuals.extend(cfs['x'])
        # predict labels of inferred counterfactuals
        preds_cf = predictor.predict(**cfs)
        for k, v in preds_cf.items():
            preds[k].extend(v)
        # targets are the interventions and/or counterfactual parents
        for k in targets.keys():
            t_k = do[k].clone() if k in do.keys() else cfs[k].clone()
            targets[k].extend(t_k)
    for k, v in targets.items():
        targets[k] = torch.stack(v).squeeze().cpu()
        preds[k] = torch.stack(preds[k]).squeeze().cpu()
    x_counterfactuals = torch.stack(x_counterfactuals).cpu()
    return targets, preds, x_counterfactuals


def eval_cf_loop(
    vae: nn.Module,
    pgm: nn.Module,
    predictor: nn.Module,
    dataloaders: Dict[str, DataLoader],
    file: IO[str],
    total_effect: bool = False,
    seeds: List[int] = [0, 1, 2],
):
    for do_pa in ['svol', 'age', 'diagnosis', None]:  # "None" is for random interventions
        acc_runs = []
        mae_runs = {
            'svol': {'predicted': [], 'measured': []},
            'age': {'predicted': [], 'measured': []}
        }

        for seed in seeds:
            print(f'do({(do_pa if do_pa is not None else "random")}), seed {seed}:')
            assert vae.cond_prior if total_effect else True
            targets, preds, x_cfs = cf_epoch(vae, pgm, predictor, dataloaders, do_pa, total_effect)
            acc = (targets['digit'].argmax(-1).numpy() == preds['digit'].argmax(-1).numpy()).mean()
            print(f'predicted digit acc:', acc)
            # evaluate inferred cfs using true causal mechanisms
            measured = {}
            measured['intensity'] = torch.tensor(get_intensity((x_cfs + 1.0) * 127.5))
            with multiprocessing.Pool() as pool:
                measured['thickness'] = torch.tensor(get_thickness((x_cfs + 1.0) * 127.5, pool=pool, chunksize=250))

            mae = {'thickness': {}, 'intensity': {}}
            for k in ['thickness', 'intensity']:
                min_max = dataloaders['train'].dataset.min_max[k]
                _min, _max = min_max[0], min_max[1]
                preds_k = ((preds[k] + 1) / 2) * (_max - _min) + _min
                targets_k = ((targets[k] + 1) / 2) * (_max - _min) + _min
                mae[k]['predicted'] = (targets_k - preds_k).abs().mean().item()
                mae[k]['measured'] = (targets_k - measured[k]).abs().mean().item()
                print(f'predicted {k} mae:', mae[k]['predicted'])
                print(f'measured {k} mae:', mae[k]['measured'])

            acc_runs.append(acc)
            for k in ['thickness', 'intensity']:
                mae_runs[k]['predicted'].append(mae[k]['predicted'])
                mae_runs[k]['measured'].append(mae[k]['measured'])

            file.write(
                f'\ndo({(do_pa if do_pa is not None else "random")}) | digit acc: {acc}, ' +
                f'thickness mae (predicted): {mae["thickness"]["predicted"]}, ' +
                f'thickness mae (measured): {mae["thickness"]["measured"]}, ' +
                f'intensity mae (predicted): {mae["intensity"]["predicted"]}, ' +
                f'intensity mae (measured): {mae["intensity"]["measured"]} | seed {seed}'
            )
            file.flush()
            gc.collect()

        v = 'Total effect: '+ str(total_effect)
        file.write(
            f'\n{(v if vae.cond_prior else "")}\n' +
            f'digit acc | mean: {np.array(acc_runs).mean()} - std: {np.array(acc_runs).std()}\n' +
            f'thickness mae (predicted) | mean: {np.array(mae_runs["thickness"]["predicted"]).mean()} - std: {np.array(mae_runs["thickness"]["predicted"]).std()}\n' +
            f'thickness mae (measured) | mean: {np.array(mae_runs["thickness"]["measured"]).mean()} - std: {np.array(mae_runs["thickness"]["measured"]).std()}\n' +
            f'intensity mae (predicted) | mean: {np.array(mae_runs["intensity"]["predicted"]).mean()} - std: {np.array(mae_runs["intensity"]["predicted"]).std()}\n' +
            f'intensity mae (measured) | mean: {np.array(mae_runs["intensity"]["measured"]).mean()} - std: {np.array(mae_runs["intensity"]["measured"]).std()}\n'
        )
        file.flush()
    return

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

## Get the name
def pair_the_synthtic():
    dataloaders = setup_dataloaders(1, 1)##dataset 1
    step = 0
    with open('sri_contrl.txt', 'w') as f:
        for bch in dataloaders['test']:
            batch, sex = bch
            fname = sex['fname'][0]
            filename = fname.split('/')[-1].split('.')[0]
            current_name = f'SYN_S{step+1:05d}'
            f.write("%s\t%s\n" % (current_name, filename))
            step = step+1

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


def cf_visualization(model):

    pgm_args.concat_pa = False
    pgm_args.bs = 10
    idx_selection = [10232, 11837, 15437, 15639, 17747, 19347, 22164]
    for idx in idx_selection:
        # idx = torch.randperm(len(dataloaders['test'].dataset))[0]
        batch = dataloaders['test'].dataset.__getitem__(idx)  # 1664
        print(idx)
        pa = {k: v.clone() for k, v in batch.items() if k != 'x'}
        batch = preprocess(batch)
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        fs = 34
        pad = 10
        m, n, s = 3, 8, 4
        fig, ax = plt.subplots(m, n, figsize=(s*n + 1, s*m + 1), facecolor='white',
            gridspec_kw=dict(wspace=-0.0, hspace=0.0,
                top=1. - 0.5 / (s*m + 1), bottom=0.5 / (s*m + 1),
                left=0.5 / (s*n + 1), right=1 - 0.5 / (s*n + 1))
        )
        for i in range(m):
            for j in range(n):
                ax[i,j].axes.xaxis.set_ticks([])
                ax[i,j].axes.yaxis.set_ticks([])

        x = batch['x'].clone()
        x = (x.squeeze().detach().cpu() + 1) * 127.5
        ax[0,-1].set_title('Observation', fontsize=fs, pad=pad+4)
        ax[0,-1].imshow(x, cmap='Greys_r')
        ax[1,-1].axis('off')
        ax[2,-1].axis('off')

        ax[0,0].set_ylabel('Counterfactual', fontsize=fs+4, labelpad=pad)
        ax[1,0].set_ylabel('Direct Effect', fontsize=fs+4, labelpad=pad)
        ax[2,0].set_ylabel('Uncertainty', fontsize=fs+4, labelpad=pad)

        dig_categories = ['Contrl', 'Drinker']  # 0,1
        # finding_categories = ['No disease', 'Pleural Effusion']

        ### "age","svol","diagnosis", "parietal", "frontal", "insula", 
        for j, do_k in enumerate(['age', 'svol', 'diagnosis', 'parietal', "frontal", "insula", 'null']):
            do = {}
            if do_k == 'null': ## Done nothing
                do = {k: v.clone() for k, v in batch.items() if k != 'x'}
            elif do_k == 'diagnosis' :#binary
                do[do_k] = 1 - batch[do_k]
            else:
                do[do_k] = train_samples[do_k].clone()[torch.randperm(n_train)][0:1]
                do = preprocess(norm(do))
            do_ = undo_norm(copy.deepcopy(do))
            if do_k == 'svol':
                msg = str(float(int(do_[do_k].item()*1000)))
                msg = rf'$do($' + f'Svol' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'age':
                msg = str(float(int(do_[do_k].item())))
                msg = rf'$do($' + f'Age' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'parietal':
                msg = str(float(int(do_[do_k].item())))
                msg = rf'$do($' + f'Parietal' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'frontal':
                msg = str(float(int(do_[do_k].item())))
                msg = rf'$do($' + f'Frontal' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'insula':
                msg = str(float(int(do_[do_k].item())))
                msg = rf'$do($' + f'Insula' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'diagnosis':
                msg = dig_categories[int(torch.argmax(do_[do_k], dim=-1))]
                msg = rf'$do($' + f'Diagnosis' + r'${{=}}$' + f'{msg}' + '$)$'
            elif do_k == 'null':
                msg = 'Null Intervention'
                
            ax[0,j].set_title(msg, fontsize=fs-10, pad=pad+4)
            
            if do_k != 'null':
                do = preprocess(do)
                if len(do[do_k].shape) < 2:
                    do[do_k] = do[do_k].unsqueeze(0)

            with torch.no_grad():
                out = model.forward(batch, do, elbo_fn, cf_particles=50)
            
            cf_x = (out['cfs']['x'].squeeze().detach().cpu() + 1) * 127.5
            ax[0,j].imshow(cf_x, cmap='Greys_r')
            _x = cf_x - x  # using x here instead of rec_x since recs are near perfect anyways so direct effect looks the same
            ax[1,j].imshow(_x, cmap='RdBu_r', norm=MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max()))
            _x = out['var_cf_x'].clamp(min=0).sqrt().detach().cpu().squeeze()
            ax[2,j].imshow(_x, cmap='jet')#, norm=MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max()))

            pa_ = undo_norm(copy.deepcopy(pa))
            s = str(sex_categories[int(pa_['sex'].item())])
            r = str(race_categories[int(torch.argmax(pa_['race'], dim=-1))])
            a = str(int(pa_['age'].item()))
            f = str(finding_categories[int(pa_['finding'].item())])

            ax[0,-1].set_xlabel(f'a={a}, s={s}, \n r={r}, d={f}',    
                labelpad=pad+4, fontsize=fs-16, multialignment='center', linespacing=1.6)

            cf_pa = {k: v for k, v in out['cfs'].items() if k != 'x'}
            cf_pa_ = undo_norm(copy.deepcopy(cf_pa))

            cf_s = str(sex_categories[int(cf_pa_['sex'].item())])
            cf_a = str(np.round(cf_pa_['age'].item(), 1))
            cf_r = str(race_categories[int(torch.argmax(cf_pa_['race'], dim=-1))])
            cf_f = str(finding_categories_[int(cf_pa_['finding'].item())])
            ax[2,j].set_xlabel(rf'$\widetilde{{a}}{{=}}{cf_a}, \ \widetilde{{s}}{{=}}{cf_s},$'+'\n'+ rf'$\widetilde{{r}}{{=}}{cf_r}, \ \widetilde{{d}}{{=}}{cf_f}$',    
                labelpad=pad+4, fontsize=fs-16, multialignment='center', linespacing=1.25)
        save_dir = "./single_plot2"
        os.makedirs(save_dir , exist_ok=True)
        plt.savefig(save_dir+f'/{idx}.pdf', bbox_inches='tight', dpi=300)
        plt.close()


batch = 1
dataloaders = setup_dataloaders(batch, 1)##dataset 1

## Model
pgm = FlowPGM(None)
pgm.cuda()

# ema = EMA(model, beta=0.999)
# ema.cuda()

file = open(f'./eval.txt', 'a')
vae_path = '../../checkpoints/checkpoint.pt'
AE_path = 'logs/test/saved_models/vqgan_ema_500000.th'

pgm_ckpt = torch.load(vae_path)
cf_paq = cf_pa(pgm, pgm_ckpt, dataloaders,AE_path=AE_path)

file.close()
import sys
sys.path.append(".")

from tqdm import tqdm
import torch
import nibabel as nib
import numpy as np
import pandas as pd 
from ae.videobase import CausalVAEModel

from utils.util_MRI import save_MRI_samples

from dataloader import get_data_loaders
import os
### help function: read the nii.gz file
def read_nii(path):
    img = nib.load(path)
    data = img.get_fdata()
    filename = str(path)
    max_value = np.percentile(data, 95)
    min_value = np.percentile(data, 5)
    data = np.where(data <= max_value, data, max_value)
    data = np.where(data <= min_value, 0., data)
    data = (data/max_value) * 2 - 1
    img2 = np.ones((144, 176, 145))*(-1)
    img2[3:3+138,:,3:3+138] = data
    img = np.transpose(img2,(2,1,0)) 
    ## also add batch
    data = torch.from_numpy(img[None,None,:,:,:]).float()
    data = data.repeat(1,3, 1, 1, 1)
    return data

# output_dir = './Sora_latents'
# # if not os.path.exists(output_dir):
# #     os.makedirs(output_dir)
### for lab 826 samples
###I have 826 samples from 400 subjects
def get_Lab_metadata(path="/home/wepeng/data/data/MRI_3Set/Lab_data/"):
    # df_SRI, df_NCANDA, df_ADNI = get_metadata(path)
    ##Only need lab data
    file_lab = '800_lab_file.csv'

    # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
    df_lab = pd.read_csv(file_lab, header = 0)
    df_lab['dx_raw'] = df_lab['dx_raw'].replace({"etoh": 1, "ctrl": 0})
    df_lab['sex_raw'] = df_lab['sex_raw'].replace({"F": 1, "M": 0})

    df_lab['subject'] = df_lab['subject'].astype(str)
    df_lab['subject'] = df_lab['subject'].str.zfill(5)
    df_lab['sday_raw'] = df_lab['sday_raw'].astype(str)

    df_lab['fname'] = path + "LAB_S"+ df_lab['subject']+'-'+df_lab['sday_raw']+'.nii.gz'
    print(df_lab['fname'])
    ##Make sure all exsited, 
    rows_to_remove = []
    print(f"There are {df_lab.shape[0]} in ADNI dataset" )

    # Create a boolean mask indicating whether each file exists
    mask = df_lab['fname'].apply(lambda x: os.path.exists(x))
    # Filter the DataFrame to keep only rows where the file exists
    df_lab = df_lab[mask]

    print(f"After remove empty, There are {df_lab.shape[0]} in SRI dataset" )

    return df_lab


if torch.cuda.is_available():
    device = torch.device("cuda")

# model_path1 = './vae_decoder_888_1_1_epoch_26.pth'
model_path1 = './vae_488.pth'
# model = CausalVAEModel.load_from_checkpoint(model_path1, strict=False)
# model.load_state_dict(torch.load(model_path1))
model = CausalVAEModel()
state_dict = torch.load(model_path1, map_location='cpu')
# new_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
# model.load_state_dict(new_state_dict, strict=False)

new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
model.load_state_dict(new_state_dict)
model.requires_grad_(False)

model = model.to(device)

#### 1. This is to extract the latent based on current 
data = get_Lab_metadata()
model.eval()

all_latents = []
with torch.no_grad():
    for idx, row in data.iterrows():
    # for step, batch in enumerate(tqdm(dl)):
        current_latent = []
        # if step > 1000:
        #     break
        columns = ['age_raw','dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
        fpath, age, dx, sex  = row["fname"], row["age_raw"], row["dx_raw"], row["sex_raw"]
        front, parietal, insula = row["Frontal_raw"],row["Parietal_raw"], row["Insula_raw"]
        input = read_nii(fpath)
        input = input.to(device)
        latent = model.encode(input).sample()
        current_latent.append(fpath)
        current_latent.append(latent.cpu())
        current_latent.append(dx)
        current_latent.append(age)
        current_latent.append(sex)
        current_latent.append(front)
        current_latent.append(parietal)
        current_latent.append(insula)
        
        all_latents.append(current_latent)
# all_latents_tensor = torch.cat(all_latents, dim=0)
output_dir = './Lab_metaLatents'
torch.save(all_latents, output_dir)





### 2.  Evaluate the generated samples

Sora_latents = torch.load("Lab_metaLatents")
print(f'the latent {len(Sora_latents)}')
### take one example 
example =  Sora_latents[10]
print(f'exampel {example[0]}, {example[1].shape}, {example[-1]}')
latent = example[1]
with torch.no_grad():
    mri = model.decode(latent.to(device))
print(mri.shape)
save_MRI_samples(mri[:,1:2], 'test_sample')


    

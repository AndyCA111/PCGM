import os
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F

def get_attr_max_min(attr):
    if attr == 'age':
        return 90, 18
    # elif attr == 'brain_volume':
    #     return 1629520, 841919
    # elif attr == 'ventricle_volume':
    #     return 157075, 7613.27001953125
    else:
        NotImplementedError

def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] *2 -1 #[-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            try:
                batch[k] = batch[k].float().unsqueeze(-1)
            except:
                batch[k] = batch[k]
    return batch
### Read metadata from csv_file
class BrainDataset_csv(Dataset):
    def __init__(
        self, 
        csv_file=None, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        use_only_pleural_effusion=True, 
        create_bias=False, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        finding_choice=None,
        only_no_finding=False,
        fold_id = 0,
        test= False
        ):
###   
    # Metadata: None, sex, Age , diagnosis , svol, Frontal_raw, Insula_raw, Parietal_raw 
        data = pd.read_csv("/home/wepeng/codes/causal_MRI/merged_output_pgm_exc_cere_mia.csv") # pd.read_csv(csv_file)
        print("load merged_output_pgm from Braindataset_csv...")
        # Step 2: Replace values in diagnose 'dx_raw' based on conditions
        data['dx_raw'] =  data['dx_raw'].replace({"etoh": 1, "ctrl": 0})
        # Not use this : Specify the columns you want to keep
        #columns = ['age_raw','dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
        columns = ['age_raw','dx_raw',"Frontal_Lobe_Sum", "Insula_Sum", "Parietal_Lobe_Sum", 'Cingulate_Sum', 'Occipital_Sum',  'Temporal_Lobe_Sum',"cerebellum_wm"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
        data = data[columns]
        data = data.dropna()

        
        self.data = data

        self.transform = transform
        self.concat_pa = concat_pa


        self.samples = {
            'age':[],
            # 'svol':[],
            'diagnosis':[],
            # 'x':[],
            'parietal':[],
            'frontal':[],
            'insula':[],
           
            'cingulate':[],
            'temporal':[],
            'occipital':[],
             'whitem': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            
            # img_path = 
            age = self.data.iloc[idx][columns[0]]
            diag = self.data.iloc[idx][columns[1]]
            # svol = self.data[idx][columns[0]]
            frontal = self.data.iloc[idx][columns[2]]
            insula = self.data.iloc[idx][columns[3]]
            parietal = self.data.iloc[idx][columns[4]]

            cingulate = self.data.iloc[idx][columns[5]]
            occipital = self.data.iloc[idx][columns[6]]
            temporal = self.data.iloc[idx][columns[7]]
            whitem = self.data.iloc[idx][columns[8]]

            self.samples['diagnosis'].append(torch.tensor(diag).unsqueeze(0))
            self.samples['age'].append(torch.tensor(age).unsqueeze(0))
            # self.samples['svol'].append(svol.unsqueeze(0))
            self.samples['parietal'].append(torch.tensor(parietal).unsqueeze(0))
            self.samples['frontal'].append(torch.tensor(frontal).unsqueeze(0))
            self.samples['insula'].append(torch.tensor(insula).unsqueeze(0))

            self.samples['cingulate'].append(torch.tensor(cingulate).unsqueeze(0))
            self.samples['temporal'].append(torch.tensor(temporal).unsqueeze(0))
            self.samples['occipital'].append(torch.tensor(occipital).unsqueeze(0))
            self.samples['whitem'].append(torch.tensor(whitem).unsqueeze(0))



    def __len__(self):
        return len(self.samples['age'])

    def __getitem__(self, idx):
        # if not isinstance(idx, int):

        sample = {k: v[idx] for k, v in self.samples.items()}

        
        # sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k,_ in self.samples.items()], dim=0)
        return sample

### Read metadata from csv_file
class BrainDataset_csv_fold(Dataset):
    def __init__(
        self, 
        csv_file=None, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        use_only_pleural_effusion=True, 
        create_bias=False, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        finding_choice=None,
        only_no_finding=False,
        fold_id = 0,
        test= False
        ):
###   
    # Metadata: None, sex, Age , diagnosis , svol, Frontal_raw, Insula_raw, Parietal_raw 
        data = pd.read_csv("/home/wepeng/codes/causal_MRI/merged_output_pgm_exc_cere_mia.csv") # pd.read_csv(csv_file)
        print("load merged_output_pgm from Braindataset_csv...")
        # Step 2: Replace values in diagnose 'dx_raw' based on conditions
        data['dx_raw'] =  data['dx_raw'].replace({"etoh": 1, "ctrl": 0})
        # Not use this : Specify the columns you want to keep
        #columns = ['age_raw','dx_raw',"Frontal_raw", "Insula_raw", "Parietal_raw"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
        columns = ['age_raw','dx_raw',"Frontal_Lobe_Sum", "Insula_Sum", "Parietal_Lobe_Sum", 'Cingulate_Sum', 'Occipital_Sum',  'Temporal_Lobe_Sum',"cerebellum_wm"] # Parietal_Lobe_Sum, Insula_Sum, Frontal_Lobe_Sum
        data = data[columns]
        data = data.dropna()

        ## 5-fold cross validation, train 5 model and put all test fold together to generate p-value
        random_seed = 42
        np.random.seed(random_seed)  # Set numpy random seed
        torch.manual_seed(random_seed)  # Set torch random seed if torch uses any random operations

        # Use KFold from sklearn
        kf = KFold(n_splits=100, shuffle=True, random_state=random_seed)

        # Create the indices for all the folds
        fold_indices = []
        for train_index, val_index in kf.split(data):
            fold_indices.append((train_index, val_index))


        # train_index, val_index = fold_indices[fold_id]
        train_index, test_index = fold_indices[fold_id]
        if test:
            index = test_index
        else:
            index = train_index
        self.data = data.iloc[index]


        self.transform = transform
        self.concat_pa = concat_pa


        self.samples = {
            'age':[],
            # 'svol':[],
            'diagnosis':[],
            # 'x':[],
            'parietal':[],
            'frontal':[],
            'insula':[],
           
            'cingulate':[],
            'temporal':[],
            'occipital':[],
             'whitem': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            
            # img_path = 
            age = self.data.iloc[idx][columns[0]]
            diag = self.data.iloc[idx][columns[1]]
            # svol = self.data[idx][columns[0]]
            frontal = self.data.iloc[idx][columns[2]]
            insula = self.data.iloc[idx][columns[3]]
            parietal = self.data.iloc[idx][columns[4]]

            cingulate = self.data.iloc[idx][columns[5]]
            occipital = self.data.iloc[idx][columns[6]]
            temporal = self.data.iloc[idx][columns[7]]
            whitem = self.data.iloc[idx][columns[8]]

            self.samples['diagnosis'].append(torch.tensor(diag).unsqueeze(0))
            self.samples['age'].append(torch.tensor(age).unsqueeze(0))
            # self.samples['svol'].append(svol.unsqueeze(0))
            self.samples['parietal'].append(torch.tensor(parietal).unsqueeze(0))
            self.samples['frontal'].append(torch.tensor(frontal).unsqueeze(0))
            self.samples['insula'].append(torch.tensor(insula).unsqueeze(0))

            self.samples['cingulate'].append(torch.tensor(cingulate).unsqueeze(0))
            self.samples['temporal'].append(torch.tensor(temporal).unsqueeze(0))
            self.samples['occipital'].append(torch.tensor(occipital).unsqueeze(0))
            self.samples['whitem'].append(torch.tensor(whitem).unsqueeze(0))



    def __len__(self):
        return len(self.samples['age'])

    def __getitem__(self, idx):
        # if not isinstance(idx, int):

        sample = {k: v[idx] for k, v in self.samples.items()}

        
        # sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k,_ in self.samples.items()], dim=0)
        return sample

class BrainDataset(Dataset):
    def __init__(
        self, 
        csv_file=None, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        use_only_pleural_effusion=True, 
        create_bias=False, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        finding_choice=None,
        only_no_finding=False,
        fold_id=0):
###   
    # Metadata: None, sex, Age , diagnosis , svol, Frontal_raw, Insula_raw, Parietal_raw 
        data = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata") # pd.read_csv(csv_file)

        ## 5-fold cross validation, train 5 model and put all test fold together to generate p-value
        random_seed = 42
        np.random.seed(random_seed)  # Set numpy random seed
        torch.manual_seed(random_seed)  # Set torch random seed if torch uses any random operations

        # If necessary, shuffle your data first (optional)
        data = data[torch.randperm(len(data))]

        # Use KFold from sklearn
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        # Convert your tensor data to a numpy array if necessary for KFold
        data_array = data.numpy()

        # Create the indices for all the folds
        fold_indices = []
        for train_index, val_index in kf.split(data_array):
            fold_indices.append((train_index, val_index))


        # train_index, val_index = fold_indices[fold_id]
        train_index, _ = fold_indices[fold_id]
    
        self.data = data[train_index]

        self.transform = transform
        self.concat_pa = concat_pa


        self.samples = {
            'age':[],
            # 'svol':[],
            'diagnosis':[],
            # 'x':[],
            'parietal':[],
            'frontal':[],
            'insula':[],
            # 'path_preproc': [],
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            
            # img_path = 
            age = self.data[idx][2]
            diag = self.data[idx][3]
            svol = self.data[idx][4]
            frontal = self.data[idx][5]
            insula = self.data[idx][6]
            parietal = self.data[idx][7]

            
            # self.samples['path_preproc'].append(img_path)
            # self.samples['x'].append(img_path)
            self.samples['diagnosis'].append(diag.unsqueeze(0))
            self.samples['age'].append(age.unsqueeze(0))
            # self.samples['svol'].append(svol.unsqueeze(0))
            self.samples['parietal'].append(parietal.unsqueeze(0))
            self.samples['frontal'].append(frontal.unsqueeze(0))
            self.samples['insula'].append(insula.unsqueeze(0))

    def __len__(self):
        return len(self.samples['age'])

    def __getitem__(self, idx):
        # if not isinstance(idx, int):

        sample = {k: v[idx] for k, v in self.samples.items()}

        
        # sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k,_ in self.samples.items()], dim=0)
        return sample



#### Take only the contrl data, so we can apply do() to the diagnosis
### Read metadata from csv_file
class BrainDataset_Ctrl(Dataset):
    def __init__(
        self, 
        csv_file=None, 
        transform=None, 
        columns=None, 
        concat_pa=True, 
        use_only_pleural_effusion=True, 
        create_bias=False, 
        select_subgroup=False, 
        race_choice=None, 
        sex_choice=None, 
        finding_choice=None,
        only_no_finding=False):
###   
    # Metadata: None, sex, Age , diagnosis , svol, Frontal_raw, Insula_raw, Parietal_raw 
        self.data = torch.load("/home/wpeng/results/Meta_DPM/latents/Metadata") # pd.read_csv(csv_file)
        Residual = torch.load("/home/wpeng/results/Meta_DPM/latents/Residual") #torch.Size([826, 256, 9, 11, 9])
        Mdata = torch.load("/home/wpeng/results/Meta_DPM/latents/MRI-meta-qLatents")
        fnames = Mdata[0]
        self.transform = transform
        self.concat_pa = concat_pa


        self.samples = {
            'age':[],
            #'svol':[],
            'diagnosis':[],
            'x':[],
            'parietal':[],
            'frontal':[],
            'insula':[],
            # 'path_preproc': [],
        }

        self.sexs = {
            'sex':[],
            'fname':[],
            'svol':[]
        }

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            if self.data[idx][3] == 1:# only for heavy drink
                continue
            if self.data[idx][2] > 0:
                continue
            
            # img_path = 
            age = self.data[idx][2]
            diag = self.data[idx][3]
            svol = self.data[idx][4]
            frontal = self.data[idx][5]
            insula = self.data[idx][6]
            parietal = self.data[idx][7]

            latent = Residual[idx]
            fname = fnames[idx]
            fname = fname.split('/')[-1].split('_')[-1].split('.')[0]


            
            # self.samples['path_preproc'].append(img_path)
            # self.samples['x'].append(img_path)
            self.samples['diagnosis'].append(diag.unsqueeze(0))
            self.samples['age'].append(age.unsqueeze(0))
            # self.samples['svol'].append(svol.unsqueeze(0))
            self.samples['parietal'].append(parietal.unsqueeze(0))
            self.samples['frontal'].append(frontal.unsqueeze(0))
            self.samples['insula'].append(insula.unsqueeze(0))
            self.samples['x'].append(latent)

            sex = self.data[idx][1]
            self.sexs['sex'].append(sex.unsqueeze(0))
            self.sexs['fname'].append(fname)
            self.sexs['svol'].append(svol.unsqueeze(0))


        print(f'There are {len(self.samples["age"])} data in the dataset')

    def __len__(self):
        return len(self.samples['age'])

    def __getitem__(self, idx):
        # if not isinstance(idx, int):

        sample = {k: v[idx] for k, v in self.samples.items()}
        sex =  {k: v[idx] for k, v in self.sexs.items()} 

        
        # sample = norm(sample)
        # print(f'sample: {sample}')
        if self.concat_pa:
            sample['pa'] = torch.cat([sample[k] for k,_ in self.samples.items() if k != 'x'], dim=0)
        return sample, sex

# BrainDataset()

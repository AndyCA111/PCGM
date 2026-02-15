import pandas as pd
import os
from datetime import datetime
import numpy as np 

def func(a, check_values):
    result = np.ones_like(a)*(-1)
    result[np.isin(a, check_values)] = 1
    return result

def get_metadata(path="/home/wpeng/data/MRI_High/", only_control=True):
    seg_path = '/scratch/m000065/binxu/gcp/binxu/wepeng/binxu/binx/synthseg/'
    # path = "/scratch/users/wepeng/data/MRI_Train_DPM/"
    file_adni = path + 'ADNIMERGE.csv'
    file_sri = path + 'sri.csv'
    file_ncanda0 = path + 'ncanda.csv'
    file_ncanda1 = path + 'ncanda_demographics.csv'

    
    file_seg_adni = seg_path + 'new_adni.csv'
    file_seg_ncanda = seg_path + 'new_NCANDA.csv'
    file_seg_sri = seg_path + 'new_LAB.csv'
    
    df_seg_adni = pd.read_csv(file_seg_adni, header = 0)
    df_seg_ncanda = pd.read_csv(file_seg_ncanda, header = 0)
    df_seg_sri = pd.read_csv(file_seg_sri, header = 0)
    # print(df_seg_sri)
    # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
    df_adni = pd.read_csv(file_adni, header = 0)
    df_sri = pd.read_csv(file_sri, header = 0)
    df_ncanda0 = pd.read_csv(file_ncanda0, header = 0)
    df_ncanda1 = pd.read_csv(file_ncanda1, header = 0)


    ###SRI
    # Iterate through the DataFrame and check if the file exists
    print(f"There are {df_sri.shape[0]} in SRI dataset" )
    rows_to_remove = []
    # for index, row in df_sri.iterrows():
    #     fn = 'Lab_data/'+ row['subject']+'-'+row['visit'].split('_')[0]+'.nii.gz'
    #     seg_fn = 'Lab_seg/' + row['subject']+'-'+row['visit'].split('_')[0]+'_synthseg.nii.gz'
    #     path_here = path + fn
    #     seg_path_here = seg_path + seg_fn
    #     if not os.path.exists(path_here) or not os.path.exists(seg_path_here):
    #         rows_to_remove.append(index)
    df_seg_sri_subjects = set(df_seg_sri['subject'])

    for index, row in df_sri.iterrows():
        subject_visit_key = row['subject'] + '-' + row['visit'].split('_')[0]
        if subject_visit_key in df_seg_sri_subjects:
            fn = 'Lab_data/' + subject_visit_key + '.nii.gz'
            seg_fn = 'Lab_padding_seg/' + subject_visit_key + '_synthseg.nii.gz'
            path_here = path + fn
            seg_path_here = seg_path + seg_fn

            if not os.path.exists(path_here) or not os.path.exists(seg_path_here):
                rows_to_remove.append(index)
        else:
            rows_to_remove.append(index)
    df_sri = df_sri.drop(rows_to_remove)

    print(f"After remove empty, There are {df_sri.shape[0]} in SRI dataset" )

    ### Get the metadat of SRI, age, gender, disease， ROI
    #demo_dob and visit time to get age:
    # dignosis: df_sri['demo_diag'] == 'C',HE E Adol
    # sex df_sri['sex'] == 'M'/'F'
    ## Age is hard to get:
    visit_age = pd.to_datetime(df_sri['visit'].str.split('_').str[0], format='%Y%m%d') #df_sri['visit'].split('_')[0] #yyyymmdd 
    birth_age = pd.to_datetime(df_sri['demo_dob'], format='%Y-%m-%d') # #yyyy-mm-dd

    df_sri['age'] = ((visit_age - birth_age) / pd.Timedelta(days=1) / 365).round(3)
    df_sri['fname'] = 'Lab_data/'+ df_sri['subject']+'-'+df_sri['visit'].str.split('_').str[0]+'.nii.gz'
    df_sri['seg_fname'] = 'Lab_padding_seg/'+ df_sri['subject']+'-'+df_sri['visit'].str.split('_').str[0]+'_synthseg.nii.gz'
    ### Output only the Age sex, diagnosis
    df_sri['diagnosis']= df_sri['demo_diag']
    df_sri['site']= 0

    
    
    
    df_SRI = df_sri[['fname', 'age', 'sex', 'diagnosis', 'site', 'seg_fname']]

    df_sri['subject_visit_key'] = df_sri['subject'] + '-' + df_sri['visit'].str.split('_').str[0]

    df_SRI = df_sri[['fname', 'age', 'sex', 'diagnosis', 'site', 'seg_fname', 'subject_visit_key']]

    df_SRI = pd.merge(
        df_SRI, 
        df_seg_sri[['subject', 'Frontal_Lobe_Sum', 'Insula_Sum', 'Parietal_Lobe_Sum', 'Total_Size']], 
        left_on='subject_visit_key',
        right_on='subject',
        how='left'
    )
    df_SRI = df_SRI[
    (df_SRI['Frontal_Lobe_Sum'].between(4000, 200000)) &
    (df_SRI['Insula_Sum'].between(4000, 200000)) &
    (df_SRI['Parietal_Lobe_Sum'].between(4000, 200000))
    ]
    df_SRI = df_SRI.drop(columns=['subject', 'subject_visit_key'])
    # df_SRI = df_SRI[(df_SRI[['Frontal_Lobe_Sum', 'Insula_Sum', 'Parietal_Lobe_Sum', 'Total_Size']] > 0).all(axis=1)]
    # df_SRI = df_SRI.dropna(subset=['Frontal_Lobe_Sum', 'Insula_Sum', 'Parietal_Lobe_Sum', 'Total_Size'])
    # print(df_sri)
    print(df_SRI[:120])
    ###ncandas
    ##Merge the gender info into ncanda
    ## remove repeat subjects:
    print(f"There are {df_ncanda1.shape[0]} in NCANDA dataset" )
    df_ncanda1 = df_ncanda1.drop_duplicates(subset='subject')
    print(f"There are {df_ncanda1.shape[0]} Subject in NCANDA dataset" )



    df_ncanda =pd.merge(df_ncanda0, df_ncanda1[['subject', 'sex']], on='subject', how='left')
    rows_to_remove = []
    print(f"There are {df_ncanda.shape[0]} in NCANDA dataset" )
    df_seg_ncanda_subjects = set(df_seg_ncanda['subject'])
    
    for index, row in df_ncanda.iterrows():
        subject_visit_key = row['subject'] + "_baseline" if int(row['visit'])==0 else row['subject']+ f"_followup_{row['visit']}y"
        if subject_visit_key in df_seg_ncanda_subjects:
            if int(row['visit'])==0:
                fn = 'NCANDA/'+ row['subject']+ "_baseline.nii.gz"
                seg_fn = 'NCANDA_padding_seg/'+ row['subject']+ "_baseline_synthseg.nii.gz"
                # print(fn)
            else:
                fn = 'NCANDA/'+ row['subject']+ f"_followup_{row['visit']}y.nii.gz"
                seg_fn = 'NCANDA_padding_seg/'+ row['subject']+ f"_followup_{row['visit']}y_synthseg.nii.gz"
            path_here = path + fn
            seg_path_here = seg_path + seg_fn
            if not os.path.exists(path_here) or not os.path.exists(seg_path_here):
                rows_to_remove.append(index)
        else:
            rows_to_remove.append(index)
    df_ncanda = df_ncanda.drop(rows_to_remove)
    
    print(f"After remove empty, There are {df_ncanda.shape[0]} in NCANDA dataset" )

    
    ### Get the metadat of SRI, age, gender, disease， ROI
    #demo_dob and visit time to get age:
    # dignosis: df_sri['demo_diag'] == 'C',HE E Adol
    # sex df_sri['sex'] == 'M'/'F'

   # Corrected expression, add file names
    df_ncanda['fname'] = ''
    # Assign 'baseline.nii.gz' to rows where 'visit' is 0, string 0::::  Not working: df_ncanda['visit'] == '0'
    df_ncanda.loc[df_ncanda['visit'].astype(str)=='0', 'fname'] = 'NCANDA/' + df_ncanda['subject'] +'_baseline.nii.gz'
    # Assign follow-up filename to rows where 'visit' is not 0
    df_ncanda.loc[df_ncanda['visit'].astype(int)!=0, 'fname'] = 'NCANDA/' + df_ncanda['subject'] + "_followup_"+df_ncanda['visit'].astype(str)+"y.nii.gz"

    df_ncanda['seg_fname'] = ''
    # Assign 'baseline.nii.gz' to rows where 'visit' is 0, string 0::::  Not working: df_ncanda['visit'] == '0'
    df_ncanda.loc[df_ncanda['visit'].astype(str)=='0', 'seg_fname'] = 'NCANDA_padding_seg/' + df_ncanda['subject'] +'_baseline_synthseg.nii.gz'
    # Assign follow-up filename to rows where 'visit' is not 0
    df_ncanda.loc[df_ncanda['visit'].astype(int)!=0, 'seg_fname'] = 'NCANDA_padding_seg/' + df_ncanda['subject'] + "_followup_"+df_ncanda['visit'].astype(str)+"y_synthseg.nii.gz"

    ### Output only the Age sex, diagnosis
    df_ncanda['diagnosis']= df_ncanda['cahalan']
    df_ncanda['age']= df_ncanda['visit_age']
    df_ncanda['site']= 1

    # df_NCANDA = df_ncanda[['fname', 'age', 'sex', 'diagnosis', 'site', 'seg_fname']]
    
    df_ncanda['subject_visit_key'] = df_ncanda['fname'].str.extract(r'([^/]+)\.nii\.gz')

    df_NCANDA = df_ncanda[['fname', 'age', 'sex', 'diagnosis', 'site', 'seg_fname', 'subject_visit_key']]

    df_NCANDA = pd.merge(
        df_NCANDA, 
        df_seg_ncanda[['subject', 'Frontal_Lobe_Sum', 'Insula_Sum', 'Parietal_Lobe_Sum', 'Total_Size']], 
        left_on='subject_visit_key',
        right_on='subject',
        how='left'
    )
    df_NCANDA = df_NCANDA.drop(columns=['subject', 'subject_visit_key'])
    df_NCANDA = df_NCANDA[
    (df_NCANDA['Frontal_Lobe_Sum'].between(4000, 200000)) &
    (df_NCANDA['Insula_Sum'].between(4000, 200000)) &
    (df_NCANDA['Parietal_Lobe_Sum'].between(4000, 200000))
    ]
    print(df_NCANDA)




    ###ADNI
    rows_to_remove = []
    print(f"There are {df_adni.shape[0]} in ADNI dataset" )
    df_seg_adni_subjects = set(df_seg_adni['subject'])
    for index, row in df_adni.iterrows():
        subject_visit_key = row['PTID'] +'-'+ row['EXAMDATE'] + '-' + 't1'
        # print(subject_visit_key)
        if subject_visit_key in df_seg_adni_subjects:
            fn = 'ADNI_extract/'+ row['PTID'] +'-'+ row['EXAMDATE'] + '-'+ 't1.nii.gz'
            seg_fn = 'ADNI_padding_seg/'+ row['PTID'] +'-'+ row['EXAMDATE'] + '-' +'t1_synthseg.nii.gz'
            path_here = path + fn
            seg_path_here = seg_path + seg_fn
            if not os.path.exists(path_here) or not os.path.exists(seg_path_here):
                rows_to_remove.append(index)
        else:
            rows_to_remove.append(index)
    df_adni = df_adni.drop(rows_to_remove)
    print(f"After remove empty, There are {df_adni.shape[0]} in ADNI dataset" )

    ### Get the metadat of ADNI, age, gender, disease， ROI
    #demo_dob and visit time to get age:
    # dignosis: df_adni['DX_bl'] == 'C',HE E Adol
    # sex df_adni['PTGENDER'] == 'Male'/'Female'
    ## Age :df_adni['AGE']+ df_adni['VISCODE']

    # df_adni['age_visit'] = df_adni['VISCODE'].apply(lambda x: int(x[1:]) / 12)
    df_adni['age_visit'] = df_adni['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x[1:]) / 12)

    df_adni['age'] = df_adni['AGE'] + df_adni['age_visit']

    # Define mapping dictionary
    gender_mapping = {'Male': 'M', 'Female': 'F'}
    # Create new column 'sex' using map
    df_adni['sex'] = df_adni['PTGENDER'].map(gender_mapping)

    df_adni['fname'] = 'ADNI_extract/'+ df_adni['PTID']+'-'+df_adni['EXAMDATE']+'-'+'t1.nii.gz'
    df_adni['seg_fname'] = 'ADNI_padding_seg/'+ df_adni['PTID']+'-'+df_adni['EXAMDATE']+'-'+'t1_synthseg.nii.gz'
    # print(df_adni['fname'])
    ### Output only the Age sex, diagnosis
    df_adni['diagnosis']= df_adni['DX_bl']
    df_adni['site']= 2

    df_adni['subject_visit_key'] = df_adni['PTID'] +'-'+ df_adni['EXAMDATE'] + '-' + 't1'

    
    df_ADNI = df_adni[['fname', 'age', 'sex', 'diagnosis', 'site', 'seg_fname', 'subject_visit_key']]

    df_ADNI = pd.merge(
        df_ADNI, 
        df_seg_adni[['subject', 'Frontal_Lobe_Sum', 'Insula_Sum', 'Parietal_Lobe_Sum', 'Total_Size']], 
        left_on='subject_visit_key',
        right_on='subject',
        how='left'
    )
    df_ADNI = df_ADNI[
    (df_ADNI['Frontal_Lobe_Sum'].between(4000, 200000)) &
    (df_ADNI['Insula_Sum'].between(4000, 200000)) &
    (df_ADNI['Parietal_Lobe_Sum'].between(4000, 200000))
    ]
    df_ADNI = df_ADNI.drop(columns=['subject', 'subject_visit_key'])
    print(df_ADNI)
    unique_values_ADNI = df_adni['PTID'].unique()
    unique_values_SRI = df_sri['subject'].unique()
    unique_values_NCANDA = df_ncanda['subject'].unique()

    # choose control
    df_adni_contrl1 = df_ADNI[df_ADNI['diagnosis'] == 'CN']
    df_sri_contrl1 = df_SRI[df_SRI['diagnosis'] == 'C']
    df_ncanda_contrl1 = df_NCANDA[(df_NCANDA['diagnosis'] == 'control') | (df_NCANDA['diagnosis'] == 'moderate')]

    print(f'there are {df_adni.shape[0]} samples in ADNI ,  and {unique_values_ADNI.shape[0]} subjects in control.')
    print(f'there are {df_sri.shape[0]} samples in SRI,  and {unique_values_SRI.shape[0]} subjects in control.')
    print(f'there are {df_ncanda.shape[0]} samples in NCANDA, and {unique_values_NCANDA.shape[0]} subjects in control.')

    #### To get corresponding subject and metadata:
    ## adni: 'PTID'+ 'EXAMDATE', VISCODE is bl (baseline), m06 (half year visit),...m36(three year); Meta data
    ## ncanda: path shoudl be geting from 'subject' + 'visit', visit==1 mean file name == subject_followup_1y.nii.gz metdadata : visit_age, where is sex?
    ## sri: 'subject'+ 'visit', visit e.g. 20080609_3422_06092008, we only need 20080609 to get the path;
    # file_names_ADNI = ['ADNI_extract/'+df_adni_contrl['PTID'][i]+'/'+ dateTrans(df_adni_contrl['EXAMDATE'][i])+ '/t1.nii.gz' for i in range(df_adni_contrl.shape[0])]
    if only_control:
        return df_sri_contrl1, df_ncanda_contrl1, df_adni_contrl1
    else:
        return df_SRI, df_NCANDA, df_ADNI

###I have 826 samples from 400 subjects
def get_Lab_metadata(path="/home/wpeng/data/MRI_High/"):
    # df_SRI, df_NCANDA, df_ADNI = get_metadata(path)
    ##Only need lab data
    file_lab = path + 'Lab_ctrl_etoh.csv'

    # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
    df_lab = pd.read_csv(file_lab, header = 0)
    df_lab['subject'] = df_lab['subject'].astype(str)
    df_lab['subject'] = df_lab['subject'].str.zfill(5)
    df_lab['sday_raw'] = df_lab['sday_raw'].astype(str)

    df_lab['fname'] = 'SRI/'+ "LAB_S"+ df_lab['subject']+'-'+df_lab['sday_raw']+'.nii.gz'
    print(df_lab['fname'])
    ##Make sure all exsited, 
    rows_to_remove = []
    print(f"There are {df_lab.shape[0]} in ADNI dataset" )

    # Create a boolean mask indicating whether each file exists
    mask = df_lab['fname'].apply(lambda x: os.path.exists(path + x))
    # Filter the DataFrame to keep only rows where the file exists
    df_lab = df_lab[mask]

    print(f"After remove empty, There are {df_lab.shape[0]} in SRI dataset" )

    return df_lab

    ### Get different metadata: age:age_raw,  gender: sex_raw, SOVL:svol_raw ; dignosis: dx_raw, Frontal_raw, Insula_raw,Parietal_ra





# get_Lab_metadata()        



import os

import imageio
import yaml
import torch
import torchvision
from skimage.transform import resize
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)

import numpy as np
import random 
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import shutil
# from mpi4py import MPI
import nibabel as nib
import pickle5 as pickle

from filelock import FileLock

LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600

class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)


video_data_paths_dict = {
    "minerl":       "datasets/minerl_navigate-torch",
    "mazes_cwvae":  "datasets/gqn_mazes-torch",
    "synthseg" : "/scratch/m000065/binxu/gcp/binxu/wepeng/binxu/binx/synthseg/",
    "MRI": "/scratch/m000065/binxu/gcp/wepeng/data/MRI_3Set/",#/home/Nobias/data/hand_orig/ADNI_extract/img_orig_longitudinal/
}

default_T_dict = {
    "minerl":       500,
    "mazes_cwvae":  300,
    "MRI": 64,
}

default_image_size_dict = {
    "minerl":       64,
    "mazes_cwvae":  64,
    "MRI": 128, # Patch in
}


def create_dataset(dataset_name, img_size=176, T=None, prompt='a mri brain image', deterministic=False, num_workers=1, return_dataset=False, ifexample = False,resize = False, age_normalize=False, only_control=True):
    data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if dataset_name == "MRI":
        dataset = MRIDataset(train=True, imgsize=img_size, path=data_path, shard=0, num_shards=0, T=T, prompt=prompt, ifexample = ifexample,resize=resize, age_normalize= age_normalize, only_control=only_control)
    else:
        raise Exception("no dataset", dataset_name)
    return dataset


def get_train_dataset(dataset_name, T=None):
    return create_dataset(
        dataset_name, return_dataset=False, T=T,
        batch_size=None, deterministic=None, num_workers=None
    )


# def get_test_dataset(dataset_name, T=None):
#     data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
#     data_path = data_root / video_data_paths_dict[dataset_name]
#     T = default_T_dict[dataset_name] if T is None else T
#
#     if dataset_name == "MRI":
#         # This is not matter as we set all first slice to all-zeros
#         data_path = "/home/groups/kpohl/t1_data/hand_orig/lab_data/img_orig_longitudinal/"
#         print(f'Sampling is based on the first slice from {data_path}')
#         dataset = MRIDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
#     else:
#         raise Exception("no dataset", dataset_name)
#     dataset.set_test()
#     return dataset


class BaseDataset(Dataset):
    """
    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T, resize):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False
        self.resize = resize
        

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path, age, sex, seg_path, v_f, v_p, v_i, v_total = self.getitem_path_all(idx)
        # path, age, sex, seg_path = self.getitem_path_age(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path)
            seg, segmask_name = self.loaditem_seg(seg_path)
        except Exception as e:
            print(f"Failed on loading {path}")
            raise e
        # video = self.postprocess_video(video)
        # age, sex, site, label = self.get_attributes(idx)
        
        return video, seg, segmask_name, age, sex, v_f, v_p, v_i, v_total
        # return video, seg, segmask_name, age, sex

    def getitem_path(self, idx):
        raise NotImplementedError

    def get_attributes(self, idx):
        raise NotImplementedError

    def loaditem(self, path):
        raise NotImplementedError

    def postprocess_video(self, video):
        raise NotImplementedError

    def cache_file(self, path):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            src_path = self.get_src_path(path)
            with Protect(path):
                shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """ Returns the source path to a file. This function is mainly used to handle SLURM_TMPDIR on ComputeCanada.
            If DATA_ROOT is defined as an environment variable, the datasets are copied to it as they are accessed. This function is called
            when we need the source path from a given path under DATA_ROOT.
        """
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            src_path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, T):
        # print(f'video length with {video.shape}')
        if T is None:
            return video
        if T < len(video):
            # Take a subsequence of the video.
            start_i = 0 if self.is_test else np.random.randint(len(video) - T + 1)
            video = video[start_i:start_i+T]
        assert len(video) == T
        return video



class MRIDataset(BaseDataset):
    def __init__(self, path, train=None, shard=None, num_shards=None, T=None, resize=False, imgsize =176, prompt ='a brain image',ifexample=False,age_normalize=True, only_control=True):
        super().__init__(path=path, T=T, resize=resize)
        seg_path = '/scratch/m000065/binxu/gcp/binxu/wepeng/binxu/binx/synthseg/'
        df_SRI, df_NCANDA, df_ADNI = get_metadata(path=path, only_control=only_control)
        # df_LAB = get_Lab_metadata(path=path)
        ## Try to build file name list
        self.imgsize = imgsize
        self.ifexample = ifexample
        self.seg_path = Path(seg_path)
        self.fnames = df_SRI['fname'].tolist() + df_NCANDA['fname'].tolist() + df_ADNI['fname'].tolist()
        self.seg_fnames = df_SRI['seg_fname'].tolist() + df_NCANDA['seg_fname'].tolist() + df_ADNI['seg_fname'].tolist()
        self.ages = df_SRI['age'].tolist() + df_NCANDA['age'].tolist() + df_ADNI['age'].tolist()
        self.frontal = df_SRI['Frontal_Lobe_Sum'].tolist() + df_NCANDA['Frontal_Lobe_Sum'].tolist() + df_ADNI['Frontal_Lobe_Sum'].tolist()
        self.parietal = df_SRI['Parietal_Lobe_Sum'].tolist() + df_NCANDA['Parietal_Lobe_Sum'].tolist() + df_ADNI['Parietal_Lobe_Sum'].tolist()
        self.insula = df_SRI['Insula_Sum'].tolist() + df_NCANDA['Insula_Sum'].tolist() + df_ADNI['Insula_Sum'].tolist()
        self.total_size = df_SRI['Total_Size'].tolist() + df_NCANDA['Total_Size'].tolist() + df_ADNI['Total_Size'].tolist()
        if age_normalize:
            self.ages = self.z_score_normalize(self.ages)
            self.frontal = self.z_score_normalize(self.frontal)
            self.parietal = self.z_score_normalize(self.parietal)
            self.insula = self.z_score_normalize(self.insula)
        self.sexs = df_SRI['sex'].tolist() + df_NCANDA['sex'].tolist() + df_ADNI['sex'].tolist()
        self.prompt = prompt
        self.prompt_ids = 1

        # self.fnames = file_names_ADNI  #+ + file_names_ADNI #
        files_nt_exist = []
        files_seg_nt_exist = []
        for i in range(len(self.fnames)):
            path_here = path + self.fnames[i]
            seg_path_here = seg_path + self.seg_fnames[i]
            # path_here = path + 'ADNI_extract/' + self.fnames[i]
            # print(path_here)
            if not os.path.isfile(path_here):
                # print(f'data {self.fnames[i]} is not exsit')
                files_nt_exist.append(self.fnames[i])
                files_seg_nt_exist.append(self.seg_fnames[i])
        print(f'Missing {len(files_nt_exist)} MRI samples~')
        print(f'Missing {len(files_seg_nt_exist)} MRI samples~')



        self.fnames = [fname for fname in self.fnames if fname not in files_nt_exist]
        self.seg_fnames = [seg_fname for seg_fname in self.seg_fnames if seg_fname not in files_seg_nt_exist]

        print(f'There are {len(self.fnames)} samples in the training!')


        # print(subject_names)
        
    def z_score_normalize(self, ages):
        mean_age = np.mean(ages)
        std_age = np.std(ages)
        print('mean_age',mean_age)
        print('std_age',std_age)
        normalized_ages = [(age - mean_age) / std_age for age in ages]
        return normalized_ages

    def loaditem(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        filename = str(path)
        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = (data/max_value) * 2 - 1
        
        # data = data[5:5+128,:,5:5+128]

        # img = np.ones((144, 176, 144))*data.min()
        
        # img = np.zeros((138, 176, 138))
        if self.imgsize == 176:
            img2 = np.ones((144, 176, 145))*(-1)
            # img2 = np.ones((144, 192, 144))*(-1)
            img2[3:3+138,:,3:3+138] = data
            img = np.transpose(img2,(2,1,0))    
        elif self.imgsize == 192:
            img2 = np.ones((160, 192, 157))*(-1)
            # img2 = np.ones((144, 192, 144))*(-1)
            img2[11:11+138,8:8+176,9:9+138] = data
            img = np.transpose(img2,(2,1,0))
        else :
            img = np.transpose(data,(2,1,0))
        # # indx_z = torch.randint(img.shape[2]-48, (1,))
        # # # data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # # img = img[:, :, indx_z:indx_z+48]
        # nan_mask = np.isnan(img) # Remove NaN
        # img[nan_mask] = LOW_THRESHOLD
        # img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])
        # valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
        # img = img[valid_plane_i,:,:]
        if self.resize:
            img = resize(img, (128, 128, 128), mode='constant', cval=-1)

        data = th.from_numpy(img[None,:,:,:]).float()
        data = data.repeat(3, 1, 1, 1)
        # if self.resize:
        #     assert data.shape == (1, 128, 128, 128)
        # if self.imgsize == 176:
        #     assert data.shape == (1, 145, 176, 144)
        # if self.imgsize == 192:
        #     assert data.shape == (1, 157, 192, 160)   
        # print(f'this is shape {data.shape}')
        example = {
            "pixel_values": data,
            "prompt_ids": self.prompt_ids,
            "path": filename
        }
        if self.ifexample:
            return example
        else: 
            return data
        
        
    def loaditem_seg(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        filename = str(path)
        # max_value = np.percentile(data, 95)
        # min_value = np.percentile(data, 5)
        # data = np.where(data <= max_value, data, max_value)
        # data = np.where(data <= min_value, 0., data)
        # data = (data/max_value) * 2 - 1
        
        # data = data[5:5+128,:,5:5+128]

        # img = np.ones((144, 176, 144))*data.min()
        selected_list1 = [2, 41] # white matter
        selected_list2 = [5, 44, 4, 43, 28, 60, 14, 15] # total ventricle
        selected_list3 = [7, 46] # cerebellum white matter
        selected_list4 = [8, 47] # cerebellum cortex
        selected_list5 = [1002, 2002, 1003, 2032, 1032, 2003, 2028, 2027, 1028, 1027, 1012, 2012, 2014, 1014, 1020, 2020, 1018, 2018, 1019, 2019, 1024, 2024] # frontal lobe
        selected_list6 = [1003, 2032, 1032, 2003, 2028, 2027, 1028, 1027, 1012, 2012, 2014, 1014, 1020, 2020, 1018, 2018, 1019, 2019, 1024, 2024, 2, 41] # wm and fl
        selected_list7 = [26, 58, 17, 53, 18, 54, 11, 50, 52, 51, 12, 13, 11, 10, 49, 8, 47] # some other
        selected_list8 = [1.001e+03, 1.002e+03, 1.003e+03, 1.005e+03, 1.006e+03,
                           1.007e+03, 1.008e+03, 1.009e+03, 1.010e+03, 1.011e+03, 1.012e+03,
                           1.013e+03, 1.014e+03, 1.015e+03, 1.016e+03, 1.017e+03, 1.018e+03,
                           1.019e+03, 1.020e+03, 1.021e+03, 1.022e+03, 1.023e+03, 1.024e+03,
                           1.025e+03, 1.026e+03, 1.027e+03, 1.028e+03, 1.029e+03, 1.030e+03,
                           1.031e+03, 1.032e+03, 1.033e+03, 1.034e+03, 1.035e+03, 2.001e+03,
                           2.002e+03, 2.003e+03, 2.005e+03, 2.006e+03, 2.007e+03, 2.008e+03,
                           2.009e+03, 2.010e+03, 2.011e+03, 2.012e+03, 2.013e+03, 2.014e+03,
                           2.015e+03, 2.016e+03, 2.017e+03, 2.018e+03, 2.019e+03, 2.020e+03,
                           2.021e+03, 2.022e+03, 2.023e+03, 2.024e+03, 2.025e+03, 2.026e+03,
                           2.027e+03, 2.028e+03, 2.029e+03, 2.030e+03, 2.031e+03, 2.032e+03,
                           2.033e+03, 2.034e+03, 2.035e+03,] # cortex 
        selected_list9 = [1008, 1029, 1021, 1030, 1025, 1017, 2008, 2029, 2021, 2030, 2025, 2017] #patriel lobe
        selected_list10 = [2.035e+03, 1035] # insula
        selected_list11 = [1.014e+03, 1.015e+03, 1.016e+03, 1.017e+03, 1.018e+03,
                           1.019e+03, 1.020e+03, 1.021e+03, 1.022e+03, 1.023e+03, 1.024e+03,
                           1.025e+03, 1.026e+03, 1.027e+03, 1.028e+03, 1.029e+03, 1.030e+03,
                           1.031e+03, 1.032e+03, 1.033e+03, 1.034e+03, 1.035e+03, 2.001e+03,
                           2.002e+03, 2.003e+03, 2.005e+03, 2.006e+03, 2.007e+03, 2.008e+03,
                           2.009e+03, 2.010e+03, 26, 58, 17, 36]
        selected_frontal = [1002, 1003, 1012, 1014, 1018, 1019, 1020, 1026, 1027, 1028, 1032, 2002, 2003, 2012, 2014, 2018, 2019, 2020, 2026, 2027, 2028, 2032]
        selected_insula = [1035, 2035]
        selected_p = [1008, 1029, 1022, 1025, 1031, 2008, 2029, 2022, 2025, 2031]
        selected_fpi = [1008, 1029, 1022, 1025, 1031, 2008, 2029, 2022, 2025, 2031, 1035, 2035, 1002, 1003, 1012, 1014, 1018, 1019, 1020, 1026, 1027, 1028, 1032, 2002, 2003, 2012, 2014, 2018, 2019, 2020, 2026, 2027, 2028, 2032]
        Frontal = [1003, 1012, 1014, 1018, 1019, 1020, 1024, 1027, 1028, 1032, 2003, 2012, 2014, 2018, 2019, 2020, 2024, 2027, 2028, 2032]
        Insula = [1035, 2035]
        Parietal = [1008, 1029, 1022, 1025, 1031, 2008, 2029, 2022, 2025, 2031]
        f_i_p = [1003, 1012, 1014, 1018, 1019, 1020, 1024, 1027, 1028, 1032, 2003, 2012, 2014, 2018, 2019, 2020, 2024, 2027, 2028, 2032 ,1008, 1029, 1022, 1025, 1031, 2008, 2029, 2022, 2025, 2031,1035, 2035]
        Cingulate = [1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026]
        Occipital = [1005, 1011, 1013, 1021, 2005, 2011, 2013, 2021]
        Temporal = [1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034, 2006, 2007, 2009, 2015, 2016, 2030, 2033, 2034]
        c_o_t = [1005, 1011, 1013, 1021, 2005, 2011, 2013, 2021, 1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034, 2006, 2007, 2009, 2015, 2016, 2030, 2033, 2034]
        f_i_p_c_o_t = [1003, 1012, 1014, 1018, 1019, 1020, 1024, 1027, 1028, 1032, 2003, 2012, 2014, 2018, 2019, 2020, 2024, 2027, 2028, 2032 ,1008, 1029, 1022, 1025, 1031, 2008, 2029, 2022, 2025, 2031,1035, 2035,1005, 1011, 1013, 1021, 2005, 2011, 2013, 2021, 1002, 1010, 1023, 1026, 2002, 2010, 2023, 2026, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034, 2006, 2007, 2009, 2015, 2016, 2030, 2033, 2034]
        selected_lists = {
            "white matter": selected_list1,
            "total ventricle": selected_list2,
            "cerebellum white matter": selected_list3,
            "cerebellum cortex": selected_list4,
            "frontal lobe": selected_list5,
            "frontal lobe and white matter": selected_list6,
            "some other": selected_list7,
            "cortex": selected_list8,
            "parietal lobe": selected_list9,
            "insula": selected_list10,
            "random": selected_list11
        }
        new_selected_lists = {
            "frontal_lobe": selected_frontal,
            "parietal_lobe": selected_p,
            "insula": selected_insula,
            "comb": selected_fpi
        }
        new_selected_lists = {
            "frontal_lobe": Frontal,
            "parietal_lobe": Parietal,
            "insula": Insula,
            "cingulate": Cingulate,
            "temporal": Temporal,
            "Occipital": Occipital,
            "comb1": c_o_t,
            "comb2": f_i_p,
            "comb3": f_i_p_c_o_t,
        }
        
        # new_selected_lists = {
        #     "frontal_lobe": selected_frontal,
        #     "parietal_lobe": selected_p,
        #     "insula": selected_insula
        # }
        # img = np.zeros((138, 176, 138))
        if self.imgsize == 176:
            # img2 = np.ones((144, 176, 145))*0
            # # img2 = np.ones((144, 192, 144))*(-1)
            # img2[3:3+138,:,3:3+138] = data
            img = np.transpose(data,(2,1,0))    
        elif self.imgsize == 192:
            # img2 = np.ones((160, 192, 157))*0
            # # img2 = np.ones((144, 192, 144))*(-1)
            # img2[11:11+138,8:8+176,9:9+138] = data
            img = np.transpose(data,(2,1,0))
        else :
            img = np.transpose(data,(2,1,0))
        # # indx_z = torch.randint(img.shape[2]-48, (1,))
        # # # data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # # img = img[:, :, indx_z:indx_z+48]
        # nan_mask = np.isnan(img) # Remove NaN
        # img[nan_mask] = LOW_THRESHOLD
        # img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])
        # valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
        # img = img[valid_plane_i,:,:]
        if self.resize:
            img = resize(img, (128, 128, 128), mode='constant', cval=-1)

        selected_name, selected_values = random.choice(list(new_selected_lists.items()))
    
        processed_data = func(img, selected_values)
        data = th.from_numpy(processed_data[None,:,:,:]).float()
        # data = data.repeat(3, 1, 1, 1)
        # if self.resize:
        #     assert data.shape == (1, 128, 128, 128)
        # if self.imgsize == 176:
        #     assert data.shape == (1, 145, 176, 144)
        # if self.imgsize == 192:
        #     assert data.shape == (1, 157, 192, 160)   
        # print(f'this is shape {data.shape}')
        return data, selected_name
    
    def loaditem2(self, path):
        img = nib.load(path)
        data = img.get_fdata()

        max_value = np.percentile(data, 98)
        data = np.where(data <= max_value, data, max_value)
        data = data/max_value

        # img = np.ones((144, 176, 138))*data.min()
        # img[3:3+138] = data

        ## Random crop a volume
        indx_x = torch.randint(data.shape[0]-64, (1,))
        indx_y = torch.randint(data.shape[1]-64, (1,))
        indx_z = torch.randint(data.shape[2]-128, (1,))
        data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # data = img[:, :, indx_z:indx_z+128]
        data = th.from_numpy(data[None,:,:,:]).float()
        return data
    
    def getitem_path_age(self, idx):
        return self.path / self.fnames[idx], self.ages[idx], self.sexs[idx], self.seg_path / self.seg_fnames[idx], 
    
    def getitem_path_all(self, idx):
        return self.path / self.fnames[idx], self.ages[idx], self.sexs[idx], self.seg_path / self.seg_fnames[idx], self.frontal[idx], self.parietal[idx], self.insula[idx], self.total_size[idx]
    def get_attributes(self, idx):
        # there are age sex etc information in the data
        # 5, 6 are For age and gender, 1 for label, label in range []
        return self.All_datas[idx][5], self.All_datas[idx][6], self.All_datas[idx][7], self.All_datas[idx][1]

    import torch.nn.functional as F
    def postprocess_video(self, video):
        o, h, w, t = video.shape
        images = []
        # This will be used when I directly generate a 3D volume
        img = F.interpolate(video[None], size= 128)
        return img.squeeze(0)
        # The following are for slices
        for i in range(t):
            img = F.interpolate(video[None, :,:,:,i], size= 128)
            images.append(img)
        return th.cat(images) #-1 + 2 * (video.permute(0, 3, 1, 2).float()/255)

    def __len__(self):
        return len(self.fnames)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths
def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=4,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=True,
    prompt = 'a mri brain image',
    ifreturn_loader=True,
    ifexample = False,
    resize = False,
    age_normalize =True,
    only_control = True,
):

    dataset = create_dataset(dataset_name, img_size=img_size, prompt=prompt, ifexample = ifexample,resize=resize,age_normalize=age_normalize, only_control=only_control)
    train_dataset, val_dataset = train_val_split(dataset=dataset, train_val_ratio=train_val_split_ratio)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last
    )
    if get_val_dataloader:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            num_workers=num_workers,
            sampler=None,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last
        )
    else:
        val_loader = None
    if ifreturn_loader:
        return train_loader, val_loader
    else:
        return dataset


    # B, T, C, H, W = x.shape
    # time_t = time_t.view(B, 1).expand(B, T)
    # indicator_template = th.ones_like(x[:, :, :1, :, :])
    # obs_indicator = indicator_template * condition_mask
    # x = th.cat([x*(1-condition_mask) + x0*condition_mask,obs_indicator],dim=2)
if __name__ == "__main__":
    a, b = get_data_loaders("MRI", 176, 1, ifreturn_loader=True, ifexample = False, resize=False)
    print(len(a))




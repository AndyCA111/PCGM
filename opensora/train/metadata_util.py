import pandas as pd
import os
from datetime import datetime
import numpy as np 

def get_metadata(path="/home/wpeng/data/MRI_High/"):

    # path = "/scratch/users/wepeng/data/MRI_Train_DPM/"
    file_adni = path + 'ADNIMERGE.csv'
    file_sri = path + 'sri.csv'
    file_ncanda0 = path + 'ncanda.csv'
    file_ncanda1 = path + 'ncanda_demographics.csv'

    # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
    df_adni = pd.read_csv(file_adni, header = 0)
    df_sri = pd.read_csv(file_sri, header = 0)
    df_ncanda0 = pd.read_csv(file_ncanda0, header = 0)
    df_ncanda1 = pd.read_csv(file_ncanda1, header = 0)

    # ### Choose the control
    # df_adni_contrl = df_adni[df_adni['DX_bl'] == 'CN']
    # df_sri_contrl = df_sri[df_sri['demo_diag'] == 'C']
    # df_ncanda_contrl = df_ncanda[(df_ncanda['cahalan'] == 'control') | (df_ncanda['cahalan'] == 'moderate')]
    # # print(f'there are {df_adni_contrl.shape[0]} samples in ADNI , {df_sri_contrl.shape[0]} in control and {df_ncanda_contrl.shape[0]} subjects in control.')
    

    # # ###remove subjects which are in test
    # # df_sri_contrl = df_sri_contrl1[~df_sri_contrl1['subject'].str.contains('|'.join(test_datas))]
    # # df_ncanda_contrl = df_ncanda_contrl1[~df_ncanda_contrl1['subject'].str.contains('|'.join(test_datas))]
    # # df_adni_contrl = df_adni_contrl1[~df_adni_contrl1['PTID'].str.contains('|'.join(test_datas))]


    ###SRI
    # Iterate through the DataFrame and check if the file exists
    print(f"There are {df_sri.shape[0]} in SRI dataset" )
    rows_to_remove = []
    for index, row in df_sri.iterrows():
        fn = 'Lab_data/img_orig_longitudinal/'+ row['subject']+'-'+row['visit'].split('_')[0]+'.nii.gz'
        path_here = path + fn
        if not os.path.exists(path_here):
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
    df_sri['fname'] = 'SRI/'+ df_sri['subject']+'-'+df_sri['visit'].str.split('_').str[0]+'.nii.gz'
    ##Drop not controls
    df_sri = df_sri[df_sri['demo_diag'] == 'C']
    ### Output only the Age sex, diagnosis
    df_sri['diagnosis']= df_sri['demo_diag']
    df_sri['site']= 0

    df_SRI = df_sri[['fname', 'age', 'sex', 'diagnosis', 'site']]


    ###ncandas
    ##Merge the gender info into ncanda
    ## remove repeat subjects:
    print(f"There are {df_ncanda1.shape[0]} in NCANDA dataset" )
    df_ncanda1 = df_ncanda1.drop_duplicates(subset='subject')
    print(f"There are {df_ncanda1.shape[0]} Subject in  dataset" )



    df_ncanda =pd.merge(df_ncanda0, df_ncanda1[['subject', 'sex']], on='subject', how='left')
    rows_to_remove = []
    print(f"There are {df_ncanda.shape[0]} in NCANDA dataset" )
    for index, row in df_ncanda.iterrows():
        if int(row['visit'])==0:
            fn = 'NCANDA/'+ row['subject']+ "_baseline.nii.gz"
            # print(fn)
        else:
            fn = 'NCANDA/'+ row['subject']+ f"_followup_{row['visit']}y.nii.gz"
        path_here = path + fn
        if not os.path.exists(path_here):
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

    ##drop not contrl
    df_ncanda = df_ncanda[(df_ncanda['cahalan'] == 'control') | (df_ncanda['cahalan'] == 'moderate')]

    ### Output only the Age sex, diagnosis
    df_ncanda['diagnosis']= df_ncanda['cahalan']
    df_ncanda['age']= df_ncanda['visit_age']
    df_ncanda['site']= 1

    df_NCANDA = df_ncanda[['fname', 'age', 'sex', 'diagnosis', 'site']]
    print(df_NCANDA)




    ###ADNI
    rows_to_remove = []
    print(f"There are {df_adni.shape[0]} in ADNI dataset" )

    for index, row in df_adni.iterrows():
        fn = 'ADNI/'+ row['PTID'] +'/'+ row['EXAMDATE'] + '/t1.nii.gz'
        path_here = path + fn
        if not os.path.exists(path_here):
            rows_to_remove.append(index)
    df_adni = df_adni.drop(rows_to_remove)
    print(f"After remove empty, There are {df_adni.shape[0]} in ADNI dataset" )

    ### Get the metadat of ADNI, age, gender, disease， ROI
    #demo_dob and visit time to get age:
    # dignosis: df_adni['DX_bl'] == 'CN',
    # sex df_adni['PTGENDER'] == 'Male'/'Female'
    ## Age :df_adni['AGE']+ df_adni['VISCODE']

    # df_adni['age_visit'] = df_adni['VISCODE'].apply(lambda x: int(x[1:]) / 12)
    df_adni['age_visit'] = df_adni['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x[1:]) / 12)

    df_adni['age'] = df_adni['AGE'] + df_adni['age_visit']

    # Define mapping dictionary
    gender_mapping = {'Male': 'M', 'Female': 'F'}
    # Create new column 'sex' using map
    df_adni['sex'] = df_adni['PTGENDER'].map(gender_mapping)

    df_adni['fname'] = 'ADNI/'+ df_adni['PTID']+'/'+df_adni['EXAMDATE']+'/t1.nii.gz'
    print(df_adni['fname'])
    ###Only keep controls
    df_adni = df_adni[df_adni['DX_bl'] == 'CN']
    ### Output only the Age sex, diagnosis
    df_adni['diagnosis']= df_adni['DX_bl']
    df_adni['site']= 2

    df_ADNI = df_adni[['fname', 'age', 'sex', 'diagnosis', 'site']]




    unique_values_ADNI = df_adni['PTID'].unique()
    unique_values_SRI = df_sri['subject'].unique()
    unique_values_NCANDA = df_ncanda['subject'].unique()



    print(f'there are {df_adni.shape[0]} samples in ADNI ,  and {unique_values_ADNI.shape[0]} subjects in control.')
    print(f'there are {df_sri.shape[0]} samples in SRI,  and {unique_values_SRI.shape[0]} subjects in control.')
    print(f'there are {df_ncanda.shape[0]} samples in NCANDA, and {unique_values_NCANDA.shape[0]} subjects in control.')

    #### To get corresponding subject and metadata:
    ## adni: 'PTID'+ 'EXAMDATE', VISCODE is bl (baseline), m06 (half year visit),...m36(three year); Meta data
    ## ncanda: path shoudl be geting from 'subject' + 'visit', visit==1 mean file name == subject_followup_1y.nii.gz metdadata : visit_age, where is sex?
    ## sri: 'subject'+ 'visit', visit e.g. 20080609_3422_06092008, we only need 20080609 to get the path;
    # file_names_ADNI = ['ADNI/'+df_adni_contrl['PTID'][i]+'/'+ dateTrans(df_adni_contrl['EXAMDATE'][i])+ '/t1.nii.gz' for i in range(df_adni_contrl.shape[0])]
    return df_SRI, df_NCANDA, df_ADNI


###I have 826 samples from 400 subjects
def get_Lab_metadata(path="/home/wpeng/data/MRI_High/"):
    # df_SRI, df_NCANDA, df_ADNI = get_metadata(path)
    ##Only need lab data
    file_lab = path + 'sri.csv'

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


import pandas as pd

df_SRI, df_NCANDA, df_ADNI = get_metadata("/scratch/project_2001654/Wpeng/data/MRI_Three/MRI_High/")
# df_lab = get_Lab_metadata("/scratch/project_2001654/Wpeng/data/MRI_Three/MRI_High/")

combined_df = pd.concat([df_SRI, df_NCANDA, df_ADNI], ignore_index=True)

output_path = "./combined_metadata.csv"
combined_df.to_csv(output_path, index=False)

print(f"Combined DataFrame has been saved to {output_path}")

# get_Lab_metadata()        
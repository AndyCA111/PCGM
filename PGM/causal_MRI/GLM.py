import torch
import numpy as np

## normalize a list
def norm_list(data):
    # Calculate mean and standard deviation
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

    # Normalize the list to z-scores
    normalized_data = [(x - mean) / std_dev for x in data]

    return normalized_data

# ## change the classes into one-hot coding
def convert_metadataC(Metadata):
    Metadata_New = []

    
    # sex: F 0, M 1
    sex = [0 if gender == 'F' else 1 for gender in Metadata[2]]
    Metadata_New.append(sex)
    ##age, norm
    age = norm_list(Metadata[1])
    Metadata_New.append(age)
    # ## diagnosis: ctrl and etoh
    # diag = [1 if gender == 'etoh' else 0 for gender in Metadata[4]]## diagnosis
    # Metadata_New.append(diag)

    return Metadata_New

## for combined dataset, here we only take age and gender
def convert_metadata(Metadata):
    Metadata_New = []

    
    # sex: F 0, M 1
    sex = [0 if gender == 'F' else 1 for gender in Metadata[2]]
    Metadata_New.append(sex)
    ##age, norm
    age = norm_list(Metadata[1])
    Metadata_New.append(age)


    ### ROIs: svol_raw, Frontal_raw Insula_raw Parietal_raw
    svol_raw = norm_list(Metadata[3])
    Metadata_New.append(svol_raw)
    Frontal_raw = norm_list(Metadata[5])
    Metadata_New.append(Frontal_raw)
    Insula_raw = norm_list(Metadata[6])
    Metadata_New.append(Insula_raw)
    Parietal_raw = norm_list(Metadata[7])
    Metadata_New.append(Parietal_raw)

    return Metadata_New



# ## metadata_withFeature = [df_lab['fname'].tolist(), df_lab['age_raw'].tolist(), df_lab['sex_raw'].tolist(), svol_raw
#       ##      df_lab['dx_raw'].tolist(),df_lab['Frontal_raw'].tolist(),df_lab['Insula_raw'].tolist(),df_lab['Parietal_raw'].tolist()]

# ### Here we fixed the dimension of the meta data:
# ### Get different metadata: age:age_raw,  gender: sex_raw, SOVL:svol_raw ; dignosis: dx_raw, Frontal_raw, Insula_raw,Parietal_ra
# ### One hot for discrete 

def get_kernelC(Metadata):
    N = len(Metadata[0])
    metadata = np.ones((N,3))
    #X_shuffled[:,0] = label_hiv
    ## desease
    metadata[:,1] = Metadata[0] ## sex## we may donot need sex in our model, so put at there(May just asign the original number)
    metadata[:,2] = Metadata[1]## Age 
    
    cf_kernel = torch.tensor(np.linalg.inv(np.transpose(metadata).dot(metadata))).float()  
    
    return cf_kernel, torch.from_numpy(metadata).float()

def get_kernel(Metadata):
    N = len(Metadata[0])
    metadata = np.ones((N,8))
    #X_shuffled[:,0] = label_hiv
    ## desease
    metadata[:,1] = Metadata[0] ## sex## we may donot need sex in our model, so put at there(May just asign the original number)
    metadata[:,2] = Metadata[1]## Age 
    metadata[:,3] = Metadata[2]# diagnosis 
    metadata[:,4] = Metadata[3]# svol
    metadata[:,5] = Metadata[4]# Frontal_raw, 
    metadata[:,6] = Metadata[5]# Insula_raw
    metadata[:,7] = Metadata[6]# Parietal_raw 
    
    cf_kernel = torch.tensor(np.linalg.inv(np.transpose(metadata).dot(metadata))).float()  
    
    return cf_kernel, torch.from_numpy(metadata).float()

## Fed in a feature vector, this functon help to compute Beta
## Then remove all the factor-related parts and return the residual.
## Inputs: feat_vector, Metadata
## Return: residual, Beta
def GLM(X_feature, Metadata):

    N = X_feature.shape[0]
    X_vec = X_feature.reshape(N, -1)
    print(f'{X_vec.shape[0]} datas with feature of {X_vec.shape}')

    ### Compute the kernel (8x8) and Metadata (Nx8)
    cf_kernel, Metadata = get_kernelC(Metadata)
    Meta_T = torch.transpose(Metadata, 0, 1) #(8xN)

    pinv = torch.mm(cf_kernel, Meta_T) 
    ## Beta is related to the feature
    Beta = torch.mm(pinv, X_vec)   #(8xN vs Nx228096)

    X_r = torch.mm(Metadata, Beta) #torch.mm(X_batch[:, 1:], B[1:]) 
    residual = X_vec -  X_r
    residual = residual.reshape(X_feature.shape)

    return residual, Beta, Metadata

def compute_GLM():
    Metadata = torch.load("/home/wpeng/results/Meta_DPM/latents/MRI-meta-qLatents")
    X_feature = Metadata[-1]
    X_feature = torch.cat(X_feature)
    print(f"Feature size {X_feature.shape}")
    Metadata = convert_metadata(Metadata)

    residual, Beta, Metadata = GLM(X_feature, Metadata)
    print(f"Feature size {residual.shape}")
    print(f"Beta size {Beta.shape}")

    torch.save(Beta, "/home/wpeng/results/Meta_DPM/latents/Beta")
    torch.save(residual, "/home/wpeng/results/Meta_DPM/latents/Residual")
    torch.save(Metadata, "/home/wpeng/results/Meta_DPM/latents/Metadata")

## Combined dataset
def compute_GLM_combined():
    Metadata = torch.load("./Sora_latents")
    X_feature = Metadata[-1]
    X_feature = torch.cat(X_feature)
    print(f"Feature size {X_feature.shape}")
    Metadata = convert_metadataC(Metadata)

    residual, Beta, Metadata = GLM(X_feature, Metadata)
    print(f"Feature size {residual.shape}")
    print(f"Beta size {Beta.shape}")

    torch.save(Beta, "/home/wpeng/results/Meta_DPM/latents/BetaC")
    torch.save(residual, "/home/wpeng/results/Meta_DPM/latents/ResidualC")
    torch.save(Metadata, "/home/wpeng/results/Meta_DPM/latents/MetadataC")

compute_GLM_combined()
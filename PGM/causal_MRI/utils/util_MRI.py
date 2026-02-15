import nibabel as nib
import os
import numpy as np 

def save_MRI_samples(images, im_name, step=0, log_dir=None):
    if log_dir is None:
        log_dir = im_name + "/"
    os.makedirs(log_dir, exist_ok=True)
    batch_indices = images.shape[0]
    # print(f'There are {batch_indices} images in the batch')
    output_filenames = [ f"{log_dir}{im_name}_{step}-{i}.nii.gz" for i in range(batch_indices)]

    images = images.permute(0,1, 4,3,2)
    samples = images.cpu().detach().numpy()
    
    samples = np.where(samples < -1, -1, samples)
    samples = (samples + 1) / 2 
    # # drange = [-1, 1]
    # samples = (samples + 1) / 2 * 255
    # # samples = (samples ) * 1000
    # samples = samples.astype(np.uint8)
    for i in range(batch_indices):
        # numpy_array = np.transpose(samples[i], (1, 2, 3, 0))
        # print(numpy_array.shape)#128 1 128 128
        numpy_array = samples[i]
        # numpy_array = numpy_array-numpy_array.min()
        # numpy_array = numpy_array/numpy_array.max()
        ni_img = nib.Nifti1Image(numpy_array.squeeze(0), affine=np.eye(4))
        nib.save(ni_img, output_filenames[i])
        # np.save(output_filenames[i], samples[i])
        print(f"*** Saved {output_filenames[i]} ***")
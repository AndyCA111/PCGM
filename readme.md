# Probabilistic Causal Graph Model (PCGM)

## Overview

3D brain MRI studies often examine subtle morphometric differences between cohorts that are difficult to detect visually. Due to the high cost of MRI acquisition, these studies can greatly benefit from image synthesis, particularly counterfactual image generation, as demonstrated in other domains such as computer vision.

However, existing counterfactual models struggle to generate anatomically plausible MRIs because they lack explicit inductive biases to preserve fine-grained anatomical structures. Most models are trained to optimize global image appearance (e.g., reconstruction loss or cross-entropy), rather than subtle yet clinically meaningful local variations.

We propose the Probabilistic Causal Graph Model (PCGM), which explicitly integrates voxel-level anatomical constraints as priors into a generative diffusion framework. PCGM captures anatomical constraints via a probabilistic causal graph module and translates them into spatial binary masks that indicate regions of subtle variation. These masks are encoded using a 3D extension of ControlNet to constrain a counterfactual denoising UNet. The resulting latent representations are decoded into high-quality 3D brain MRIs using a 3D diffusion decoder.

Extensive experiments on multiple datasets show that PCGM produces higher-quality structural MRIs than strong baselines. Moreover, brain measurements extracted from PCGM-generated counterfactuals replicate subtle disease-related effects on cortical regions previously reported in the neuroscience literature.

---

## Dataset Overview

In `models/dataloader.py`, define your data path:
```python
video_data_paths_dict = {
    ...
    "MRI": "your data path"
}
```

Set your own data location in `dataloader.py` wherever it is used.

## Modules Overview

The codebase consists of three main components:

- PGM: Probabilistic Graph Module 
- MGD: Mask-Guided Diffusion
- CMG: Counterfactual Mask Generator

---

# 🚀 Get Started

```bash
git clone https://github.com/AndyCA111/PCGM.git
cd PCGM
pip install -r requirement.txt
```

## Segmentation Model for Preprocessing

In this project, we use [SynthSeg](https://github.com/BBillot/SynthSeg) for brain segmentation.

Please use this model to segment all brain data before every step, and save:
1. The output mask (including parcellation)
2. The probability maps for both the general segmentation (`post`) and parcellation (`parc`) masks

These can be easily found in the SynthSeg repo.



## Probabilistic Graph Module (PGM)

The Probabilistic Graph Module models anatomical and causal constraints used to guide counterfactual generation.

Documentation can be found at:

    PGM/causal_MRI/readme_pgm.md

---

## Counterfactual Mask Generator (CMG)

The Counterfactual Mask Generator converts causal effects inferred by the PGM into spatial binary masks for diffusion guidance.

Implementation notebook:

    modify_mask.ipynb

---

## Mask-Guided Diffusion (MGD)

### Variational Autoencoder (VAE)

Training:

    bash train_vae.sh

---

### Diffusion Model

Training:

    accelerate launch \
      --num_processes 2 \
      --mixed_precision fp16 \
      modelx/unet_diff.py

Alternative training script:

    bash run_diff.sh

Evaluation:

    accelerate launch \
      --num_processes 2 \
      --mixed_precision fp16 \
      models/unet_diff_eval.py

---

### Diffusion Decoder

Scripts for decoding latent representations into full-resolution 3D MRIs:

    unet_diff_diffusion_decoder.py
    unet_diff_diffusion_decoder_eval.py

---

### ControlNet (3D Mask Conditioning)

Training:

    accelerate launch \
      --num_processes 2 \
      --mixed_precision fp16 \
      models/unet_controlnet.py

---

## Inference

Once the PGM is trained, PCGM supports the following workflows:

Unconditional MRI generation:

    # Diffusion → Diffusion Decoder
    models/unet_diff_eval.py
    models/unet_diff_diffusion_decoder_eval.py

Age counterfactual generation:

    # Diffusion → Diffusion Decoder
    models/unet_diff_eval_editing_age.py
    models/unet_diff_diffusion_decoder_eval.py

AUD counterfactual generation:

    PGM → CMG → Diffusion (AUD counterfactual eval) → Diffusion Decoder (eval)

codes:
    
    PGM/causal_MRI/CE_sample_5fold.py
    using relevant code in modify_mask.ipynb to modify the mask you want
    then we need a file to save the counterfactual volume , and then use models/unet_diff_eval_editing_age_controlnet.py
    models/unet_diff_diffusion_decoder_eval.py


---




# Probabilistic Causal Graph Model (PCGM)

## Overview

3D brain MRI studies often examine subtle morphometric differences between cohorts that are difficult to detect visually. Due to the high cost of MRI acquisition, these studies can greatly benefit from image synthesis, particularly counterfactual image generation, as demonstrated in other domains such as computer vision.

However, existing counterfactual models struggle to generate anatomically plausible MRIs because they lack explicit inductive biases to preserve fine-grained anatomical structures. Most models are trained to optimize global image appearance (e.g., reconstruction loss or cross-entropy), rather than subtle yet clinically meaningful local variations.

We propose the Probabilistic Causal Graph Model (PCGM), which explicitly integrates voxel-level anatomical constraints as priors into a generative diffusion framework. PCGM captures anatomical constraints via a probabilistic causal graph module and translates them into spatial binary masks that indicate regions of subtle variation. These masks are encoded using a 3D extension of ControlNet to constrain a counterfactual denoising UNet. The resulting latent representations are decoded into high-quality 3D brain MRIs using a 3D diffusion decoder.

Extensive experiments on multiple datasets show that PCGM produces higher-quality structural MRIs than strong baselines. Moreover, brain measurements extracted from PCGM-generated counterfactuals replicate subtle disease-related effects on cortical regions previously reported in the neuroscience literature.

---

## Dataset Overview

To be filled in. Please describe dataset sources, cohort definitions, preprocessing steps, and data splits.

---

## Modules Overview

The codebase consists of three main components:

- PGM: Probabilistic Graph Module (data modeling, causal structure, preprocessing)
- MGD: Mask-Guided Diffusion (core generative backbone)
- CMD: Counterfactual Modeling and Diagnostics (task-specific logic and evaluation)

---

# 🚀 Get Started

```bash
git clone https://github.com/<your-name>/PCGM.git
cd PCGM
pip install -r requirement.txt
```




## Probabilistic Graph Module (PGM)

The Probabilistic Graph Module models anatomical and causal constraints used to guide counterfactual generation.

Documentation can be found at:

    PGM/causal_MRI/readme_pgm.md

---

## Counterfactual Mask Generator (CMG)

The Counterfactual Mask Generator converts causal effects inferred by the PGM into spatial binary masks for diffusion guidance.

Implementation notebook:

    modify_mask_wei.ipynb

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
      --main_process_port 29566 \
      --mixed_precision fp16 \
      modelx/unet_diff.py

Alternative training script:

    bash run_diff.sh

Evaluation:

    accelerate launch \
      --num_processes 2 \
      --main_process_port 29566 \
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
      --main_process_port 29566 \
      --mixed_precision fp16 \
      models/unet_controlnet.py

---

## End-to-End Pipeline

Once the PGM is trained, PCGM supports the following workflows:

Pure image generation:

    # Diffusion (eval) → Diffusion Decoder (eval)
    models/unet_diff_eval.py
    models/unet_diff_diffusion_decoder_eval.py

Age counterfactual generation:

    # Diffusion (age counterfactual eval) → Diffusion Decoder (eval)
    models/unet_diff_eval_editing_age.py
    models/unet_diff_diffusion_decoder_eval.py

codes:

    models/unet_diff_eval_editing_age.py
    models/unet_diff_diffusion_decoder_eval.py

AUD counterfactual generation:

    PGM → CMG → Diffusion (AUD counterfactual eval) → Diffusion Decoder (eval)

codes:



---




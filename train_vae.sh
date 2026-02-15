#!/bin/bash
#
#SBATCH --job-name=ftdec
#SBATCH --account=project_2001654
#SBATCH -p gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

# Activate Conda environment

source /scratch/project_2002846/Binxu/anaconda3/etc/profile.d/conda.sh
# echo ". /scratch/project_2002846/Binxu/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
conda activate binxu

python ./opensora/train/train_causalvae.py \
    --exp_name "vae888_8_decoder_seg" \
    --batch_size 1 \
    --precision bf16 \
    --max_steps 4000000 \
    --save_steps 8000 \
    --output_dir results/vae_seg \
    --video_num_frames 17 \
    --resolution 224 \
    --sample_rate 1 \
    --dataset mixed \
    --n_nodes 1 \
    --devices 1 \
    --num_workers 4 

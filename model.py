"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start,beta_end,T,dtype=torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1-betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    return a.gather(0,t.long()).reshape(-1,1,1,1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t=extract_into_batch(alphas_cumprod,t,x0)
    x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * noise
    return x_t

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas=linear_beta_schedule(T,beta_start,beta_end)
    alphas=alphas_from_betas(betas)
    alphas_cumprod=cumprod_alphas(alphas)
    sqrt_one_minus_alphas_cumprod=torch.sqrt(1-alphas_cumprod)
    return {'betas':betas,
    'alphas':alphas,
    'alphas_cumprod':alphas_cumprod,
    'sqrt_alphas_cumprod':torch.sqrt(alphas_cumprod),
    'sqrt_one_minus_alphas_cumprod':sqrt_one_minus_alphas_cumprod,
    'T':T}

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return torch.mean((noise-noise_pred)**2)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t=q_sample(x0,t,noise,alphas_cumprod)
    pred_noise=model(x_t,t)
    loss=noise_prediction_loss(pred_noise,noise)
    return loss

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    B=t.shape[0]
    embd=torch.zeros([B,dim])
    half=int(dim/2)
    for i in range(B):
        for j in range(dim):
            if j<half:
                embd[i,j]=torch.sin(torch.tensor(t[i]/10000**(j/dim)))
            else:
                embd[i,j]=torch.cos(torch.tensor(t[i]/10000**((j-1)/dim)))
    return embd

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    conv_in_w=torch.tensor(torch.normal(mean=0,std=0.02,size=(hidden,in_ch,3,3))).requires_grad_(True)
    conv_in_b=torch.zeros(hidden).requires_grad_(True)
    time_mlp_w=torch.tensor(torch.normal(mean=0,std=0.02,size=(hidden,time_dim))).requires_grad_(True)
    time_mlp_b=torch.zeros(hidden).requires_grad_(True)
    conv_mid_w=torch.tensor(torch.normal(mean=0,std=0.02,size=(hidden,hidden,3,3))).requires_grad_(True)
    conv_mid_b=torch.zeros(hidden).requires_grad_(True)
    conv_out_w=torch.tensor(torch.normal(mean=0,std=0.02,size=(in_ch,hidden,3,3))).requires_grad_(True)
    conv_out_b=torch.zeros(in_ch).requires_grad_(True)
    return {'conv_in_w':conv_in_w,
    'conv_in_b':conv_in_b,
    'time_mlp_w':time_mlp_w,
    'time_mlp_b':time_mlp_b,
    'conv_mid_w':conv_mid_w,
    'conv_mid_b':conv_mid_b,
    'conv_out_w':conv_out_w,
    'conv_out_b':conv_out_b
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    conv_in_w=params['conv_in_w']
    conv_in_b=params['conv_in_b']
    time_mlp_w=params['time_mlp_w']
    time_mlp_b=params['time_mlp_b']
    conv_mid_w=params['conv_mid_w']
    conv_mid_b=params['conv_mid_b']
    conv_out_w=params['conv_out_w']
    conv_out_b=params['conv_out_b']
    h=F.conv2d(input=x,weight=conv_in_w,bias= conv_in_b, padding=1)
    temb=timestep_embedding(t,int(time_mlp_w.shape[1]))
    temb=torch.relu(F.linear(input=temb,weight=time_mlp_w,bias=time_mlp_b))
    h=h+temb[:,:,None,None]
    h=torch.relu(h)
    h=F.conv2d(input=h,weight=conv_mid_w,bias=conv_mid_b,padding=1)
    h=F.conv2d(input=h,weight=conv_out_w,bias=conv_out_b,padding=1)
    return h

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    a=torch.zeros(n,1,size,size)
    radius=size//4
    for p in range(n):
        x,y=torch.randint(radius,size-radius,(2,))
        for i in range(size):
            for j in range(size):
                if (x-i)**2+(y-j)**2<=radius**2:
                    a[p,0,i,j]=1
                else:
                    a[p,0,i,j]=0
    return a

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    
    alphas_cumprod=extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat=(x_t-torch.sqrt(1-alphas_cumprod)*eps)/torch.sqrt(alphas_cumprod)
    return x0_hat

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement


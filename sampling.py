"""
Implements the Annealed Langevin Posterior Sampling Algorithm for inverse problems such as inpainting, Gaussian Motion deblurring.
 
Jyothi Rikhab Chand, 2026
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
import pickle
import dnnlib
from training.dataset import ImageFolderDataset
from operators import *  
from utils import Denoiser, ALPS

import json
from typing import Dict, Any
import os
import sys


# -----------------------------
# helper functions
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def to_img(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().clamp(-1, 1)
    img = (img + 1) / 2.0
    return img.permute(0, 2, 3, 1).squeeze(0).numpy()


    
def save_image(img: torch.Tensor, path: str): 
    plt.figure(figsize=(3,3)) 
    plt.imshow(to_img(img)) 
    plt.axis("off") 
    plt.savefig(path, bbox_inches="tight", pad_inches=0) 
    plt.close() 
def save_final_comparison(xorig, y, x, path): 
    fig, ax = plt.subplots(1, 3, figsize=(3, 3)) 
    ax[0].imshow(to_img(xorig)) 
    ax[0].set_title("Original",fontsize=5) 
    ax[0].axis("off") 
    ax[1].imshow(to_img(y)) 
    ax[1].set_title("Measurements",fontsize=5) 
    ax[1].axis("off") 
    ax[2].imshow(to_img(x)) 
    ax[2].set_title("Reconstruction",fontsize=5) 
    ax[2].axis("off") 
    plt.tight_layout() 
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05) 
    plt.close(fig)


# -----------------------------
# Options
# -----------------------------
@dataclass
class Options:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho: float
    K: int


# -----------------------------
# Operator factory
# -----------------------------
def build_operator_and_measurements(xorig: torch.Tensor, device: torch.device, cfg: Dict[str, Any]):
    """
    Returns (A, y, operator_name_for_filenames)
    Uses cfg["operator"]["type"] and cfg["operator"]["params"].
    """

    if "operator" not in cfg:
        raise KeyError("Config must contain an 'operator' section.")

    op_cfg = cfg["operator"]
    op_type = op_cfg.get("type", None)
    params = op_cfg.get("params", {})

    if op_type is None:
        raise KeyError("cfg['operator']['type'] is required.")

    op_type = op_type.lower()

    # ---- Inpainting / Masking ----
    if op_type in ["inpainting", "masking"]:
        mg = params.get("mask_generator", cfg.get("mask_generator", None))
        if mg is None:
            raise KeyError("Inpainting requires mask_generator config under operator.params.mask_generator (or top-level mask_generator).")

        mask_gen = mask_generator(
            mask_type=mg["mask_type"],
            mask_len_range=mg["mask_len_range"],
            mask_prob_range=mg["mask_prob_range"],
            image_size=mg["image_size"],
            margin=tuple(mg.get("margin", [0, 0])),
            **mg.get("extra_kwargs", {})
        )
        mask = mask_gen(xorig)

        eta = float(params.get("eta", cfg.get("eta", 0.1)))
        A = Inpainting(mask, eta=eta).to(device)

 
        y = A.forward(xorig + eta * torch.randn_like(xorig))
       

        return A, y, f"inpainting_{mg['mask_type']}"

    # ---- Gaussian blur operator ----
    elif op_type in ["gaussian_blur", "gaussianblur", "gaussian"]:
        
        sigma = float(params["sigma"])  
        eta = float(params.get("eta", cfg.get("eta", 0.1)))

        A = GaussianBlurOperator(xorig, sigma=sigma, eta=eta).to(device)
        y = A.forward(xorig)
        y = y+ (torch.tensor(eta)*(torch.randn_like(y)))
        return A, y, f"gaussian_blur_sigma{sigma}"

    # ---- Motion blur operator ----
    elif op_type in ["motion_blur", "motionblur", "motion"]:
     
        length = int(params["length"])        
        theta_deg = float(params["theta_deg"]) 
        eta = float(params.get("eta", cfg.get("eta", 0.1)))

        A = MotionBlurOperator(xorig, length=length, theta_deg=theta_deg, eta=eta).to(device)
        y = A.forward(xorig)
        y = y+ (torch.tensor(eta)*(torch.randn_like(y)))
        return A, y, f"motion_blur_L{length}_theta{theta_deg}"

    else:
        raise ValueError(f"Unknown operator type: '{op_type}'. Add it in build_operator_and_measurements().")


# -----------------------------
# Main
# -----------------------------
def main(config_path: str):
    cfg = load_config(config_path)
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Loading the model
    # -----------------------------
    network_pkl = cfg.get("network_pkl", "./models/edm_diffusion/face_dataset/edm-ffhq-64x64-uncond-ve.pkl")
    with dnnlib.util.open_url(network_pkl) as f:
        net0 = pickle.load(f)["ema"].to(device)

    net = Denoiser(net0).to(device)

    net_path = cfg.get("net_path", "./models/score_distillation/face_dataset/best_model_epoch_36.pth")
    state_dict = torch.load(net_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # -----------------------------
    # Loading the dataset
    # -----------------------------
    dataset_path = cfg.get("dataset_path", "./datasets/ffhq-64x64.zip")
    dataset = ImageFolderDataset(dataset_path, resolution=64, xflip=False)

    (xorig, _) = dataset[int(cfg.get("sample_index", 0))]
    xorig = torch.tensor(xorig).to(device) / 127.5 - 1
    xorig = torch.unsqueeze(xorig, 0)

    # -----------------------------
    # Build operator A and measurements y (based on config)
    # -----------------------------
    A, y, op_name = build_operator_and_measurements(xorig, device, cfg)

    # -----------------------------
    # Options
    # -----------------------------
    opts = Options(
        num_steps=cfg["alps_options"]["num_steps"],
        sigma_max=cfg["alps_options"]["sigma_max"],
        sigma_min=cfg["alps_options"]["sigma_min"],
        rho=cfg["alps_options"]["rho"],
        K=cfg["alps_options"]["K"],
    )

    # -----------------------------
    # Run ALPS
    # -----------------------------
    x, xarray = ALPS(A, net, y, opts, isALPS=True, storeIntermediate=True)

    # -----------------------------
    # Save results
    # -----------------------------
    output_dir = cfg.get("output_dir", "outputs")
    run_dir = os.path.join(output_dir, op_name)
    os.makedirs(run_dir, exist_ok=True)

    save_final_comparison(xorig, y, x, os.path.join(run_dir, "final_comparison.png"))

    interm_dir = os.path.join(run_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    # subsample for saving if too many
    max_frames = int(cfg.get("max_saved_frames", 10))
    if xarray.shape[0] > max_frames:
        step = max(1, xarray.shape[0] // max_frames)
        xarray = xarray[::step]

    for i in range(xarray.shape[0]):
        save_image(xarray[i:i + 1], os.path.join(interm_dir, f"step_{i:03d}.png"))

    print(f"Saved results to '{run_dir}/'")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_path)
#example usage: python sampling.py config/inpainting_box.json
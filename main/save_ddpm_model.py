import argparse
import torch
from models.vae import VAE

def save_vae(model_path, save_path):
    model_dict = torch.load(model_path)['state_dict']

    ddpm_model = torch.load('/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar2/checkpoints/ddpmv2-cifar10-epoch=499-loss=0.0327.ckpt')

    only_original_weight = {k: v for k, v in model_dict.items() if 'origin_vae' not in k}
    
    ddpm_model['state_dict'] = only_original_weight

    torch.save(ddpm_model, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", default='/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_kd_fbp_cifar2/checkpoints/ddpmv2-cifar10-epoch=499-loss=54749.5977.ckpt')          # extra value
    parser.add_argument("-save_path", default='/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_kd_fbp_cifar2/ddpm-epoch=499.ckpt')           # existence/nonexistence
    args = parser.parse_args()

    save_vae(args.model_path, args.save_path)

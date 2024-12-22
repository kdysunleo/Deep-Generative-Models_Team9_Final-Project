import argparse
import torch
from models.vae import VAE

def save_vae(model_path, save_path):
    model_dict = torch.load(model_path)['state_dict']

    vae_model = torch.load('/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt')
    # vae_model = torch.load('/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar2_alpha=1.0_epochs=500/checkpoints/vae-cifar2_alpha=1.0-epoch=499-train_loss=0.0000.ckpt')

    model_dict = {k: v for k, v in model_dict.items() if 'origin_vae' not in k}
    only_vae = {k.replace('vae.', ''): v for k, v in model_dict.items() if 'vae' in k}
    
    vae_model['state_dict'] = only_vae

    torch.save(vae_model, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", default='/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_cifar10/checkpoints/diffusevae_e2e_cifar10-epoch=499-loss=0.0164.ckpt')          # extra value
    parser.add_argument("-save_path", default='/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_cifar10/vae-epoch=499.ckpt')           # existence/nonexistence
    args = parser.parse_args()

    save_vae(args.model_path, args.save_path)

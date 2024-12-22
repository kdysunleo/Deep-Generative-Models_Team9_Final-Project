# # CelebAMaskHQ training
# python main/train_ae.py +dataset=celebamaskhq128/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/CelebAMask-HQ/' \
#                      dataset.vae.data.name='celebamaskhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=42 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_cmhq128_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'cmhq128_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # FFHQ 128 training
# python main/train_ae.py +dataset=ffhq/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/ffhq/' \
#                      dataset.vae.data.name='ffhq' \
#                      dataset.vae.data.hflip=True \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.log_step=50 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_ffhq128_11thJune_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'ffhq128_11thJune_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

# # AFHQv2 training
# python main/train_ae.py +dataset=afhq256/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/afhq_v2/' \
#                      dataset.vae.data.name='afhq' \
#                      dataset.vae.training.batch_size=8 \
#                      dataset.vae.training.epochs=500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_afhq256_10thJuly_alpha=1.0/\' \
#                      dataset.vae.training.workers=2 \
#                      dataset.vae.training.chkpt_prefix=\'afhq256_10thJuly_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0


# # CelebA training
# python main/train_ae.py +dataset=celeba64/train \
#                      dataset.vae.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
#                      dataset.vae.data.name='celeba' \
#                      dataset.vae.training.batch_size=32 \
#                      dataset.vae.training.epochs=1500 \
#                      dataset.vae.training.device=\'gpu:0,1,2,3\' \
#                      dataset.vae.training.results_dir=\'/data1/kushagrap20/vae_celeba64_alpha=1.0/\' \
#                      dataset.vae.training.workers=4 \
#                      dataset.vae.training.chkpt_prefix=\'celeba64_alpha=1.0\' \
#                      dataset.vae.training.alpha=1.0

python main/train_ae_fbp.py +dataset=cifar10/train \
                     dataset.vae.data.root="/nlp_data/kdy/genai/DiffuseVAE/main/datasets/img_cifar10" \
                     dataset.vae.data.name='cifar10' \
                     dataset.vae.training.batch_size=128 \
                     dataset.vae.training.epochs=500 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar10_alpha=1.0_epochs=500/\' \
                     dataset.vae.training.workers=8 \
                     dataset.vae.training.chkpt_prefix=\'cifar10_alpha=1.0\' \
                     dataset.vae.training.alpha=1.0

# python main/train_ddpm_with_fbp.py +dataset=cifar10/train \
#                      dataset.ddpm.data.root=\'/nlp_data/kdy/genai/DiffuseVAE/main/datasets/img_cifar10\' \
#                      dataset.ddpm.data.name='cifar10' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.3 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=500 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=128 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar2_alpha=1.0_epochs=500//checkpoints/vae-cifar2_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:1\' \
#                      dataset.ddpm.training.results_dir=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar2\' \
#                      dataset.ddpm.training.workers=8 \
#                      dataset.ddpm.training.chkpt_prefix=\'cifar10\'
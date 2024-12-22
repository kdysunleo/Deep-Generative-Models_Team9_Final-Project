# baseline
python main/eval/ddpm/interpolate_vae.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=42 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_cifar10/checkpoints/ddpmv2-cifar10-epoch=499-loss=0.0405.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/linear_interpolate_ddpm_vae/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_vae_latents/gmm_z/gmm_50.joblib\'

python main/eval/ddpm/interpolate_vae.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=42 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_cifar10/checkpoints/diffusevae_e2e_cifar10-epoch=499-loss=0.0164.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/linear_interpolate_ddpm_e2e_vae/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_cifar10/vae-epoch=499.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_e2e_latents/gmm_z/gmm_50.joblib\'

python main/eval/ddpm/interpolate_vae.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=42 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_reg_cifar10/ddpm-epoch=499.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/linear_interpolate_ddpm_reg_vae/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_reg_cifar10/vae-epoch=499.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_e2e_reg_latents/gmm_z/gmm_50.joblib\'

python main/eval/ddpm/interpolate_vae.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=42 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_kd_cifar10/ddpm-epoch=499.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/linear_interpolate_ddpm_kd_vae/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_kd_cifar10/vae-epoch=499.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_kd_latents/gmm_z/gmm_50.joblib\'
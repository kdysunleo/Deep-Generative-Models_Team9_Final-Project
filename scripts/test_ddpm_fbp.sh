python main/extract_latents_fbp.py extract --device gpu:0 \
                                --dataset-name cifar10 \
                                --image-size 32 \
                                --save-path /nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/ \
                                '/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt' \
                                /nlp_data/kdy/genai/DiffuseVAE/main/datasets/img_cifar10

python main/expde.py fit-gmm /nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/latents_cifar10.npy --save-path '/nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/gmm_z/' --n-components 50

python main/eval/ddpm/sample_cond_fbp.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar10/checkpoints/ddpmv2-cifar10-epoch=499-loss=0.0418.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/ddpm_fbp_cifar10/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=2500 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=4 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/gmm_z/gmm_50.joblib\'

python main/eval/ddpm/sample_cond_fbp.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar10/checkpoints/ddpmv2-cifar10-epoch=499-loss=0.0418.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/ddpm_fbp_cifar10/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=2500 \
                        dataset.ddpm.evaluation.n_steps=250 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=4 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/gmm_z/gmm_50.joblib\'

python main/eval/ddpm/sample_cond_fbp.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar10/checkpoints/ddpmv2-cifar10-epoch=499-loss=0.0418.ckpt\' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/ddpm_fbp_cifar10/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=2500 \
                        dataset.ddpm.evaluation.n_steps=300 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=4 \
                        dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/vae_fbp_cifar10_alpha=1.0_epochs=500/checkpoints/vae-cifar10_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar10_fbp_latents/gmm_z/gmm_50.joblib\'

# python main/eval/ddpm/sample_cond_fbp.py +dataset=cifar10/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.model.dim=128 \
#                         dataset.ddpm.model.dropout=0.3 \
#                         dataset.ddpm.model.attn_resolutions=\'16,\' \
#                         dataset.ddpm.model.n_residual=2 \
#                         dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                         dataset.ddpm.model.n_heads=8 \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.guidance_weight=0.0 \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.device=\'gpu:3\' \
#                         dataset.ddpm.evaluation.save_mode='image' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_kd_cifar2/checkpoints/ddpmv2-cifar2_e2e_kd-epoch=499-loss=27891.6992.ckpt\' \
#                         dataset.ddpm.evaluation.resample_strategy='truncated' \
#                         dataset.ddpm.evaluation.skip_strategy='quad' \
#                         dataset.ddpm.evaluation.sample_method='ddpm' \
#                         dataset.ddpm.evaluation.sample_from='target' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.save_path=\'/nlp_data/kdy/genai/DiffuseVAE/image_sample/ddpm_fbp_e2e_kd_cifar2_sample_cond/\' \
#                         dataset.ddpm.evaluation.z_cond=False \
#                         dataset.ddpm.evaluation.n_samples=2500 \
#                         dataset.ddpm.evaluation.n_steps=250 \
#                         dataset.ddpm.evaluation.save_vae=True \
#                         dataset.ddpm.evaluation.workers=4 \
#                         dataset.vae.evaluation.chkpt_path=\'/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_fbp_cifar2/vae-epoch=499.ckpt\' \
#                         dataset.vae.evaluation.expde_model_path=\'/nlp_data/kdy/genai/DiffuseVAE/cifar2_fbp_e2e_kd_latents/gmm_z/gmm_50.joblib\'

# fidelity --gpu 1 --fid --isc --input1 /nlp_data/kdy/genai/DiffuseVAE/image_sample/ddpm_fbp_e2e_kd_cifar2_sample_cond/1000/images --input2 cifar10-train

# python main/extract_latents.py extract --device gpu:0 \
#                                 --dataset-name cifar10 \
#                                 --image-size 32 \
#                                 --save-path /nlp_data/kdy/genai/DiffuseVAE/cifar2_reg_latents/ \
#                                 '/nlp_data/kdy/genai/DiffuseVAE/main/results/diffusevae_e2e_reg_cifar2/vae-epoch=499.ckpt' \
#                                 /nlp_data/kdy/genai/DiffuseVAE/main/datasets/img_cifar10



# Fit GMM CMHQ-128
# python main/expde.py fit-gmm ~/cmhq128_latents/latents_celebamaskhq.npy --save-path '/data1/kushagrap20/cmhq128_latents/gmm_z/' --n-components 150

# Fit GMM CIFAR-10
# python main/expde.py fit-gmm ~/cifar10_latents/latents_cifar10.npy --save-path '/data1/kushagrap20/cifar10_latents/gmm_z/' --n-components 100
python main/expde.py fit-gmm /nlp_data/kdy/genai/DiffuseVAE/cifar2_reg_latents/latents_cifar2.npy --save-path '/nlp_data/kdy/genai/DiffuseVAE/cifar2_reg_latents/gmm_z/' --n-components 50

# Fit GMM CelebA-64
# python main/expde.py fit-gmm ~/celeba64_latents/latents_celeba.npy --save-path '/data1/kushagrap20/celeba64_latents/gmm_z/' --n-components 50

# Fit GMM AFHQ-256 Dogs
# python main/expde.py fit-gmm ~/afhq256_dog_latents/latents_afhq.npy --save-path '/data1/kushagrap20/afhq256_dog_latents/gmm_z/' --n-components 100

# Fit GMM CelebA-HQ-256
# python main/expde.py fit-gmm ~/celebahq_latents/latents_celebahq.npy --save-path '/data1/kushagrap20/celebahq_latents/gmm_z/' --n-components 150
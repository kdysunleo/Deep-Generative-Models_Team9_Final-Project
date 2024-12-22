import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# CIFAR-10 원본 이미지와 생성된 이미지 폴더 경로
cifar10_folder = "/nlp_data/kdy/genai/DiffuseVAE/original_images/filtered"
generated_folder = "/nlp_data/kdy/genai/DiffuseVAE/image_recons/ddpm_e2e_kd_cifar10/filtered"

# PSNR 및 SSIM 계산 함수
def calculate_metrics(original_path, generated_path):
    original = np.asarray(Image.open(original_path).convert("RGB")) / 255.0  # [0, 1] 정규화
    generated = np.asarray(Image.open(generated_path).convert("RGB")) / 255.0

    psnr_value = psnr(original, generated, data_range=1)

    # SSIM 계산: win_size와 채널 축 설정
    ssim_value = ssim(
        original,
        generated,
        data_range=1,
        win_size=3,  # 윈도우 크기 (이미지 크기에 맞게 설정)
        channel_axis=2,  # 채널 축 설정 (RGB 이미지의 경우 축=2)
    )
    return psnr_value, ssim_value


# 이미지 파일 리스트 로드
def get_image_files(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])

# 실행 스크립트
def main():
    # CIFAR-10 및 생성된 이미지 파일 로드
    cifar_images = get_image_files(cifar10_folder)
    generated_images = get_image_files(generated_folder)
    
    # 파일 수 확인
    if len(cifar_images) != len(generated_images):
        print(f"Warning: CIFAR-10 images ({len(cifar_images)}) and generated images ({len(generated_images)}) count mismatch.")
    
    # PSNR 및 SSIM 계산
    metrics = []
    for original_path, generated_path in zip(cifar_images, generated_images):
        psnr_value, ssim_value = calculate_metrics(original_path, generated_path)
        metrics.append((psnr_value, ssim_value))
        print(f"Original: {original_path}, Generated: {generated_path}")
        print(f"PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}")
    
    # 평균 PSNR 및 SSIM 출력
    avg_psnr = np.mean([m[0] for m in metrics])
    avg_ssim = np.mean([m[1] for m in metrics])
    print(f"\nAverage PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
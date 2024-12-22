import os
from PIL import Image
import numpy as np
import cv2  # OpenCV 필요
import matplotlib.pyplot as plt

# Smoothing 평가 함수
def calculate_smoothness(image_path):
    """
    이미지의 Smoothing 정도를 평가하기 위해 Laplacian Variance를 계산합니다.
    - Variance 값이 낮을수록 이미지가 더 smooth합니다.
    """
    # 이미지 로드 및 그레이스케일 변환
    image = np.asarray(Image.open(image_path).convert("L"))  # 그레이스케일 변환
    
    # Laplacian 필터 적용
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Laplacian의 분산 계산
    laplacian_variance = laplacian.var()
    return laplacian_variance

# 이미지 시각화 및 저장 함수
def visualize_laplacian(image_path, save_path="./visualization"):
    """
    이미지를 그레이스케일로 변환하고 Laplacian 필터를 적용한 결과를 시각화합니다.
    - save_path: 결과 이미지를 저장할 경로 (None이면 화면에 표시)
    """
    # 이미지 로드 및 처리
    image = np.asarray(Image.open(image_path).convert("L"))
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Grayscale")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Laplacian Filter")
    plt.imshow(np.abs(laplacian), cmap='gray')

    # 화면 표시 또는 저장
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 이미지 파일 리스트 로드
def get_image_files(folder):
    """
    지정된 폴더에서 이미지 파일(.png, .jpg)을 정렬하여 반환합니다.
    """
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])

# 실행 스크립트
def main():
    # CIFAR-10 원본 이미지와 생성된 이미지 폴더 경로
    cifar10_folder = "/nlp_data/kdy/genai/DiffuseVAE/original_images/filtered"
    generated_folder = "/nlp_data/kdy/genai/DiffuseVAE/image_recons/ddpm/filtered"

    # 이미지 파일 리스트
    cifar_images = get_image_files(cifar10_folder)
    generated_images = get_image_files(generated_folder)

    # Smoothing 평가 및 시각화
    metrics = []
    for i, (original_path, generated_path) in enumerate(zip(cifar_images, generated_images)):
        # Smoothness 계산
        original_smoothness = calculate_smoothness(original_path)
        generated_smoothness = calculate_smoothness(generated_path)
        metrics.append((original_smoothness, generated_smoothness))

        # 결과 출력
        print(f"Image {i + 1}:")
        print(f"Original Smoothness: {original_smoothness:.4f}, Generated Smoothness: {generated_smoothness:.4f}")

        # 시각화 및 저장
        #visualize_laplacian(original_path, save_path=f"results/original_laplacian_{i + 1}.png")
        #visualize_laplacian(generated_path, save_path=f"results/generated_laplacian_{i + 1}.png")

    # 평균 Smoothing 평가값 출력
    avg_original_smoothness = np.mean([m[0] for m in metrics])
    avg_generated_smoothness = np.mean([m[1] for m in metrics])
    print(f"\nAverage Original Smoothness: {avg_original_smoothness:.4f}")
    print(f"Average Generated Smoothness: {avg_generated_smoothness:.4f}")

# 메인 실행
if __name__ == "__main__":
    # 결과 저장 폴더 생성
    #os.makedirs("results", exist_ok=True)
    main()
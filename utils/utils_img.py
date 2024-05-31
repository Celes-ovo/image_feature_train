import cv2
import numpy as np

def process_image(binary_array, image_path, size):
    # 이미지 불러오기 및 그레이스케일로 변환
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 이미지 리사이즈
    resized_image = cv2.resize(image, (size, size))
    # output_image = resized_image
    output_image = (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image))

    # 이진 코드의 각 위치를 체크하여 각 변환 적용
    if binary_array[2] == '1' or binary_array[2] == '1.':  # 좌우 반전
        output_image = cv2.flip(output_image, 1)
    if binary_array[1] == '1' or binary_array[1] == '1.':  # 상하 반전
        output_image = cv2.flip(output_image, 0)
    if binary_array[0] == '1' or binary_array[0] == '1.':  # 색상 반전
        # output_image = 255 - output_image
        output_image = 1.0 - output_image

    # 이미지 저장
    # cv2.imwrite(save_path, output_image)
    # print(f"Processed image saved to {save_path}")

    return output_image


def calculate_features(image, size=32, bins=32):
    resized_image = cv2.resize(image, (size, size))
    normalized_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    histogram = cv2.calcHist([normalized_image], [0], None, [bins], [0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    row_means = np.mean(normalized_image, axis=1)
    col_means = np.mean(normalized_image, axis=0)

    features = np.concatenate((histogram, row_means, col_means))
    # arrays   = np.concatenate(binary_array)
    
    return features
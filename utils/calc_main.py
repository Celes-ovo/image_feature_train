import os, cv2
import numpy as np
from utils_img import process_image, calculate_features

# our modules
import ceph_config as cf


def calc_output(size, random_int, image_path):
    
    # 예) binary_array = '101'  # 상하반전과 색상 반전 적용
    # binary_array = str(bin(random_int)[2:]).zfill(3)
    binary_array = f'{random_int:03b}'
    binary_ndarray = np.array([binary_array[0], binary_array[1], binary_array[2]]).astype('float32')

    print(f'[Log] Binary array ({random_int}) : {binary_array}')

    # image_path = 'path_to_input_image.jpg'
    # size = 300
    # save_path = 'path_to_save_image.jpg'
    # process_and_save_image(binary_array, image_path, size, save_path)

    output_image = process_image(binary_array, image_path, size)

    # 이미지 feature 계산
    features = calculate_features(output_image, size=size, bins=cf.RESOLUTION_CONFIG)
    # print("Feature array:", features)

    return output_image, features, binary_ndarray
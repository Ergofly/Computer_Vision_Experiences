import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    img_m = img.shape[0]
    img_n = img.shape[1]
    km = kernel.shape[0]
    kn = kernel.shape[1]
    arr = np.zeros(shape=(img.shape))

    # 对于灰度图像和彩色图像分开处理
    if len(img.shape) >= 3:
        # 边缘补零，调用np数组方法或者传统方法
        img_with_edge = np.pad(img, ((int((km - 1) / 2), int((km - 1) / 2)), (int((kn - 1) / 2), int((kn - 1) / 2)),(0,0)))
        """补零传统方法
        img_with_edge=np.zeros(shape=[img_m+km-1,img_n+kn-1,img_d],dtype=np.uint8)
        img_with_edge[int((km-1)/2):int(-(km-1)/2),int((kn-1)/2):int(-(kn-1)/2),:]+=img"""
        for m in range(img_m):
            for n in range(img_n):
                sum = np.zeros(shape=[3], dtype=np.float)
                for i in range(km):
                    for j in range(kn):
                        for k in range(3):
                            sum[k] += img_with_edge[i + m][j + n][k] * kernel[i][j]
                arr[m][n] = sum

    else:
        img_with_edge = np.pad(img, ((int((km- 1) / 2), int((km - 1) / 2)), (int((kn - 1) / 2), int((kn - 1) / 2))))
        # 自相关
        for m in range(img_m):
            for n in range(img_n):
                sum = float(0)
                for i in range(km):
                    for j in range(kn):
                        sum += img_with_edge[i + m][j + n] * kernel[i][j]
                arr[m][n] = sum
    # arr=arr.astype(np.uint8)
    return arr
    # TODO-BLOCK-END


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # 权重核旋转180°
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    # 调用自相关，权重核替换成旋转过的权重核
    arr = cross_correlation_2d(img, kernel)
    return arr
    # TODO-BLOCK-END


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.zeros(shape=(height, width), dtype=np.float)
    # 中心点偏移量
    central_i = (height - 1) / 2
    central_j = (width - 1) / 2
    for i in range(height):
        for j in range(width):
            kernel[i][j] = (1 / (2 * math.pi * sigma ** 2)) * math.e ** (
                    -((i - central_i) ** 2 + (j - central_j) ** 2) / (2 * sigma ** 2))  # 求解高斯核的值
    return kernel/kernel.sum()
    # TODO-BLOCK-END


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    arr = convolve_2d(img, kernel)
    return arr
    # TODO-BLOCK-END


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    arr_low = convolve_2d(img, kernel)
    arr_high = img - arr_low
    return arr_high
    # TODO-BLOCK-END


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
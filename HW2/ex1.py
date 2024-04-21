import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


# Display an image as function
def display_image(image, title="Image"):
    cv2.imshow(title, image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # Need to implement here
    padding_size = filter_size // 2
    h, w = img.shape[:2]
    padded_h, padded_w = h + 2 * padding_size, w + 2 * padding_size
    padded_img = np.zeros((padded_h, padded_w), dtype=img.dtype)
    padded_img[padding_size:padded_h - padding_size, padding_size:padded_w - padding_size] = img

    # Replicate padding for the top and bottom borders
    padded_img[0:padding_size, padding_size:padded_w - padding_size] = img[0]
    padded_img[padded_h - padding_size:padded_h, padding_size:padded_w - padding_size] = img[h - 1]

    # Replicate padding for the left and right borders
    padded_img[:, 0:padding_size] = padded_img[:, padding_size:padding_size + 1]
    padded_img[:, padded_w - padding_size:padded_w] = padded_img[:, padded_w - padding_size - 1:padded_w - padding_size]

    return padded_img


def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Need to implement here
    padded_img = padding_img(img, filter_size)
    h, w = padded_img.shape
    padding_size = filter_size // 2
    for i in range(padding_size, h - padding_size):
        for j in range(padding_size, w - padding_size):
            window = padded_img[i - padding_size: i + padding_size + 1, j - padding_size: j + padding_size + 1]
            padded_img[i][j] = np.mean(window)

    filtered_img = padded_img[padding_size:h - padding_size, padding_size:w - padding_size]

    return filtered_img


def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    # Need to implement here
    padded_img = padding_img(img, filter_size)
    h, w = padded_img.shape
    padding_size = filter_size // 2
    for i in range(padding_size, h - padding_size):
        for j in range(padding_size, w - padding_size):
            window = padded_img[i - padding_size: i + padding_size + 1, j - padding_size: j + padding_size + 1]
            padded_img[i][j] = np.median(window)

    filtered_img = padded_img[padding_size:h - padding_size, padding_size:w - padding_size]

    return filtered_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    mse = np.mean((gt_img - smooth_img) ** 2)

    # MSE = 0 means no noise
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "ex1_images/noise.png"  # <- need to specify the path to the noise image
    img_gt = ""  # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 5

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

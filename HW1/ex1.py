import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load an image from file as function
def load_image(image_path):
    image = cv2.imread(image_path, 1)
    return image


# Display an image as function
def display_image(image, title="Image"):
    cv2.imshow(title, image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# grayscale an image as function
def grayscale_image(image):
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]
    grey_image = 0.299 * R + 0.587 * G + 0.114 * B

    grey_image = grey_image.astype(np.uint8)

    return grey_image


# Save an image as function
def save_image(image, output_path):
    cv2.imwrite(output_path, image)


# flip an image as function 
def flip_image(image):
    flipped = cv2.flip(image, 1)
    return flipped


# rotate an image as function
def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    save_image(img_gray_flipped, "images/img_gray_flipped.png")

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images

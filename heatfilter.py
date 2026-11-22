import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import sys

def add_gaussian_noise(image, variance):
    """
    Add Gaussian noise to an image.
    :param image: Input image as numpy array (float)
    :param variance: Noise variance
    :return: Noisy image as numpy array (float)
    """
    noise = np.random.normal(0, np.sqrt(variance), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def denoise_diffusion(image, iterations=30, dt=0.2):
    """
    Perform diffusion-based denoising using discrete Laplacian convolution.
    :param image: Input image as numpy array (float64)
    :param iterations: Number of iterations
    :param dt: Time step size
    :return: Denoised image as numpy array (float64)
    """
    J = image.copy()

    # 4-neighbor Laplacian kernel (N,S,E,W neighbors)
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1,-4, 1],
                                 [0, 1, 0]])

    for _ in range(iterations):
        laplacian = convolve(J, laplacian_kernel, mode='nearest')
        J = J + dt * laplacian
        # Clip to maintain valid range during iterations
        J = np.clip(J, 0, 255)

    return J

def main():

    image_path = 'C:Downloads/brain.bmp'
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found. Please check the path and filename.")
        sys.exit(1)

    I = np.array(img, dtype=np.float64)

    plt.figure(1)
    plt.title('Original Image')
    plt.imshow(I.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # Add Gaussian noise with variance 0.1
    noisy = add_gaussian_noise(I, variance=0.1)

    plt.figure(2)
    plt.title('Noisy Image')
    plt.imshow(noisy.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # Denoise by iterative diffusion smoothing
    denoised = denoise_diffusion(noisy, iterations=30, dt=0.2)

    plt.figure(3)
    plt.title('Denoised Image After Diffusion')
    plt.imshow(denoised.astype(np.uint8), cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()

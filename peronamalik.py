import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
I = cv2.imread('brain.bmp', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
Ny, Nx = I.shape

# Display original image
plt.figure(1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Add Gaussian noise
noise_variance = 0.2  # Variance for Gaussian noise
J = I + np.random.normal(0, noise_variance * 255, I.shape).astype(np.uint8)  # Add Gaussian noise
plt.figure(2)
plt.imshow(J, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

J = J.astype(np.float64)  # Convert to double
method = 'pm2'
K = 10
iterations = 50
dt = 0.2

for i in range(iterations):
    # Calculate gradient in all directions (N, S, E, W)
    In = np.vstack((J[0, :], J[:-1, :])) - J
    Is = np.vstack((J[1:, :], J[-1, :])) - J
    Ie = np.hstack((J[:, 1:], J[:, -1].reshape(-1, 1))) - J
    Iw = np.hstack((J[:, 0].reshape(-1, 1), J[:, :-1])) - J
    
    if method == 'pm1':
        Cn = np.exp(-(np.abs(In) / K) ** 2)
        Cs = np.exp(-(np.abs(Is) / K) ** 2)
        Ce = np.exp(-(np.abs(Ie) / K) ** 2)
        Cw = np.exp(-(np.abs(Iw) / K) ** 2)
    elif method == 'pm2':
        Cn = 1 / (1 + (np.abs(In) / K) ** 2)
        Cs = 1 / (1 + (np.abs(Is) / K) ** 2)
        Ce = 1 / (1 + (np.abs(Ie) / K) ** 2)
        Cw = 1 / (1 + (np.abs(Iw) / K) ** 2)
    else:
        raise ValueError(f'Unknown method "{method}"')

    J += dt * (Cn * In + Cs * Is + Ce * Ie + Cw * Iw)

# Display the final image
plt.figure(3)
plt.imshow(J, cmap='gray')
plt.title('Processed Image')
plt.axis('off')
plt.show()

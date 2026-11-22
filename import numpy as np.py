import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# Load image (gunakan path absolut atau pastikan file ada di direktori yang sama)
image_path = 'brain.bmp'
I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cek apakah file berhasil dibaca
if I is None:
    print(f"Gagal membaca file gambar: {image_path}")
    print("Periksa apakah path sudah benar dan file tidak rusak.")
    sys.exit()  # Hentikan program jika gambar tidak terbaca

# Ambil ukuran gambar
Ny, Nx = I.shape

# Tampilkan gambar asli
plt.figure(1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Tambahkan noise Gaussian
noise_sigma = 0.2  # Deviasi standar noise Gaussian
J = I + np.random.normal(0, noise_sigma * 255, I.shape).astype(np.uint8)
plt.figure(2)
plt.imshow(J, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

# Ubah tipe data ke float64 untuk diproses
J = J.astype(np.float64)
method = 'pm2'     # Pilihan metode: 'pm1' atau 'pm2'
K = 10             # Parameter konduktivitas
iterations = 50    # Jumlah iterasi
dt = 0.2           # Time step

# Proses anisotropic diffusion
for i in range(iterations):
    # Hitung gradien arah utara, selatan, timur, barat
    In = np.vstack((J[0, :], J[:-1, :])) - J
    Is = np.vstack((J[1:, :], J[-1, :])) - J
    Ie = np.hstack((J[:, 1:], J[:, -1].reshape(-1, 1))) - J
    Iw = np.hstack((J[:, 0].reshape(-1, 1), J[:, :-1])) - J

    # Hitung koefisien difusi
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
        raise ValueError(f'Metode tidak dikenal: "{method}"')

    # Update citra
    J += dt * (Cn * In + Cs * Is + Ce * Ie + Cw * Iw)

# Tampilkan hasil akhir
plt.figure(3)
plt.imshow(J, cmap='gray')
plt.title('Processed Image after Anisotropic Diffusion')
plt.axis('off')
plt.show()

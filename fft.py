import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import os
from scipy.sparse import save_npz
import scipy

def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Fast Fourier Transform and Applications')

    # Define the command-line arguments
    parser.add_argument('-m', '--mode', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Mode: 1 - Display FFT, 2 - Denoise, 3 - Compress and plot, 4 - Plot runtime graphs')
    parser.add_argument('-i', '--image', type=str, default='moonlanding.png',
                        help='Filename of the image to be processed')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    mode = args.mode
    image_filename = args.image
    return (mode, image_filename)

def resize(image):
    # Calculate the nearest power of 2 dimensions
    height, width = image.shape[:2]
    new_height = 2 ** int(np.ceil(np.log2(height)))
    new_width = 2 ** int(np.ceil(np.log2(width)))

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def naive_dft(x):
    """
    Compute the 1D Discrete Fourier Transform (DFT) of input signal x using the naive approach.
    
    Parameters:
        x (np.ndarray): Input signal.
    
    Returns:
        np.ndarray: DFT of the input signal.
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    omega = np.exp(-2j * np.pi * k * n / N)
    return np.dot(omega, x)

def fft_cooley_tukey(x):
    """
    Compute the 1D Fast Fourier Transform (FFT) of input signal x using the Cooley-Tukey algorithm
    with the naive DFT applied at the end.
    
    Parameters:
        x (np.ndarray): Input signal.
    
    Returns:
        np.ndarray: FFT of the input signal.
    """
    N = len(x)
    
    # Base case: if the input size is 1, return the input itself
    if N <= 32:
        return naive_dft(x)
    
    # Split the input into even and odd indices
    even_indices = x[::2]
    odd_indices = x[1::2]
    
    # Recursively compute FFT for even and odd indices
    fft_even = fft_cooley_tukey(even_indices)
    fft_odd = fft_cooley_tukey(odd_indices)
    
    # Combine results using naive DFT instead of butterfly operation
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    fft_combined = np.concatenate([fft_even + t[:N//2] * fft_odd, fft_even + t[N//2:] * fft_odd])
    
    return fft_combined

def fft2d(x, naive = False):
    """
    Compute the 2D Fast Fourier Transform (FFT) of input signal x.
    
    Parameters:
        x (np.ndarray): Input signal (2D array).
    
    Returns:
        np.ndarray: 2D FFT of the input signal.
    """
    # Apply 1D FFT to rows
    if naive:
        # Apply 1D FFT to rows
        rows_fft = np.apply_along_axis(naive_dft, axis=1, arr=x)
        # Apply 1D FFT to columns
        cols_fft = np.apply_along_axis(naive_dft, axis=0, arr=rows_fft)
    else:
        # Apply 1D FFT to rows
        rows_fft = np.apply_along_axis(fft_cooley_tukey, axis=1, arr=x)
        # Apply 1D FFT to columns
        cols_fft = np.apply_along_axis(fft_cooley_tukey, axis=0, arr=rows_fft)
    
    return cols_fft


def ifft(x):
    """
    Compute the 1D Inverse Discrete Fourier Transform (IDFT) of input signal x using the naive approach.

    Parameters:
        x (np.ndarray): Frequency-domain signal.

    Returns:
        np.ndarray: Time-domain signal (IDFT of the input signal).
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    omega = np.exp(2j * np.pi * k * n / N)
    return np.dot(omega, x) / N


def ifft2d(x):
    """
    Compute the 2D Inverse Fast Fourier Transform (IFFT) of input signal x.

    Parameters:
        x (np.ndarray): Frequency-domain signal (2D array).

    Returns:
        np.ndarray: Time-domain signal (2D IFFT of the input signal).
    """
    # Apply 1D IFFT to rows
    rows_ifft = np.apply_along_axis(ifft, axis=1, arr=x)

    # Apply 1D IFFT to columns
    cols_ifft = np.apply_along_axis(ifft, axis=0, arr=rows_ifft)

    return cols_ifft

def denoise(img):
    # Perform FFT on the image
    fft_image = np.fft.fft2(img)

    # Get the dimensions of the FFT image
    rows, cols = fft_image.shape

    # We want to zero out frequencies in the middle 80% of the image 
    # (since we keep bottom 10% and top 10%)
    # We do so for both rows and columns
    fft_image[int(rows*0.1):int(rows*(0.9))] = 0
    fft_image[:, int(cols*0.1):int(cols*(0.9))] = 0

    denoised_img = ifft2d(fft_image)

    # Total number of pixels: rows * cols
    num_zeros = rows * cols - np.count_nonzero(denoised_img)
    print("Number of zeros in the denoised image: ", num_zeros)

    return denoised_img.astype(np.uint8)

def run_compression(img):
    # Perform FFT on the image
    fft_image = fft2d(img)
    # Coefficient of compressions (in percentages)
    coefficients = [0, 15, 40, 60, 75, 99.9]
    num_non_zeros = []
    file_sizes = []  # List to store file sizes
    image_index = 1
    results = []
    
    for percentage in coefficients:
        copy = fft_image.copy()
        cutoff = np.percentile(abs(copy), percentage)
        copy[abs(copy) < cutoff] = 0
        num_non_zeros.append(np.count_nonzero(copy))
        
        # Save sparse matrix to an npz file
        sparse_matrix = scipy.sparse.csr_matrix(copy)
        save_npz(f'compressed_matrix_{percentage}.npz', sparse_matrix)
        
        # Calculate and store the file size
        file_size = os.path.getsize(f'compressed_matrix_{percentage}.npz')
        file_sizes.append(file_size)
        
        results.append(ifft2d(copy).astype(np.uint8))
        image_index += 1

    # Plot the original and compressed images
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(results):
        plt.subplot(2, 3, i+1)
        plt.imshow(image, cmap="gray")
        plt.title(f"{coefficients[i]}% Compression\nSize: {file_sizes[i] / 1024:.2f} KB")
        plt.axis('off')
    plt.show()

def display_fft(org_img, trfm_img):
    # Magnitude spectrum
    title="Fourier Transform (Log Scaled)"
    magnitude_spectrum_org = np.abs(trfm_img)

    # Plot original image and its Fourier Transform
    plt.figure(figsize=(10, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(org_img, cmap='gray')
    plt.title('Original Image')

    # Fourier Transform with log scale
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum_org, cmap='gray', norm=LogNorm())
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

def display_denoised(org_img, trfm_img):
    # Display the original and 2D Inverse FFT images using matplotlib

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1), plt.imshow(org_img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2), plt.imshow(
        trfm_img.astype(np.uint8), cmap="gray"
    )
    plt.title("2D Inverse FFT of Image"), plt.xticks([]), plt.yticks([])

    plt.show()

def runtime_comparison(mode=4, min_size_exp=5, max_size_exp=10, num_trials=10):
    """
    Perform runtime complexity comparison between FFT and naive DFT for different problem sizes.

    Parameters:
        mode (int): The mode of operation (4 for runtime comparison).
        min_size_exp (int): The minimum exponent for the problem size (2**min_size_exp).
        max_size_exp (int): The maximum exponent for the problem size (2**max_size_exp).
        num_trials (int): Number of trials for each problem size.

    Returns:
        None
    """
    problem_sizes = [2**i for i in range(min_size_exp, max_size_exp + 1)]

    fft_runtimes = []
    naive_dft_runtimes = []

    for size in problem_sizes:
        # Generate a random 2D array of size x size
        test_input = np.random.rand(size, size)

        fft_times = []
        naive_dft_times = []

        for _ in range(num_trials):
            # Measure runtime for FFT
            start_time = time.time()
            fft_result = fft2d(test_input)
            end_time = time.time()
            fft_times.append(end_time - start_time)

            # Measure runtime for naive DFT
            start_time = time.time()
            naive_dft_result = fft2d(test_input, True)
            end_time = time.time()
            naive_dft_times.append(end_time - start_time)

        # Calculate mean and standard deviation for each algorithm
        mean_fft_time = np.mean(fft_times)
        mean_naive_dft_time = np.mean(naive_dft_times)
        std_fft_time = np.std(fft_times)
        std_naive_dft_time = np.std(naive_dft_times)

        # Record mean runtime for each algorithm
        fft_runtimes.append(mean_fft_time)
        naive_dft_runtimes.append(mean_naive_dft_time)

        # Print information
        print(f"Size: {size}, FFT Mean Time: {mean_fft_time:.6f} seconds, FFT Std Dev: {std_fft_time:.6f} seconds")
        print(f"Size: {size}, Naive DFT Mean Time: {mean_naive_dft_time:.6f} seconds, Naive DFT Std Dev: {std_naive_dft_time:.6f} seconds")

    # Create a plot
    # Plot the run times
    plt.plot(problem_sizes, fft_runtimes, label='FFT', marker='o')
    plt.plot(problem_sizes, naive_dft_runtimes, label='Naive DFT', marker='o')
    # Plot the error bars
    plt.errorbar(problem_sizes, fft_runtimes, yerr=2*std_fft_time, fmt='o', capsize=5, label='FFT Deviance', color='blue')
    plt.errorbar(problem_sizes, naive_dft_runtimes, yerr=2*std_naive_dft_time, fmt='o', capsize=5, label='Naive DFT Deviance')

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Problem Size (log scale)')
    plt.ylabel('Mean Runtime (log scale) in s')
    plt.title('Runtime Complexity Comparison: FFT vs Naive DFT')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parse the command-line arguments
    mode, file_name = parse_arguments() 

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    r_img = resize(img)

    # Perform actions based on the selected mode
    if mode == 1:
        print(f"Displaying FFT of the image: {file_name}")
        # Call the function to display FFT
        fft_transform = fft2d(r_img)
        fft_transform = np.fft.fftshift(fft_transform)
        display_fft(r_img, fft_transform)
    elif mode == 2:
        print(f"Denoising the image: {file_name}")
        # Call the function for denoising
        denoised_img = denoise(r_img)
        display_denoised(r_img, denoised_img)
    elif mode == 3:
        print(f"Compressing and plotting the image: {file_name}")
        run_compression(r_img)
    elif mode == 4:
        print("Plotting runtime graphs")
        # Call the function for plotting runtime graphs
        runtime_comparison()
    else:
        print("Invalid mode. Please choose a valid mode: 1, 2, 3, or 4.")


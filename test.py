import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift


def fft(x):
    """
    Compute the 1D Fast Fourier Transform (FFT) of input signal x.

    Parameters:
        x (np.ndarray): Input signal.

    Returns:
        np.ndarray: FFT of the input signal.
    """
    N = len(x)

    # Base case: if the input size is 1, return the input itself
    if N == 1:
        return x

    # Split the input into even and odd indices
    even_indices = x[::2]
    odd_indices = x[1::2]

    # Recursively compute FFT for even and odd indices
    fft_even = fft(even_indices)
    fft_odd = fft(odd_indices)

    # Combine results using FFT butterfly operation
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    fft_combined = np.concatenate(
        [fft_even + t[: N // 2] * fft_odd, fft_even + t[N // 2 :] * fft_odd]
    )

    return fft_combined

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
    M = np.exp(-2j * np.pi * n * k / N)
    return np.dot(M, x)


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
    
    # Base case: if the input size is small, use the naive DFT
    if N <= 16:
        return naive_dft(x)
    
    # Cooley-Tukey FFT implementation
    even_indices = x[::2]
    odd_indices = x[1::2]
    
    # Recursively compute FFT for even and odd parts
    fft_even = fft_cooley_tukey(even_indices)
    fft_odd = fft_cooley_tukey(odd_indices)
    
    # Pre-allocate a list to hold the results
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    fft_combined = np.concatenate(
        [fft_even + t[: N // 2] * fft_odd, fft_even + t[N // 2 :] * fft_odd]
    )

    return fft_combined

def fft2d(x):
    """
    Compute the 2D Fast Fourier Transform (FFT) of input signal x.

    Parameters:
        x (np.ndarray): Input signal (2D array).

    Returns:
        np.ndarray: 2D FFT of the input signal.
    """
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


def test_signal():
    # Test signal
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Calculate FFT using your implementation
    your_fft_result = fft_cooley_tukey(x)

    # Calculate FFT using NumPy
    numpy_fft_result = np.fft.fft(x)

    # Compare the results
    print("Your FFT Result:", your_fft_result)
    print("NumPy FFT Result:", numpy_fft_result)

    # Check if the results are close within a certain tolerance
    tolerance = 1e-10
    if np.allclose(your_fft_result, numpy_fft_result, rtol=tolerance, atol=tolerance):
        print("Results are close. Your FFT implementation seems correct.")
    else:
        print("Results differ. Please double-check your FFT implementation.")


def test_signal2():
    # Test signal
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Calculate FFT using your implementation
    your_fft_result = naive_dft(x)

    # Calculate Inverse FFT using your implementation
    your_ifft_result = ifft(your_fft_result)

    print(x)
    print(your_ifft_result)

    # Check if the original signal is recovered
    tolerance = 1e-10
    if np.allclose(x, your_ifft_result, rtol=tolerance, atol=tolerance):
        print("Original signal is successfully recovered with IFFT.")
    else:
        print("IFFT implementation issue: Original signal not recovered.")


def test_images():
    # Read an image using cv2
    image_path = "flash.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load the image.")
    else:
        # Apply 2D FFT to the image
        fft_result_2d = fft2d(image)

        reference_fft_result = np.fft.fft2(image)

        # Check if the results are close within a certain tolerance
        tolerance = 1e-10
        if np.allclose(
            fft_result_2d, reference_fft_result, rtol=tolerance, atol=tolerance
        ):
            print("Results are close. Your 2D FFT implementation seems correct.")
        else:
            print("Results differ. Please double-check your 2D FFT implementation.")

        # Display the original and 2D FFT images using matplotlib
        plt.subplot(1, 2, 1), plt.imshow(image, cmap="gray")
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2), plt.imshow(np.log(1 + np.abs(fft_result_2d)), cmap="gray")
        plt.title("2D FFT of Image"), plt.xticks([]), plt.yticks([])

        plt.show()


def test_images2():
    # Read an image using cv2
    image_path = "flash.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load the image.")
    else:
        # Apply 2D FFT to the image
        fft_result = np.fft.fft2(image)

        inverse_fft_result = ifft2d(fft_result)

        # Check if the results are close within a certain tolerance
        tolerance = 1e-10
        if np.allclose(inverse_fft_result, image, rtol=tolerance, atol=tolerance):
            print("Results are close. Your 2D FFT implementation seems correct.")
        else:
            print("Results differ. Please double-check your 2D FFT implementation.")

        # Display the original and 2D Inverse FFT images using matplotlib
        plt.subplot(1, 2, 1), plt.imshow(image, cmap="gray")
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2), plt.imshow(
            inverse_fft_result.astype(np.uint8), cmap="gray"
        )
        plt.title("2D Inverse FFT of Image"), plt.xticks([]), plt.yticks([])

        plt.show()


def ref_test():
    # Load the image
    image = plt.imread(
        "flash.png"
    )  # Replace 'path/to/your/image.jpg' with the actual path to your image

    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    # Perform 2D FFT
    fft_result = fft2(image)

    # Shift zero frequency components to the center
    fft_result_shifted = fftshift(fft_result)

    # Calculate magnitude spectrum (log-scaled for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1)

    # Display the original image and its magnitude spectrum
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum (2D FFT)")
    plt.show()

# test_signal()

# test_signal2()

# test_images()

# test_images2()

def resize(image):
    # Calculate the nearest power of 2 dimensions
    height, width = image.shape[:2]
    new_height = 2 ** int(np.ceil(np.log2(height)))
    new_width = 2 ** int(np.ceil(np.log2(width)))

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def test_exp1():
    # Load the image
    image = cv2.imread('moonlanding.png', cv2.IMREAD_GRAYSCALE)
    image = resize(image)
    # Apply 2D FFT to the image
    fft_image = np.fft.fft2(image)

    # Shift the zero frequency component to the center
    fft_image_shifted = np.fft.fftshift(fft_image)

   # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fft_image_shifted)

    # Display the original image and FFT magnitude spectrum
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Fourier Transform (np.fft.fft2()) (Log Scaled)')
    plt.axis('off')

    plt.show()

test_exp1()
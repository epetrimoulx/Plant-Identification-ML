import numpy as np
import nibabel as nib
import os
import pandas as pd

# Define paths
data_folder = "IXI-T1.nosync"
slice_index = 135  # Middle slice

k_space_data = []
slice_offsets = [-2, -1, 0, 1, 2]

# Loop through all .nii files in the folder
for file in os.listdir(data_folder):
    if file.endswith(".nii"):
        file_path = os.path.join(data_folder, file)
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()

        # Extract the middle slice
        slice_2D = img_data[:, slice_index, :]

        image_height, image_width, num_slices = img_data.shape

        for offset in slice_offsets:
            slice_index = min(max(slice_index + offset, 0), num_slices - 1)  # Ensure valid slice index
            slice_2D = img_data[:, slice_index, :]

            # Compute 2D Fourier Transform
            k_space = np.fft.fft2(slice_2D)
            k_space = np.fft.fftshift(k_space)  # Center the low-frequency components

            # Store real and imaginary components separately
            k_space_real = np.real(k_space).flatten()
            k_space_imag = np.imag(k_space).flatten()

            k_space_data.append(np.concatenate([k_space_real, k_space_imag]))

# Convert to DataFrame and save
df = pd.DataFrame(k_space_data)
image_height, image_width = slice_2D.shape
print(f"Detected Image Shape: {image_height} x {image_width}")

df.columns = [f"({x},{y},Re)" for x in range(image_height) for y in range(image_width)] + \
             [f"({x},{y},Im)" for x in range(image_height) for y in range(image_width)]

df.to_parquet("MRI_kspace.parquet", compression="zstd")

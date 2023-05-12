from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import sys
from skimage.measure import block_reduce, label
from tqdm import tqdm
import h5py


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title("use scroll wheel to navigate images")

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(np.abs(self.X[:, :, self.ind]))
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(np.abs(self.X[:, :, self.ind]))
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_volume(volume_data):
    # Check if values are complex
    if np.iscomplexobj(volume_data):

        # Take absolute value of volume data
        volume_data = np.abs(volume_data)

    # Check dimensions of volume data
    if len(volume_data.shape) != 3:

        # Remove dimension with length 1
        volume_data = np.squeeze(volume_data)

    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volume_data)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

    plt.show()


def plot_slice_in_orientation_from_volume(volume_data, orientation, slice_number):
    if orientation == 0:
        slice_data = volume_data[slice_number, :, :]
    elif orientation == 1:
        slice_data = volume_data[:, slice_number, :]
    elif orientation == 2:
        slice_data = volume_data[:, :, slice_number]
    else:
        print("Invalid orientation")
        return

    plt.imshow(slice_data, cmap="gray")
    plt.show()


def get_first_nifti_file_in_dir(path):
    nifti_files = [f for f in os.listdir(
        path) if f.endswith(".nii.gz") and "trans" in f]
    if len(nifti_files) == 0:
        nifti_files = [f for f in os.listdir(path) if f.endswith(".nii.gz")]
    return nifti_files[0]


def downsample_slice_data(slice_data, downsample_rate):
    # Downsample the slice data
    downsampled_slice_data = slice_data[::downsample_rate, ::downsample_rate]
    return downsampled_slice_data


def get_slice_from_dimension(volume_data, dimension, slice_number):
    # Get the slice from the volume data
    if dimension == 0:
        slice_data = volume_data[slice_number, :, :]
    elif dimension == 1:
        slice_data = volume_data[:, slice_number, :]
    elif dimension == 2:
        slice_data = volume_data[:, :, slice_number]
    else:
        print("Invalid dimension")
        return
    return slice_data


def downsample_volume(volume, downsample_rate):
    # Define the downsampling factor based on the downsample rate
    factor = int(downsample_rate)

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=(factor, factor, factor), func=np.mean)

    # Normalize the volume to be between 0 and 1
    downsampled_volume = downsampled_volume - np.min(downsampled_volume)
    downsampled_volume = downsampled_volume / np.max(downsampled_volume)

    # Return the downsampled volume
    return downsampled_volume


def save_stack_in_directory(volume_data, niftii_filename, output_path, new_voxel_spacing=None):
    # Create nifti file with new voxel spacing
    niftii_data = nib.Nifti1Image(volume_data, np.eye(4))

    # Set the new voxel spacing if provided
    if new_voxel_spacing is not None:
        niftii_data.header.set_zooms(new_voxel_spacing)

    # Save the nifti file
    nib.save(niftii_data, os.path.join(output_path, niftii_filename))


def compute_brain_mask(volume, threshold=None, min_size=100):
    # Compute threshold value using Otsu's algorithm if not provided
    if threshold is None:
        threshold = threshold_otsu(volume)
        print("Calculated Otsu threshold:", threshold)

    # Apply threshold to the volume
    mask = np.zeros_like(volume, dtype=np.int64)
    mask[volume > threshold] = 1

    # Fill any holes inside the brain in all dimensions
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])
    for i in range(mask.shape[1]):
        mask[:, i, :] = binary_fill_holes(mask[:, i, :])
    for i in range(mask.shape[0]):
        mask[i, :, :] = binary_fill_holes(mask[i, :, :])

    # Remove any small disconnected regions
    mask = mask > 0
    # mask = remove_small_objects(label(mask), min_size=min_size)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = np.where(mask == True, 1.0, 0.0)

    return mask


def normalize_volume(volume):
    # Normalize the volume to be between 0 and 1
    volume = volume - np.min(volume)
    volume = volume / np.max(volume)

    return volume


def load_nifti_file(niftiPath):
    # Open the nifti file
    nifti = nib.load(niftiPath)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    return nifti_data


def plot_abs_angle_real_imag_from_complex_volume(volume):
    # Check if values are complex
    if np.iscomplexobj(volume):
        # Get the absolute value, angle, real, and imaginary parts of the volume
        abs_volume = np.abs(volume)
        angle_volume = np.angle(volume)
        real_volume = np.real(volume)
        imag_volume = np.imag(volume)

        # Plot the absolute value, angle, real, and imaginary parts of the volume
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(abs_volume[:, :, 0], cmap="gray")
        axes[0, 0].set_title("Absolute value")
        axes[0, 1].imshow(angle_volume[:, :, 0], cmap="gray")
        axes[0, 1].set_title("Angle")
        axes[1, 0].imshow(real_volume[:, :, 0], cmap="gray")
        axes[1, 0].set_title("Real")
        axes[1, 1].imshow(imag_volume[:, :, 0], cmap="gray")
        axes[1, 1].set_title("Imaginary")
        plt.show()

    else:
        print("Volume is not complex")

        # Plot the absolute value of the volume
        fig, ax = plt.subplots(1, 1)
        ax.imshow(volume[:, :, 0], cmap="gray")
        ax.set_title("Absolute value")
        plt.show()


def preprocess_file(nifti_path, nifti_filename, threshold, output_path, debug=False):
    with tqdm(total=3, desc="Preprocessing {}".format(nifti_filename), leave=False) as pbar:

        # Load the nifti file
        nifti_data = load_nifti_file(
            nifti_path)

        pbar.update(1)

        # Normalize the volume
        # nifti_data = normalize_volume(nifti_data)

        if debug:
            print("Volume shape:", nifti_data.shape)
            print("Volume min:", np.min(nifti_data))
            print("Volume max:", np.max(nifti_data))

        pbar.update(1)

        # Compute the brain mask of first orientation
        brain_mask = compute_brain_mask(nifti_data, threshold)

        if debug:
            plot_volume(nifti_data)
            plot_volume(brain_mask)

        pbar.update(1)

        # Create filename for mask (add _mask before extension)
        mask_filename = nifti_filename.split(".")[0]
        mask_filename = nifti_filename + "_mask.nii.gz"

        # Save the mask
        save_stack_in_directory(
            brain_mask, mask_filename, output_path)


def convert_hdf_file_to_nifti(hdf_file_path, output_path, debug=False, normalize_per_slice=False, resolution=None):
    # Open the hdf file
    hdf_data = h5py.File(hdf_file_path, "r")["reconstruction"][
        ()
    ].squeeze()

    # Move dimension with lowest length to the back
    dimension_lowest_length = np.argmin(hdf_data.shape)
    hdf_data = np.moveaxis(hdf_data, dimension_lowest_length, -1)

    if normalize_per_slice:
        # Normalize the volume per slice
        for i in range(hdf_data.shape[2]):
            hdf_data[:, :, i] = hdf_data[:, :, i] / \
                np.max(np.abs(hdf_data[:, :, i]))

    else:
        # Normalize the volume to be between 0 and 1
        hdf_data = hdf_data / np.max(np.abs(hdf_data))

    if debug:
        # Plot volume
        print("File", hdf_file_path)
        print("Volume min:", np.min(np.abs(hdf_data)))
        print("Volume max:", np.max(np.abs(hdf_data)))
        print("Volume shape:", hdf_data.shape)
        plot_abs_angle_real_imag_from_complex_volume(hdf_data)

    # Create nifti file with new voxel spacing
    niftii_data = nib.Nifti1Image(np.abs(hdf_data), np.eye(4))

    # Set the new voxel spacing if provided
    if resolution is not None:
        niftii_data.header.set_zooms(resolution)

    # Save the nifti file
    nib.save(niftii_data, output_path)


def is_hdf_file(file_path):
    try:
        with h5py.File(file_path, 'r'):
            return True
    except OSError:
        return False


def convert_all_hdf_files_in_dir_to_nifti(dir, debug=False, normalize_per_slice=False, resolution=None):
    # Loop over all hdf files in the directory
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if is_hdf_file(file_path):
            # Get the nifti file path
            nifti_file_path = os.path.join(file_path + ".nii.gz")

            # Convert the hdf file to nifti
            convert_hdf_file_to_nifti(
                file_path, nifti_file_path, debug, normalize_per_slice, resolution)


def move_all_nifti_files_to_child_directory_named_input(dir):
    # Create directory named input
    input_dir = os.path.join(dir, "input")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # Loop over all nifti files in the directory
    for file in os.listdir(dir):
        if file.endswith(".nii.gz"):
            # Get the nifti file path
            nifti_file_path = os.path.join(dir, file)

            # Move the nifti file to the input directory
            os.rename(nifti_file_path, os.path.join(input_dir, file))


def move_all_hdf_files_to_child_directory_named_HDF(dir):
    # Create directory named input
    hdf_dir = os.path.join(dir, "HDF")
    if not os.path.exists(hdf_dir):
        os.makedirs(hdf_dir)

    # Loop over all nifti files in the directory
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if is_hdf_file(file_path):
            # Move the nifti file to the input directory
            os.rename(file_path, os.path.join(hdf_dir, file))


def main(args):
    if args.process_all_files:
        # Loop over all subdirectories in the data path
        for subdirectory_name in tqdm(os.listdir(args.data_path), desc="Processing all files"):

            # Ignore if file is ds_store
            if subdirectory_name == ".DS_Store":
                continue

            # Get the subdirectory path
            subdirectory_path = os.path.join(
                args.data_path, subdirectory_name)

            # Convert hdf files to nifti
            convert_all_hdf_files_in_dir_to_nifti(
                subdirectory_path)

            # Get first nifti file in subdirectory
            nifti_filename = get_first_nifti_file_in_dir(
                subdirectory_path)
            nifti_file_path = os.path.join(
                subdirectory_path, nifti_filename)

            # Preprocess the file
            preprocess_file(nifti_file_path, subdirectory_name, args.threshold)

            # Move all nifti files to child directory named input
            move_all_nifti_files_to_child_directory_named_input(
                subdirectory_path)
    else:
        # Get first subdirectoy path in data path
        subdirectory_name = os.listdir(args.data_path)[0] if os.listdir(args.data_path)[
            0] != ".DS_Store" else os.listdir(args.data_path)[1]

        # Get the subdirectory path
        subdirectory_path = os.path.join(args.data_path, subdirectory_name)

        # Convert hdf files to nifti
        convert_all_hdf_files_in_dir_to_nifti(
            subdirectory_path, args.debug, args.normalize_per_slice, args.resolution)

        # Get first nifti file in subdirectory
        nifti_filename = get_first_nifti_file_in_dir(subdirectory_path)

        # Get the nifti file path
        nifti_file_path = os.path.join(subdirectory_path, nifti_filename)

        # Preprocess the file
        preprocess_file(nifti_file_path, nifti_filename,
                        args.threshold, subdirectory_path, args.debug)

        # Move all nifti files to child directory named input
        move_all_nifti_files_to_child_directory_named_input(subdirectory_path)

        # Move all hdf files to child directory named HDF
        move_all_hdf_files_to_child_directory_named_HDF(subdirectory_path)

    tqdm.write("Processing complete.")


if __name__ == "__main__":
    import argparse

    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Prepare data for SVR toolbox")

    # Add arguments
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to directory containing high-resolution NIfTI images")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.085")
    parser.add_argument("-a", "--process_all_files", action="store_true",
                        help="If provided, all files in the data directory will be processed. Otherwise, only one file will be processed.")
    parser.add_argument("-r", "--resolution", type=tuple, default=(1, 1, 1),
                        help="Resolution of the input data. Default: (1, 1, 1)")
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Enable debug mode, only for single file processing (plots volume and mask)")
    parser.add_argument("-nps", "--normalize-per-slice", action="store_true",
                        help="If provided, the volume will be normalized per slice. Otherwise, the volume will be normalized as a whole.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Subsampling data...")
    print("Data path:", args.data_path)
    print("Threshold:", args.threshold)
    print("Process all files:", args.process_all_files)
    main(args)

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


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_volume(volume_data):
    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volume_data)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

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

    plt.imshow(slice_data, cmap='gray')
    plt.show()


def get_first_nifti_file_in_dir(path):
    nifti_files = [f for f in os.listdir(path) if f.endswith('.nii.gz')]
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


def save_stack_in_directory(downsampled_volume_data, orientation, niftii_filename, output_path, new_voxel_spacing=None):
    formatted_niftii_filename = niftii_filename + \
        "_" + str(orientation) + ".nii.gz"
    patient_id = niftii_filename.split("_")[0]

    # Create nifti file with new voxel spacing
    niftii_data = nib.Nifti1Image(downsampled_volume_data, np.eye(4))

    # Set the new voxel spacing if provided
    if new_voxel_spacing is not None:
        niftii_data.header.set_zooms(new_voxel_spacing)

    # Create folder in data/downsampled directory with patient id if it doesn't exist
    path = os.path.join(output_path, patient_id)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the nifti file
    nib.save(niftii_data, os.path.join(path, formatted_niftii_filename))


def compute_brain_mask(volume, threshold=None, min_size=100):
    # Compute threshold value using Otsu's algorithm if not provided
    if threshold is None:
        print(threshold_otsu(volume))
        threshold = threshold_otsu(volume)

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


def load_nifti_file_and_get_new_voxel_spacing(niftiPath):
    # Open the nifti file
    nifti = nib.load(niftiPath)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    # Get absolute value of data
    nifti_data = np.abs(nifti_data)

    return nifti_data


def preprocess_file(nifti_path, nifti_filename, output_path, threshold, debug=False):
    with tqdm(total=5, desc="Preprocessing {}".format(nifti_filename), leave=False) as pbar:

        # Load the nifti file
        nifti_data = load_nifti_file_and_get_new_voxel_spacing(
            nifti_path)

        pbar.update(1)

        # Normalize the volume
        nifti_data = normalize_volume(nifti_data)

        pbar.update(1)

        # Compute the brain mask of first orientation
        brainMask = compute_brain_mask(nifti_data, threshold)

        if debug:
            plot_volume(nifti_data)
            plot_volume(brainMask)

        pbar.update(1)

        # Save the brain mask
        save_stack_in_directory(
            brainMask, 0, nifti_filename + "_mask", output_path)


def main(args):
    if args.process_all_files:
        # Loop over all subdirectories in the data path
        for subdirectory_name in tqdm(os.listdir(args.data_path), desc="Processing all files"):

            # Ignore if file is ds_store
            if subdirectory_name == ".DS_Store":
                continue

            # Get first nifti file in subdirectory
            nifti_file_path = get_first_nifti_file_in_dir(
                os.path.join(args.data_path, subdirectory_name))

            # Preprocess the file
            preprocess_file(nifti_file_path, subdirectory_name,
                            args.output_path, args.downsample_factor, args.threshold)
    else:
        # Get first subdirectoy path in data path
        subdirectory_name = os.listdir(args.data_path)[0] if os.listdir(args.data_path)[
            0] != ".DS_Store" else os.listdir(args.data_path)[1]
        subdirectory_path = os.path.join(args.data_path, subdirectory_name)

        # Get first nifti file in subdirectory
        nifti_file_path = get_first_nifti_file_in_dir(subdirectory_path)

        # Preprocess the file
        preprocess_file(nifti_file_path, subdirectory_name,
                        args.output_path, args.downsample_factor, args.threshold, args.debug)

    tqdm.write("Processing complete.")


if __name__ == "__main__":
    import argparse

    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Downsample high resolution NIfTI image to lower resolution stacks and create brain mask")

    # Add arguments
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='Path to directory containing high-resolution NIfTI images')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path to directory where downsampled NIfTI images will be saved')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.085')
    parser.add_argument('-a', '--process_all_files', action='store_true',
                        help='If provided, all files in the data directory will be processed. Otherwise, only one file will be processed.')
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Enable debug mode, only for single file processing (plots volume and mask)")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Subsampling data...")
    print('Data path:', args.data_path)
    print('Output path:', args.output_path)
    print('Threshold:', args.threshold)
    print('Process all files:', args.process_all_files)
    main(args)

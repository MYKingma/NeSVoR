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


def subsample_slice_data(slice_data, subsample_rate):
    # Subsample the slice data
    subsampled_slice_data = slice_data[::subsample_rate, ::subsample_rate]
    return subsampled_slice_data


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


def subsample_volume(volume, subsample_rate):
    # Define the downsampling factor based on the subsample rate
    factor = int(subsample_rate)

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=(factor, factor, factor), func=np.mean)

    # Normalize the volume to be between 0 and 1
    downsampled_volume = downsampled_volume - np.min(downsampled_volume)
    downsampled_volume = downsampled_volume / np.max(downsampled_volume)

    # Return the downsampled volume
    return downsampled_volume


def save_stack_in_directory(subsampled_volume_data, orientation, niftii_filename, output_path, new_voxel_spacing):
    formatted_niftii_filename = niftii_filename + \
        "_" + str(orientation) + ".nii.gz"
    patient_id = niftii_filename.split("_")[0]

    # Create nifti file with new voxel spacing
    niftii_data = nib.Nifti1Image(subsampled_volume_data, np.eye(4))
    niftii_data.header.set_zooms(new_voxel_spacing)

    # Create folder in data/subsampled directory with patient id if it doesn't exist
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


def load_nifti_file_and_get_new_voxel_spacing(niftiPath, subsample_rate):
    # Open the nifti file
    nifti = nib.load(niftiPath)

    # Get voxel spacing
    voxel_spacing = nifti.header.get_zooms()

    # Calculate new voxel spacing
    new_voxel_spacing = (voxel_spacing[0] * subsample_rate,
                         voxel_spacing[1] * subsample_rate, voxel_spacing[2] * subsample_rate)

    # Round the new voxel spacing to 1 decimal place
    new_voxel_spacing = tuple(round(x, 1) for x in new_voxel_spacing)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    # Get absolute value of data
    nifti_data = np.abs(nifti_data)

    return nifti_data, new_voxel_spacing


def preprocess_file(nifti_path, nifti_filename, output_path, subsample_rate, threshold, debug=False):
    with tqdm(total=5, desc="Preprocessing {}".format(nifti_filename), leave=False) as pbar:

        # Load the nifti file
        nifti_data, new_voxel_spacing = load_nifti_file_and_get_new_voxel_spacing(
            nifti_path, subsample_rate)

        pbar.update(1)

        # Normalize the volume
        nifti_data = normalize_volume(nifti_data)

        pbar.update(1)

        # Subsample the volume data
        subsampled_volume_data_x = subsample_volume(nifti_data, subsample_rate)
        subsampled_volume_data_y = subsample_volume(nifti_data, subsample_rate)
        subsampled_volume_data_z = subsample_volume(nifti_data, subsample_rate)

        pbar.update(1)

        # Save the subsampled volume data
        save_stack_in_directory(subsampled_volume_data_x, 0,
                                nifti_filename, output_path, new_voxel_spacing)
        save_stack_in_directory(subsampled_volume_data_y, 1,
                                nifti_filename, output_path, new_voxel_spacing)
        save_stack_in_directory(subsampled_volume_data_z, 2,
                                nifti_filename, output_path, new_voxel_spacing)

        pbar.update(1)

        # Compute the brain mask of first orientation
        brainMask = compute_brain_mask(subsampled_volume_data_x, threshold)

        if debug:
            plot_volume(subsampled_volume_data_x)
            plot_volume(brainMask)

        pbar.update(1)

        # Save the brain mask
        save_stack_in_directory(
            brainMask, 0, nifti_filename + "_mask", output_path, new_voxel_spacing)


def main(args):
    if args.process_all_files:
        # Loop over files in directory
        for filename in tqdm(os.listdir(args.data_path), desc="Processing all files"):

            # Ignore if file is ds_store
            if filename == ".DS_Store":
                continue

            # Get nifti filepath
            nifti_file_path = os.path.join(args.data_path, filename)

            # Preprocess the file
            preprocess_file(nifti_file_path, filename,
                            args.output_path, args.subsample_factor, args.threshold)
    else:
        # Get first nifti file in directory
        filename = get_first_nifti_file_in_dir(args.data_path)
        nifti_file_path = os.path.join(args.data_path, filename)

        # Preprocess the file
        preprocess_file(nifti_file_path, filename,
                        args.output_path, args.subsample_factor, args.threshold, args.debug)

    tqdm.write("Processing complete.")


if __name__ == "__main__":
    import argparse

    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Subsample high resolution NIfTI image to lower resolution stacks and create brain mask")

    # Add arguments
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='Path to directory containing high-resolution NIfTI images')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path to directory where subsampled NIfTI images will be saved')
    parser.add_argument('-s', '--subsample_factor', type=float, default=2,
                        help='Factor at which to subsample the images (e.g. 2 for 50%% subsampling)')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.085')
    parser.add_argument('-a', '--process_all_files', action='store_true',
                        help='If provided, all files in the data directory will be processed. Otherwise, only one file will be processed.')
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Enable debug mode, only for single file processing (plots subsampled volume and mask)")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Subsampling data...")
    print('Data path:', args.data_path)
    print('Output path:', args.output_path)
    print('Subsample factor:', args.subsample_factor)
    print('Threshold:', args.threshold)
    print('Process all files:', args.process_all_files)
    main(args)

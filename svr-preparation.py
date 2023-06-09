from scipy.ndimage import binary_fill_holes, affine_transform
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


def get_first_nifti_file_in_dir(args):
    nifti_files = [f for f in os.listdir(
        args.data_path) if f.endswith(".nii.gz") and "trans" in f and "sag" not in f]
    if len(nifti_files) == 0:
        nifti_files = [f for f in os.listdir(args.data_path) if f.endswith(".nii.gz") and "sag" not in f]
    return nifti_files[0]


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


def save_stack_in_directory(volume_data, nifti_filename, output_path, new_voxel_spacing, args):
    if not args.nifti_template and not args.nifti_template_sagittal:
        if args.debug:
            print("Volume data shape before permutation:", volume_data.shape)

        # Move slice dimension to the end (slice dimension is the dimension with the smallesr size)
        slice_dimension_index = np.argmin(volume_data.shape)

        if slice_dimension_index == 0:
            volume_data = np.transpose(volume_data, (2, 1, 0))
        elif slice_dimension_index == 1:
            volume_data = np.transpose(volume_data, (2, 0, 1))

        if args.debug:
            print("Volume data shape after permutation:", volume_data.shape)

    # Set the new voxel spacing if provided
    if new_voxel_spacing is not None:
        # Get index of the dimension with the smallest size
        min_dimension_index = np.argmin(volume_data.shape)

        # Get the highest value voxel spacing value
        max_voxel_value = np.max(new_voxel_spacing)

        # Move the highest voxel spacing value to the index of the dimension with the smallest size
        new_voxel_spacing = np.delete(new_voxel_spacing, 2)
        new_voxel_spacing = np.insert(
            new_voxel_spacing, min_dimension_index, max_voxel_value, axis=0)
        
        if args.debug:
            print("New voxel spacing:", new_voxel_spacing)

    # Create transformation matrix
    transformation_matrix = create_transformation_matrix_nifti(volume_data.shape, new_voxel_spacing, nifti_filename, args)

    if args.debug:
        transformed_volume_data = apply_transformations_to_data(volume_data, nifti_filename, args)

        # Plot the volume data in all orientations
        plot_orientations(transformed_volume_data, new_voxel_spacing, nifti_filename)

    if args.int16:
        # Convert to int16
        volume_data = (volume_data * 32767).astype(np.int16)

    # Create nifti file with new voxel spacing
    nifti_data = nib.Nifti1Image(volume_data, transformation_matrix)

    if not args.nifti_template and not args.nifti_template_sagittal:
        # Set the new voxel spacing
        nifti_data.header.set_zooms(new_voxel_spacing)

        # Save the nifti file
        nib.save(nifti_data, os.path.join(output_path, nifti_filename))
    
    else:
        if "sag" in nifti_filename:
            nifti_data = nib.load(args.nifti_template_sagittal)
        else:
            nifti_data = nib.load(args.nifti_template)

        nifti_image = nifti_data.get_fdata()
        nifti_image = volume_data
        
        new_nifti_data = nib.Nifti1Image(nifti_image, nifti_data.affine, nifti_data.header)

        nib.save(new_nifti_data, os.path.join(output_path, nifti_filename))


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


def set_slice_in_dimension(volume, dimension, slice_number, slice_data):
    # Set the slice in the volume data
    if dimension == 0:
        volume[slice_number, :, :] = slice_data
    elif dimension == 1:
        volume[:, slice_number, :] = slice_data
    elif dimension == 2:
        volume[:, :, slice_number] = slice_data
    else:
        print("Invalid dimension")
        return
    return volume


def normalize_volume(volume, normalize_per_slice=False):
    if normalize_per_slice:
        # Get slice dimension
        slice_dimension = get_lowest_length_dimension(volume.shape)

        # Loop over slice dimension
        for i in range(volume.shape[slice_dimension]):
            # Get the slice data
            slice_data = get_slice_from_dimension(volume, slice_dimension, i)

            # Normalize the slice data to be between 0 and 1
            slice_data = slice_data - np.min(np.abs(slice_data))

            if np.max(np.abs(slice_data)) != 0:
                slice_data = slice_data / np.max(np.abs(slice_data))

            # Set the slice data in the volume
            volume = set_slice_in_dimension(
                volume, slice_dimension, i, slice_data)

    else:
        # Normalize the volume to be between 0 and 1
        volume = volume - np.min(np.abs(volume))

        if np.max(np.abs(volume)) != 0:
            volume = volume / np.max(np.abs(volume))

    return volume


def load_nifti_file(nifti_path):
    # Open the nifti file
    nifti = nib.load(nifti_path)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    # Get the voxel spacing from the nifti file
    nifti_voxel_spacing = nifti.header.get_zooms()

    return nifti_data, nifti_voxel_spacing


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


def get_lowest_length_dimension(shape):
    # Initialize the selected dimension as the first one
    selected_dimension = 0

    # Iterate through the dimensions
    for i in range(1, len(shape)):
        # Check if the current dimension has a lower length
        if shape[i] < shape[selected_dimension]:
            selected_dimension = i
        # Check if the current dimension has the same length but comes after the selected dimension
        elif shape[i] == shape[selected_dimension] and i > selected_dimension:
            selected_dimension = i

    # Return the dimension with the lowest length
    return selected_dimension


def get_highest_length_dimension(shape):
    # Initialize the selected dimension as the first one
    selected_dimension = 0

    # Iterate through the dimensions
    for i in range(1, len(shape)):
        # Check if the current dimension has a lower length
        if shape[i] > shape[selected_dimension]:
            selected_dimension = i
        # Check if the current dimension has the same length but comes after the selected dimension
        elif shape[i] == shape[selected_dimension] and i > selected_dimension:
            selected_dimension = i

    # Return the dimension with the lowest length
    return selected_dimension


def downsample_volume(volume, nifti_voxel_spacing, args):
    if args.debug:
        print("Original volume shape:", volume.shape)

    # Get the voxel spacing to use
    if resolution is None:
        resolution = nifti_voxel_spacing

    # Define the downsampling factor based on the downsample rate
    factor = int(args.downsample_rate)

    # Get the slice dimension
    slice_dimension = get_lowest_length_dimension(volume.shape)

    # Create block size with factors for each dimension
    block_size = [factor, factor, factor]

    if args.ignore_slice_dimension:
        # Set the block size for the slice dimension to 1
        block_size[slice_dimension] = 1

    else:
        if args.slice_thickness is not None:
            # Calculate the downsampling factor for the slice dimension
            slice_factor = int(args.slice_thickness / resolution[slice_dimension])

            # Set the block size for the slice dimension to the slice thickness
            block_size[slice_dimension] = int(slice_factor)

            if args.debug:
                print("Slice thickness:", args.slice_thickness)
                print("Slice factor:", slice_factor)

    if args.debug:
        print("Used block size:", block_size)

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=tuple(block_size), func=np.mean)

    # Calculate the new voxel spacing in integer values
    new_voxel_spacing = np.array(resolution) * np.array(block_size)

    if args.debug:
        print("Downsampled volume shape:", downsampled_volume.shape)
        print("New voxel spacing:", new_voxel_spacing)

    return downsampled_volume, new_voxel_spacing


def preprocess_file(nifti_path, nifti_filename, output_path, args):
    # with tqdm(total=3, desc="Preprocessing {}".format(nifti_filename), leave=False) as pbar:

    # Load the nifti file
    nifti_data, nifti_voxel_spacing = load_nifti_file(
        nifti_path)

    # pbar.update(1)

    # Normalize the volume
    nifti_data = normalize_volume(nifti_data, args.normalize_per_slice)

    if args.debug:
        print("Volume shape:", nifti_data.shape)
        print("Nifti voxel spacing:", nifti_voxel_spacing)
        print("Resolution:", args.used_resolution)
        print("Normalised volume min:", np.min(nifti_data))
        print("Normalised volume max:", np.max(nifti_data))

    if args.downsample:
        # Downsample the volume
        nifti_data, new_voxel_spacing = downsample_volume(
            nifti_data, nifti_voxel_spacing, args)

        # Normalize the volume
        nifti_data = normalize_volume(nifti_data, args)

        nifti_voxel_spacing = new_voxel_spacing

    # Save the downsampled volume
    save_stack_in_directory(
        nifti_data, nifti_filename, output_path, nifti_voxel_spacing, args)

    # pbar.update(1)

    if args.mask:
        # Compute the brain mask of first orientation
        brain_mask = compute_brain_mask(nifti_data, args)

        if args.debug and args.plot:
            plot_volume(nifti_data)
            plot_volume(brain_mask)
            print("Brain mask shape:", brain_mask.shape)

        # pbar.update(1)

        # Create filename for mask (add _mask before extension)
        mask_filename = nifti_filename.split(".")[0]
        mask_filename = nifti_filename + "_mask.nii.gz"

        # Get correct mask voxel spacing
        if args.donwsample_file:
            mask_voxel_spacing = new_voxel_spacing
        else:
            mask_voxel_spacing = nifti_voxel_spacing

        if args.debug:
            print("Mask voxel spacing:", mask_voxel_spacing)

        # Save the mask
        save_stack_in_directory(
            brain_mask, mask_filename, output_path, mask_voxel_spacing, args)
        
def flip_data_with_transformation_matrix(transformation_matrix, filename, args):
    flip_arguments = args.flip if "sag" not in filename else args.flip_sag

    if not flip_arguments:
        return transformation_matrix

    if bool(flip_arguments[0]):
        transformation_matrix[0, 0] = -1 * transformation_matrix[0, 0]
    if bool(flip_arguments[1]):
        transformation_matrix[1, 1] = -1 * transformation_matrix[1, 1]
    if bool(flip_arguments[2]):
        transformation_matrix[2, 2] = -1 * transformation_matrix[2, 2]

    if args.debug:
        print("Flipped transformation matrix:\n", transformation_matrix)

    return transformation_matrix

def transpose_data_with_transformation_matrix(transformation_matrix, filename, args):
    transpose_args = args.transpose if "sag" not in filename else args.transpose_sag

    if not transpose_args:
        return transformation_matrix

    transposed_transformation_matrix = np.eye(4)
    transposed_transformation_matrix[0, :] = transformation_matrix[transpose_args[0], :]
    transposed_transformation_matrix[1, :] = transformation_matrix[transpose_args[1], :]
    transposed_transformation_matrix[2, :] = transformation_matrix[transpose_args[2], :]

    if args.debug:
        print("Transposed transformation matrix:\n", transposed_transformation_matrix)

    return transposed_transformation_matrix


def create_transformation_matrix_nifti(volume_shape, volume_spacing, filename, args):
    # Create a transformation matrix to transform the nifti volume to the correct voxel spacing
    transformation_matrix = np.eye(4)

    # Set the diagonal values to the voxel spacing
    transformation_matrix[0, 0] = volume_spacing[0]
    transformation_matrix[1, 1] = volume_spacing[1]
    transformation_matrix[2, 2] = volume_spacing[2]

    if args.offset or args.offset_sag:
        used_offset = args.offset if "sag" not in filename else args.offset_sag

        if used_offset:
            transformation_matrix[0, 3] = used_offset[0]
            transformation_matrix[1, 3] = used_offset[1]
            transformation_matrix[2, 3] = used_offset[2]

    if args.debug:
        print("Volume shape:", volume_shape)
        print("Transformation matrix:\n", transformation_matrix)

    if args.flip or args.flip_sag:
        transformation_matrix = flip_data_with_transformation_matrix(transformation_matrix, filename, args)

    if args.transpose or args.transpose_sag:
        transformation_matrix = transpose_data_with_transformation_matrix(transformation_matrix, filename, args)

    return transformation_matrix


def convert_hdf_file_to_nifti(hdf_file_path, output_path, args):
    # Open the hdf file
    hdf_data = h5py.File(hdf_file_path, "r")["reconstruction"][
        ()
    ].squeeze()

    if args.debug:
        # Plot volume
        print("File", hdf_file_path)
        print("Volume min:", np.min(np.abs(hdf_data)))
        print("Volume max:", np.max(np.abs(hdf_data)))
        print("Volume shape:", hdf_data.shape)

        if args.plot:
            plot_abs_angle_real_imag_from_complex_volume(hdf_data)

    # Create nifti file with new voxel spacing
    nifti_data = nib.Nifti1Image(np.abs(hdf_data), np.eye(4))

    # Set the new voxel spacing if provided
    if args.used_resolution is not None:
        nifti_data.header.set_zooms(args.used_resolution)

    # Save the nifti file
    nib.save(nifti_data, output_path)


def is_hdf_file(file_path):
    try:
        with h5py.File(file_path, 'r'):
            return True
    except OSError:
        return False


def convert_all_hdf_files_in_dir_to_nifti(args):
    # Loop over all hdf files in the directory
    for file in os.listdir(args.data_path):
        file_path = os.path.join(args.data_path, file)
        args.used_resolution = args.resolution_sagittal if "sag" in file_path else args.resolution
        if is_hdf_file(file_path):
            # Get the nifti file path
            nifti_file_path = os.path.join(file_path + ".nii.gz")

            # Convert the hdf file to nifti
            convert_hdf_file_to_nifti(
                file_path, nifti_file_path, args)


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

def apply_transformations_to_data(volume_data, filename, args):
    transpose_axes = args.transpose if "sag" not in filename else args.transpose_sag
    flip_axes = args.flip if "sag" not in filename else args.flip_sag
    if transpose_axes:
        volume_data = np.transpose(volume_data, transpose_axes)
    if flip_axes:
        formatted_flip_axes = []
        if flip_axes[0]:
            formatted_flip_axes.append(0)
        if flip_axes[1]:
            formatted_flip_axes.append(1)
        if flip_axes[2]:
            formatted_flip_axes.append(2)
        volume_data = np.flip(volume_data, formatted_flip_axes)

    return volume_data

def plot_orientations(volume_data, spacing, filename, slice_index = 25):
    # Get slice dimension spacing, is highest spacing value
    slice_dimension_spacing = np.max(spacing)

    # Get voxexl spacing, is lowest spacing value
    voxel_spacing = np.min(spacing)

    # Plot Axial, Coronal and Sagittal slices, remove axis ticks and labels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume_data[:, :, slice_index], cmap="gray")
    axes[0].set_title("Axial")
    axes[0].axis("off")
    axes[1].imshow(volume_data[:, slice_index, :], cmap="gray")
    axes[1].set_title("Coronal")
    axes[1].axis("off")
    axes[2].imshow(volume_data[slice_index, :, :], cmap="gray")
    axes[2].set_title("Sagittal")
    axes[2].axis("off")

    # Set the spacing of the plotted image
    if "sag" in filename:
        axes[0].set_aspect(slice_dimension_spacing / voxel_spacing)
        axes[1].set_aspect(slice_dimension_spacing / voxel_spacing)
        axes[2].set_aspect(voxel_spacing / voxel_spacing)
    else:
        axes[0].set_aspect(voxel_spacing / voxel_spacing)
        axes[1].set_aspect(voxel_spacing / slice_dimension_spacing)
        axes[2].set_aspect(voxel_spacing / slice_dimension_spacing)

    plt.show()

def move_all_hdf_files_to_child_directory_named_HDF(dir):
    # Check if directory contains hdf files
    hdf_files = [f for f in os.listdir(dir) if is_hdf_file(
        os.path.join(dir, f))]

    if len(hdf_files) > 0:
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
        # for subdirectory_name in tqdm(os.listdir(args.data_path), desc="Processing all files"):
        for subdirectory_name in os.listdir(args.data_path):

            # Ignore if file is ds_store
            if subdirectory_name == ".DS_Store":
                continue

            # Get the subdirectory path
            subdirectory_path = os.path.join(
                args.data_path, subdirectory_name)

            # Convert hdf files to nifti
            convert_all_hdf_files_in_dir_to_nifti(
                subdirectory_path, args)

            # Get first nifti file in subdirectory
            nifti_filename = get_first_nifti_file_in_dir(
                subdirectory_path)
            nifti_file_path = os.path.join(
                subdirectory_path, nifti_filename)
            
            # Set the used resolution
            args.used_resolution = args.resolution

            # Preprocess the file
            preprocess_file(nifti_file_path,
                            nifti_filename, subdirectory_path, args)

            # Get all other nifti files in the subdirectory
            nifti_files = [f for f in os.listdir(subdirectory_path) if f.endswith(
                ".nii.gz") and f != nifti_filename and "mask" not in f]
            
            # Preprocess other nifti files
            for nifti_filename in nifti_files:
                # Get used resolution
                args.used_resolution = args.resolution_sagittal if "sag" in nifti_filename else args.resolution

                # Get the nifti file path
                nifti_file_path = os.path.join(
                    subdirectory_path, nifti_filename)

                # Preprocess the file
                preprocess_file(nifti_file_path, nifti_filename,
                                subdirectory_path, args)

            if not args.debug:
                # Move all nifti files to child directory named input
                move_all_nifti_files_to_child_directory_named_input(
                    subdirectory_path)

                # Move all hdf files to child directory named HDF
                move_all_hdf_files_to_child_directory_named_HDF(subdirectory_path)
    else:
        # Convert hdf files to nifti
        convert_all_hdf_files_in_dir_to_nifti(
            args)

        # Get first nifti file in subdirectory
        nifti_filename = get_first_nifti_file_in_dir(args)

        # Get the nifti file path
        nifti_file_path = os.path.join(args.data_path, nifti_filename)

        # Set the used resolution
        args.used_resolution = args.resolution

        # Preprocess the file
        preprocess_file(nifti_file_path, nifti_filename, args.data_path, args)

        # Get all other nifti files in the subdirectory
        nifti_files = [f for f in os.listdir(args.data_path) if f.endswith(
            ".nii.gz") and f != nifti_filename and "mask" not in f]

        # Preprocess other nifti files
        for nifti_filename in nifti_files:
            # Get used resolution
            args.used_resolution = args.resolution_sagittal if "sag" in nifti_filename else args.resolution

            # Get the nifti file path
            nifti_file_path = os.path.join(args.data_path, nifti_filename)

            # Preprocess the file
            preprocess_file(nifti_file_path, nifti_filename, args.data_path, args)

        if not args.debug:
            # Move all nifti files to child directory named input
            move_all_nifti_files_to_child_directory_named_input(args.data_path)

            # Move all hdf files to child directory named HDF
            move_all_hdf_files_to_child_directory_named_HDF(args.data_path)

    # tqdm.write("Processing complete.")


if __name__ == "__main__":
    import argparse

    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Prepare data for SVR toolbox")
    
    def str_to_bool(value):
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Add arguments
    parser.add_argument("-d", "--data-path", type=str, required=True,
                        help="Path to directory containing high-resolution NIfTI images")
    parser.add_argument("-m", "--mask", action="store_true", default=False, help="If provided, the mask will be created.")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.1")
    parser.add_argument("-a", "--process-all-files", action="store_true",
                        help="If provided, all files in the data directory will be processed. Otherwise, only one file will be processed.")
    parser.add_argument("-nps", "--normalize-per-slice", action="store_true",
                        help="If provided, the volume will be normalized per slice. Otherwise, the volume will be normalized as a whole.")
    parser.add_argument("-r", "--resolution", type=float, nargs=3,
                        help="Resolution of the input data in mm.")
    parser.add_argument("-rs", "--resolution-sagittal", type=float, nargs=3,
                        help="Resolution of the input data in mm for the sagittal orientation.")
    parser.add_argument("-ds", "--downsample", action="store_true",
                        default=False, help="If provided, the input data will be downsampled.")
    parser.add_argument("-dsr", "--downsample-rate", type=int,
                        default=2, help="Downsample rate. Default: 2")
    parser.add_argument("-isd", "--ignore-slice-dimension", action="store_true",
                        default=False, help="If provided, the slice dimension will be ignored when downsampling.")
    parser.add_argument("-st", "--slice-thickness", type=float,
                        default=1, help="Slice thickness used for downsampling. Default: 1")
    parser.add_argument("-nt", "--nifti-template", type=str, default=None, help="Path to nifti file to use as template for the output files. If not provided, the first nifti file in the data directory will be used.")
    parser.add_argument("-nts", "--nifti-template-sagittal", type=str, default=None, help="Path to nifti file to use as template for the output files for the sagittal orientation. If not provided, the first nifti file in the data directory will be used.")
    parser.add_argument("-int16", "--int16", action="store_true",
                        default=False, help="If provided, the output files will be saved as int16. Otherwise, they will be saved as the data type provided.")
    parser.add_argument("-tp", "--transpose", type=int, nargs=3,
                        default=None, help="Transpose the volume. Default: None")
    parser.add_argument("-tps", "--transpose-sag", type=int, nargs=3,
                        default=None, help="Transpose the volume for the sagittal orientation. Default: None")
    parser.add_argument("-fl", "--flip", type=str_to_bool, nargs=3,
                        default=None, help="Flip the volume. Default: None")
    parser.add_argument("-fls", "--flip-sag", type=str_to_bool, nargs=3,
                        default=None, help="Flip the volume for the sagittal orientation. Default: None")
    parser.add_argument("-off", "--offset", type=float, nargs=3,
                        default=None, help="Offset the volume. Default: None")
    parser.add_argument("-offs", "--offset-sag", type=float, nargs=3,
                        default=None, help="Offset the volume for the sagittal orientation. Default: None")
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Enable debug mode, only for single file processing (plots volume and mask)")
    parser.add_argument("-plt", "--plot", action="store_true",
                        help="Enable plotting of the volume and mask during debug mode")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running SVR preparation with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)

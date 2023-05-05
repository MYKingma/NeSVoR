from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import sys
from skimage.measure import block_reduce, label

# Plot volume


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


def plot_volume(volumeData):
    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volumeData)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    plt.show()


def plot_slice_in_orientation_from_volume(volumeData, orientation, sliceNumber):
    if orientation == 0:
        sliceData = volumeData[sliceNumber, :, :]
    elif orientation == 1:
        sliceData = volumeData[:, sliceNumber, :]
    elif orientation == 2:
        sliceData = volumeData[:, :, sliceNumber]
    else:
        print("Invalid orientation")
        return

    plt.imshow(sliceData, cmap='gray')
    plt.show()


def get_first_nifti_file_in_dir(path):
    niftiFiles = [f for f in os.listdir(path) if f.endswith('.nii.gz')]
    return niftiFiles[0]


def subsample_slice_data(sliceData, subsampleRate):
    # Subsample the slice data
    subsampledSliceData = sliceData[::subsampleRate, ::subsampleRate]
    return subsampledSliceData


def get_slice_from_dimension(volumeData, dimension, sliceNumber):
    # Get the slice from the volume data
    if dimension == 0:
        sliceData = volumeData[sliceNumber, :, :]
    elif dimension == 1:
        sliceData = volumeData[:, sliceNumber, :]
    elif dimension == 2:
        sliceData = volumeData[:, :, sliceNumber]
    else:
        print("Invalid dimension")
        return
    return sliceData


def subsample_volume(volume, subsample_rate):
    # Define the downsampling factor based on the subsample rate
    factor = int(subsample_rate)

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=(factor, factor, factor), func=np.mean)

    # Return the downsampled volume
    return downsampled_volume


def save_stack_in_directory(subsampledVolumeData, orientation, niftiiFilename, outputPath):
    newFilename = niftiiFilename + "_" + str(orientation) + ".nii.gz"
    patientId = niftiiFilename.split("_")[0]
    newNifti = nib.Nifti1Image(subsampledVolumeData, np.eye(4))

    # Create folder in data/subsampled directory with patient id if it doesn't exist
    path = os.path.join(outputPath, patientId)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the nifti file
    nib.save(newNifti, os.path.join(path, newFilename))


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


def load_nifti_file(niftiPath):
    # Open the nifti file
    nifti = nib.load(niftiPath)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    # Get absolute value of data
    nifti_data = np.abs(nifti_data)

    return nifti_data


def preprocess_file(nifti_path, nifti_file_name, output_path):
    # Load the nifti file
    nifti_data = load_nifti_file(nifti_path)

    # Normalize the volume
    nifti_data = normalize_volume(nifti_data)

    # Subsample the volume data
    subsampledVolumeDataX = subsample_volume(nifti_data, 2)
    subsampledVolumeDataY = subsample_volume(nifti_data, 2)
    subsampledVolumeDataZ = subsample_volume(nifti_data, 2)

    # Save the subsampled volume data
    save_stack_in_directory(subsampledVolumeDataX, 0,
                            nifti_file_name, output_path)
    save_stack_in_directory(subsampledVolumeDataY, 1,
                            nifti_file_name, output_path)
    save_stack_in_directory(subsampledVolumeDataZ, 2,
                            nifti_file_name, output_path)

    # Compute the brain mask of first orientation
    brainMask = compute_brain_mask(subsampledVolumeDataX, 0.085)

    # Save the brain mask
    save_stack_in_directory(
        brainMask, 0, nifti_file_name + "_mask", output_path)


def main():
    # Get the path to the directory containing the nifti files
    path = sys.argv[1]
    output_path = sys.argv[2]

    # Get parameter which defines if all or one file should be processed
    process_all_files = True
    if len(sys.argv) > 3:
        process_all_files = False

    if process_all_files:
        # Loop over files in directory
        for file_name in os.listdir(path):

            # Ignore if file is ds_store
            if file_name == ".DS_Store":
                continue

            # Get nifti filepath
            nifti_file_path = os.path.join(path, file_name)

            # Preprocess the file
            preprocess_file(nifti_file_path, file_name, output_path)
    else:
        # Get first nifti file in directory
        nifti_file_path = get_first_nifti_file_in_dir(path)

        # Preprocess the file
        preprocess_file(nifti_file_path, output_path, output_path)


if __name__ == "__main__":
    main()

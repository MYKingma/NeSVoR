from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import sys
from skimage.measure import block_reduce

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


def plotVolume(volumeData):
    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volumeData)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    plt.show()


def plotSliceInOrientationFromVolume(volumeData, orientation, sliceNumber):
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


def getFirstNiftiFileInDir(path):
    niftiFiles = [f for f in os.listdir(path) if f.endswith('.nii.gz')]
    return niftiFiles[0]


def subsampleSliceData(sliceData, subsampleRate):
    # Subsample the slice data
    subsampledSliceData = sliceData[::subsampleRate, ::subsampleRate]
    return subsampledSliceData


def getSliceFromDimension(volumeData, dimension, sliceNumber):
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


def subsampleVolumeForSliceDimension(volumeData, dimension, subsampleRate):
    sliceStack = []
    for sliceNumber in range(volumeData.shape[dimension]):
        sliceData = getSliceFromDimension(volumeData, dimension, sliceNumber)
        subsampledSliceData = subsampleSliceData(sliceData, subsampleRate)
        sliceStack.append(subsampledSliceData)
    subsampledVolumeData = np.stack(sliceStack, axis=dimension)
    return subsampledVolumeData


# def subsample_volume(volumeData, subsampleRate):
#     # Subsample the volume data
#     subsampledVolumeData = volumeData[::subsampleRate,
#                                       ::subsampleRate, ::subsampleRate]
#     return subsampledVolumeData

# def subsample_volume(volumeData, subsampleRate):
#     # Get dimensions of the input volume
#     xDim, yDim, zDim = volumeData.shape

#     # Compute the new dimensions of the subsampled volume
#     xNewDim = int(np.ceil(xDim / subsampleRate))
#     yNewDim = int(np.ceil(yDim / subsampleRate))
#     zNewDim = int(np.ceil(zDim / subsampleRate))

#     # Create an empty array to store the subsampled volume data
#     subsampledVolumeData = np.zeros((xNewDim, yNewDim, zNewDim))

#     # Iterate over each subsampled pixel and compute the average value of the corresponding pixels in the input volume
#     for xIndex in range(xNewDim):
#         for yIndex in range(yNewDim):
#             for zIndex in range(zNewDim):
#                 xStart = xIndex * subsampleRate
#                 xEnd = min(xStart + subsampleRate, xDim)
#                 yStart = yIndex * subsampleRate
#                 yEnd = min(yStart + subsampleRate, yDim)
#                 zStart = zIndex * subsampleRate
#                 zEnd = min(zStart + subsampleRate, zDim)
#                 subsampledVolumeData[xIndex, yIndex, zIndex] = np.mean(
#                     volumeData[xStart:xEnd, yStart:yEnd, zStart:zEnd])

#     return subsampledVolumeData

def subsample_volume(volume, subsample_rate):
    # Define the downsampling factor based on the subsample rate
    factor = int(subsample_rate)

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=(factor, factor, factor), func=np.mean)

    # Return the downsampled volume
    return downsampled_volume


def saveStackInDirectory(subsampledVolumeData, orientation, niftiiFilename, outputPath):
    newFilename = niftiiFilename + "_" + str(orientation) + ".nii.gz"
    patientId = niftiiFilename.split("_")[0]
    newNifti = nib.Nifti1Image(subsampledVolumeData, np.eye(4))

    # Create folder in data/subsampled directory with patient id if it doesn't exist
    path = os.path.join(outputPath, patientId)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the nifti file
    nib.save(newNifti, os.path.join(path, newFilename))


def compute_brain_mask(volume, threshold=None, min_size=1000):
    # Compute threshold value using Otsu's algorithm if not provided
    if threshold is None:
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
    mask = remove_small_objects(mask, min_size=min_size)

    mask = np.where(mask == True, 1.0, 0.0)


    return mask


def main():
    # Get the path to the directory containing the nifti files
    path = sys.argv[1]
    outputPath = sys.argv[2]

    # Get the first nifti file in the directory
    niftiFile = getFirstNiftiFileInDir(path)

    # Open the nifti file
    niftiPath = os.path.join(path, niftiFile)
    nifti = nib.load(niftiPath)

    # Get the data from the nifti file
    niftiData = nifti.get_fdata()

    # Get absolute value of data
    niftiData = np.abs(niftiData)

    # Normalize values in volume between 0 and 1
    niftiData = niftiData / np.max(np.abs(niftiData))
    
    # # Plot histogram of values in volume
    # plt.hist(niftiData.flatten(), bins=100)
    # plt.show()

    # # Plot histogram of values in volume
    # plt.hist(niftiData.flatten(), bins=100)
    # plt.show()

    # # Subsample the volume data
    # subsampledVolumeDataX = subsampleVolumeForSliceDimension(niftiData, 0, 2)
    # subsampledVolumeDataY = subsampleVolumeForSliceDimension(niftiData, 1, 2)
    # subsampledVolumeDataZ = subsampleVolumeForSliceDimension(niftiData, 2, 2)

    # # Subsample the volume data
    subsampledVolumeDataX = subsample_volume(niftiData, 2)
    subsampledVolumeDataY = subsample_volume(niftiData, 2)
    subsampledVolumeDataZ = subsample_volume(niftiData, 2)

    # # Save the subsampled volume data
    saveStackInDirectory(subsampledVolumeDataX, 0, niftiFile, outputPath)
    saveStackInDirectory(subsampledVolumeDataY, 1, niftiFile, outputPath)
    saveStackInDirectory(subsampledVolumeDataZ, 2, niftiFile, outputPath)

    # Compute the brain mask of first orientation
    brainMask = compute_brain_mask(subsampledVolumeDataX, 0.1)

    # brainMask = np.where(subsampledVolumeDataX > 0, subsampledVolumeDataX, 0.0)
    # brainMask = np.where(brainMask > 0, 1, 0)

    # # plot mask and volume side by side
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(subsampledVolumeDataX[15, :, :], cmap="gray")
    # axes[1].imshow(brainMask[15, :, :], cmap="gray")
    # plt.show()

    # # Save the brain mask
    # saveStackInDirectory(brainMask, 0, niftiFile + "_mask", outputPath)


if __name__ == "__main__":
    main()

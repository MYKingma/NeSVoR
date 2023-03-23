import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import sys

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

def saveStackInDirectory(subsampledVolumeData, orientation, niftiiFilename):
    newFilename = niftiiFilename + "_" + str(orientation) + ".nii.gz"
    patientId = niftiiFilename.split("_")[0]
    newNifti = nib.Nifti1Image(subsampledVolumeData, np.eye(4))

    # Create folder in data/subsampled directory with patient id if it doesn't exist
    path = os.path.join("data", "subsampled", patientId)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the nifti file
    nib.save(newNifti, os.path.join(path, newFilename))

def main():
    # Get the path to the directory containing the nifti files
    path = sys.argv[1]

    # Get the first nifti file in the directory
    niftiFile = getFirstNiftiFileInDir(path)

    # Open the nifti file
    niftiPath = os.path.join(path, niftiFile)
    nifti = nib.load(niftiPath)

    # Get the data from the nifti file
    niftiData = nifti.get_fdata()

    # Subsample the volume data
    subsampledVolumeDataX = subsampleVolumeForSliceDimension(niftiData, 0, 3)
    subsampledVolumeDataY = subsampleVolumeForSliceDimension(niftiData, 1, 3)
    subsampledVolumeDataZ = subsampleVolumeForSliceDimension(niftiData, 2, 3)

    # Save the subsampled volume data
    saveStackInDirectory(subsampledVolumeDataX, 0, niftiFile)
    saveStackInDirectory(subsampledVolumeDataY, 1, niftiFile)
    saveStackInDirectory(subsampledVolumeDataZ, 2, niftiFile)

if __name__ == "__main__":
    main()
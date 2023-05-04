import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import sys

def openAndPlotNifti(niftiFile):

    # Open the nifti file
    nifti = nib.load(niftiFile)

    # Get the data from the nifti file
    niftiData = nifti.get_fdata()
    print(nifti.file_map["image"].filename)

    # plt.imshow(niftiData[100, :, :], cmap='gray')


    fig, ax = plt.subplots(1, 1)
    plt.gray()
    try:
        tracker = IndexTracker(ax, niftiData)
    except ValueError:
        print("ValueError")
        return


    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

def getAllNiftiFilesInDir(path):

    # Get all nifti files in the path
    niftiFiles = [f for f in os.listdir(path) if f.endswith('.nii.gz')]
    return niftiFiles

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




if __name__ == '__main__':
    # Get data dir filepath from sys.argv
    dataDir = sys.argv[1]

    # Get all nifti files in the path
    niftiFiles = getAllNiftiFilesInDir(dataDir)
    
    # Open and plot all nifti files
    for niftiFile in niftiFiles:
        openAndPlotNifti(dataDir + "/" + niftiFile)

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse


def open_nifti_file(file_path):
    """Open and load a nifti file."""
    img = nib.load(file_path)
    data = img.get_fdata()

    # Get spacing
    spacing = img.header['pixdim'][1:4]

    print("BEFORE:", data.shape)

    if "sag" in file_path:
        data = np.transpose(data,  (0, 1, 2))
    else:
        data = np.flip(data, axis=(0, 1))
        print(data.shape)
        # data = np.transpose(data, (0, 1, 2))

    print(data.shape)


    return data, spacing

def get_slice_data_for_orientation(volume, slice_idx, orientation):
    """Get a slice from a 3D volume."""
    if orientation == 0:
        return volume[:, :, 40]
    elif orientation == 1:
        return volume[:, 40, :]
    elif orientation == 2:
        return volume[40, :, :]
    else:
        raise ValueError('Orientation must be "axial", "sagittal", or "coronal".')
    
def plot_orientations_side_by_side(volume, spacing):
    """Plot the three orthogonal views of a 3D volume."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Get the slice data for each orientation
    axial_slice = get_slice_data_for_orientation(volume, 30, 0)
    sagittal_slice = get_slice_data_for_orientation(volume, 30, 1)
    coronal_slice = get_slice_data_for_orientation(volume, 30, 2)

    # Set the extent parameter to stretch out the images
    axial_extent = [0, axial_slice.shape[1]*spacing[1], 0, axial_slice.shape[0]*spacing[0]]
    sagittal_extent = [0, sagittal_slice.shape[1]*spacing[2], 0, sagittal_slice.shape[0]*spacing[0]]
    coronal_extent = [0, coronal_slice.shape[1]*spacing[2], 0, coronal_slice.shape[0]*spacing[1]]

    # Plot the axial view
    ax[0].imshow(axial_slice, cmap='gray', extent=axial_extent)

    # Plot the sagittal view
    ax[1].imshow(sagittal_slice, cmap='gray', extent=sagittal_extent)

    # Plot the coronal view
    ax[2].imshow(coronal_slice, cmap='gray', extent=coronal_extent)

    # Remove the ticks from all the axes
    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])

    plt.show()

def transform_spacing_to_shape(spacing, shape):
    """Transform spacing to shape."""

    # Get slice dimension, dimension with smallest size
    slice_dim = np.argmin(shape)

    # Get slice resolution, highest resolution value
    slice_res = np.argmax(spacing)

    # Get isotropic resolution
    iso_res = np.min(spacing)

    # Transform spacing
    spacing = np.full((3,), iso_res)
    spacing[slice_dim] = slice_res

    return spacing


def main(args):
    """Main function."""
    # Load the data
    volume, spacing = open_nifti_file(args.data)

    if args.resolution:
        spacing = args.resolution
    else:
        # Transform spacing to shape
        spacing = transform_spacing_to_shape(spacing, volume.shape)

    # Plot the data
    plot_orientations_side_by_side(volume, spacing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to the data directory.')
    parser.add_argument("-r", "--resolution", type=float, nargs=3, required=False,
                        help="Resolution of the data.")
    
    args = parser.parse_args()
    main(args)
    
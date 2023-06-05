import nibabel as nib
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare two nifti files')
parser.add_argument('-file1', help='Path to first nifti file')
parser.add_argument('-file2', help='Path to second nifti file')
args = parser.parse_args()

# Load the two nifti files
nii1 = nib.load(args.file1)
nii2 = nib.load(args.file2)

# Compare the headers
header1 = nii1.header
header2 = nii2.header
for key in header1:
    if key not in header2:
        print(f'Header key {key} not found in file2')
    else:
        print(f'Header key {key} has values: {header1[key]} vs {header2[key]}')

# # Compare the data
# data1 = nii1.get_fdata()
# data2 = nii2.get_fdata()
# if data1.shape != data2.shape:
#     print(f'Data shape is different: {data1.shape} vs {data2.shape}')
# elif not (data1 == data2).all():
#     print('Data is different')
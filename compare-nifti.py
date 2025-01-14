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


# Compare transformation matrices
matrix1 = nii1.affine
matrix2 = nii2.affine
if matrix1.shape != matrix2.shape:
    print(f'Matrix shape is different: {matrix1.shape} vs {matrix2.shape}')
elif not (matrix1 == matrix2).all():
    print('Matrix is different')

# Print transformation matrix
print(matrix1)
print(matrix2)

# Compare the shape of the data
data1 = nii1.get_fdata()
data2 = nii2.get_fdata()

# Print data shape
print(data1.shape)
print(data2.shape)
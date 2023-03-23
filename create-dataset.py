import os

def getAllNiftiFiles(path):
    dataset = []

    # Loop through all directories in the path
    for root, dirs, files in os.walk(path):

        # Get T2w_acpc_dc_restore.nii.gz file from the T1w folder
        if 'T1w' in root:
            for file in files:
                if file == 'T2w_acpc_dc_restore.nii.gz':
                    subjectId = root.split('/')[-2]
                    newFileName = subjectId + '_' + file

                    # Rename the file, prefix with the subject id
                    os.rename(os.path.join(root, file), os.path.join(root, newFileName))
                    
                    # Add the file to the dataset
                    dataset.append(os.path.join(root, newFileName))
    return dataset



if __name__ == '__main__':
    # Get all nifti files in the path
    dataset = getAllNiftiFiles('./data/unzipped')

    # Move files to a new directory 'processed' in the data directory
    for file in dataset:
        os.rename(file, './data/processed/' + os.path.basename(file))

import os
import zipfile
import sys

def unzipFilesInDirectory(directory, destination):
    
    # Loop through all files in the directory
    for file in os.listdir(directory):

        # Check if the file is a zip file
        if file.endswith('.zip'):

            # Get the full path to the file
            filePath = os.path.join(directory, file)

            # Unzip the file
            with zipfile.ZipFile(filePath, 'r') as zip_ref:
                zip_ref.extractall(destination)


if __name__ == '__main__':
    # Get directory and destination path from system arguments
    directory = sys.argv[1]
    destination = sys.argv[2]

    # Unzip all files in the directory
    unzipFilesInDirectory(directory, destination)

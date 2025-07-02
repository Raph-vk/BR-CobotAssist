import os
import argparse
import time

def rename_files_in_directory(directory):
    """
    Renames all .hdf5 files in a specified directory.
    Args:
    directory (str): The directory path where the files are located.
    """
    # List all .hdf5 files in the directory
    files_unsorted = [f for f in os.listdir(directory) if f.endswith('.hdf5')]
    files = sorted(files_unsorted,key=lambda fn: int(fn.split('_')[1].split('.')[0]))


    # Rename files to a temporary name to avoid conflicts
    temp_files = []
    for i, filename in enumerate(files):
        temp_name = f"temp2_{i}.hdf5"
        os.rename(os.path.join(directory, filename), os.path.join(directory, temp_name))
        temp_files.append(temp_name)
        time.sleep(0.001)  # Sleep to ensure file system can register changes

    # Wait briefly before final renaming
    time.sleep(1)

    # Rename from temporary names to final desired names
    for i, temp_name in enumerate(temp_files):
        new_name = f"episode_{i}.hdf5"
        os.rename(os.path.join(directory, temp_name), os.path.join(directory, new_name))
        time.sleep(0.001)  # Brief sleep to manage the load on file system operations

def rename_files_based_on_task(parent_directory):
    """
    Finds directories starting with a given task name and renames all .hdf5 files within those directories.
    Args:
    task_name (str): The base name of the task associated with the directories.
    parent_directory (str): The parent directory containing the task-specific subdirectories.
    """
    # # Find directories that start with the task name
    # directories = [d for d in os.listdir(parent_directory)]
    # print(f"Found directories: {directories}")
    # # Process each directory
    # for dir_name in directories:
    #     directory_path = os.path.join(parent_directory, dir_name)
    rename_files_in_directory(parent_directory)
    print(f"Renaming complete in {parent_directory}")


def main():
    path = '/home/teun/tos_app_data/converted_20250701_105519'
    rename_files_based_on_task(path)

if __name__ == "__main__":
    main()

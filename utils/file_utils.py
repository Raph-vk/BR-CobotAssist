#!/usr/bin/env python3
"""
File management utility functions.
These functions provide generic helpers for file and directory operations
that can be reused across different components.
"""

import os
import time


def _move_files_to_mistakes(dataset_name, threshold_secs, parent_dir, logger):
    """
    Directory-scanning utility that decides which files are "recent" and moves them.
    Can accept dataset_name, threshold_secs, parent_dir and return the moved-count.
    
    Args:
        dataset_name: Name of the dataset to process
        threshold_secs: Threshold in seconds for determining "recent" files
        parent_dir: Parent directory containing the dataset directories
        logger: Logger instance for logging events
        
    Returns:
        int: Number of files moved to mistakes directories
        
    Raises:
        Exception: If directory processing fails
    """
    current_time = time.time()
    moved_files_count = 0
    
    try:
        # Generate a list of directory paths that start with the dataset name
        if not os.path.exists(parent_dir):
            logger.warning(f"Parent directory does not exist: {parent_dir}")
            return 0
            
        directories = [
            d for d in os.listdir(parent_dir) 
            if d.startswith(dataset_name) and os.path.isdir(os.path.join(parent_dir, d))
        ]
        
        # Process each relevant directory
        for dir_name in directories:
            if dir_name.endswith("_mistakes"):
                continue  # Skip existing mistake directories
                
            directory_path = os.path.join(parent_dir, dir_name)
            moved_count = _move_directory_files_to_mistakes(
                directory_path, threshold_secs, current_time, logger
            )
            moved_files_count += moved_count
            
    except Exception as e:
        logger.error(f"Error in _move_files_to_mistakes: {e}")
        raise
        
    return moved_files_count


def _move_directory_files_to_mistakes(directory_path, threshold_secs, current_time, logger):
    """
    Move files within a single directory that are within the threshold_secs to a '_mistakes' subdirectory.
    Inner helper for the previous one – same reasoning.
    
    Args:
        directory_path: Path to the directory to process
        threshold_secs: Threshold in seconds for determining "recent" files
        current_time: Current timestamp for comparison
        logger: Logger instance for logging events
        
    Returns:
        int: Number of files moved to mistakes directory
        
    Raises:
        Exception: If file moving operations fail
    """
    moved_count = 0
    
    try:
        mistakes_directory = directory_path + '_mistakes'
        
        # Create the '_mistakes' directory if it does not exist
        if not os.path.exists(mistakes_directory):
            os.makedirs(mistakes_directory)
            logger.info(f"Created mistakes directory: {mistakes_directory}")
        
        # Process each file in the directory
        if not os.path.exists(directory_path):
            return 0
            
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                creation_time = os.path.getctime(file_path)
                if (current_time - creation_time) <= threshold_secs:
                    new_path = os.path.join(mistakes_directory, filename)
                    logger.info(f"Moving file to mistakes directory: {filename}")
                    os.rename(file_path, new_path)
                    moved_count += 1
                    
    except Exception as e:
        logger.error(f"Error moving files in directory {directory_path}: {e}")
        raise
        
    return moved_count

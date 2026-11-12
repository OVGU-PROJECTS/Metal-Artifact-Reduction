import os
import shutil
import random

def split_dataset(main_dir, train_dir, val_dir, val_size=1000, seed=None):
    # Set random seed for reproducibility (optional)
    if seed is not None:
        random.seed(seed)
    
    # Ensure the main directory exists
    if not os.path.exists(main_dir):
        print(f"Main directory {main_dir} does not exist!")
        return
    
    # Create training and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of all folders in the main directory
    all_folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    
    # Check if we have enough folders
    if len(all_folders) < val_size:
        print(f"Not enough folders! Found {len(all_folders)}, need {val_size} for validation.")
        return
    
    # Shuffle the folder list to ensure randomness
    random.shuffle(all_folders)
    
    # Select first 1000 folders for validation
    val_folders = all_folders[:val_size]
    train_folders = all_folders[val_size:]
    
    # Optional: Check distribution of selected folders (for debugging)
    val_indices = [int(f.split('_')[-1]) for f in val_folders if f.split('_')[-1].isdigit()]
    if val_indices:
        print(f"Validation folder index range: {min(val_indices)} to {max(val_indices)}")
    
    # Move validation folders
    for folder in val_folders:
        src_path = os.path.join(main_dir, folder)
        dest_path = os.path.join(val_dir, folder)
        shutil.move(src_path, dest_path)
        print(f"Moved {folder} to validation directory")
    
    # Move training folders
    for folder in train_folders:
        src_path = os.path.join(main_dir, folder)
        dest_path = os.path.join(train_dir, folder)
        shutil.move(src_path, dest_path)
        print(f"Moved {folder} to training directory")
    
    print(f"Dataset split complete: {len(train_folders)} training folders, {len(val_folders)} validation folders")

if __name__ == "__main__":
    # Define paths
    main_dataset_dir = "/mnt/public/bahaa/body"  # Path to the main dataset with 14000 folders
    training_dir = "/mnt/public/bahaa/dataset_training"  # Path for training set
    validation_dir = "/mnt/public/bahaa/dataset_validation"  # Path for validation set
    
    # Split the dataset with a random seed for reproducibility
    split_dataset(main_dataset_dir, training_dir, validation_dir, val_size=1000, seed=42)

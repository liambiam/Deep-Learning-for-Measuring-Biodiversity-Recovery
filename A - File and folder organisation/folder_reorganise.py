import os
import shutil

# Set base path
data_path = 'E:/Spectrograms'

# List of PAM directories
pam_dirs = ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']

def move_subfolders(data_path, pam_dirs):
    for pam in pam_dirs:
        # Path to current PAM directory
        pam_folder = os.path.join(data_path, pam)
        
        # Check if PAM directory exists
        if not os.path.isdir(pam_folder):
            print(f"Directory {pam_folder} doesn't exist.")
            continue

        # Go through directory structure
        for root, dirs, files in os.walk(pam_folder, topdown=False):
            for folder in dirs:
                # Check if folder name starts with 'S4'
                if folder.startswith('S4'):
                    src_folder = os.path.join(root, folder)
                    dest_folder = os.path.join(pam_folder, folder)
                    
                    # Move folder
                    if not os.path.exists(dest_folder):
                        shutil.move(src_folder, dest_folder)
                        print(f"Moved: {src_folder} -> {dest_folder}")
                    else:
                        print(f"Directory already exists: {dest_folder}")

if __name__ == "__main__":
    move_subfolders(data_path, pam_dirs)

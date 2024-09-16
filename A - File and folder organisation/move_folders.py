import os
import shutil

# Main folder path
root_dir = 'E:/Spectrograms64second'

# List of PAM folders
pam_folders = ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']

def move_folders(root_dir, pam_folders):
    for pam_folder in pam_folders:
        # Path to  current PAM folder
        full_pam_path = os.path.join(root_dir, pam_folder)
        
        # Check if  PAM folder exists
        if not os.path.isdir(full_pam_path):
            print(f"Folder {full_pam_path} not found.")
            continue

        # Walk through  directory 
        for base, subfolders, files in os.walk(full_pam_path, topdown=False):
            for folder in subfolders:
                # Move folders starting with 'S4'
                if folder.startswith('S4'):
                    src_path = os.path.join(base, folder)
                    dest_path = os.path.join(full_pam_path, folder)
                    
                    # Move folder if it doesn't already exist at  destination
                    if not os.path.exists(dest_path):
                        shutil.move(src_path, dest_path)
                        print(f"Moved: {src_path} to {dest_path}")
                    else:
                        print(f"Folder already exists: {dest_path}")

if __name__ == "__main__":
    move_folders(root_dir, pam_folders)

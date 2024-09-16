import os
import shutil

# Main folder with spectrograms
main_folder = 'E:/Spectrograms/PAM_2'

# Function to sort folders by year and month
def sort_folders_by_year_month(main_folder):
    folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    print(f"Found {len(folders)} folders in {main_folder}")
    
    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        
        # Get year and month from folder name
        try:
            date_part = folder.split('_')[1]
            year_month = date_part[:6]  # e.g., '202109' from '20210917'
        except IndexError:
            print(f"Skipping folder with name format issue: {folder}")
            continue
        
        # New folder name based on year and month
        new_folder_name = f"{year_month[:4]}-{year_month[4:]}"
        new_folder_path = os.path.join(main_folder, new_folder_name)
        
        # Make new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
        
        # Move folder to new location
        shutil.move(folder_path, new_folder_path)
        print(f"Moved {folder} to {new_folder_name}")

# Sort folders by year and month
sort_folders_by_year_month(main_folder)
print("Folders sorted.")

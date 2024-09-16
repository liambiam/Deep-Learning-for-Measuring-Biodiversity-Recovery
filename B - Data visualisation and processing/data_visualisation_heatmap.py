import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from calendar import month_name
from matplotlib import rcParams

# Set font to Poppins
rcParams['font.family'] = 'Poppins'

# Main directory
main_dir = 'E:/Audio'

# Dictionary to keep file counts
count_dict = {}

# Go through main directory to find location folders (e.g., PAM_1, PAM_2, ...)
for loc in os.listdir(main_dir):
    loc_path = os.path.join(main_dir, loc)
    if os.path.isdir(loc_path):
        # Go through year/location folders (e.g., 2020_PAM_1_POS_2_PAGI)
        for year_loc in os.listdir(loc_path):
            year_loc_path = os.path.join(loc_path, year_loc)
            if os.path.isdir(year_loc_path):
                # Get year from folder name
                year = year_loc.split('_')[0]
                
                # Skip year 2024
                if year == '2024':
                    continue
                
                loc_year = f"{loc}_{year}"

                if loc_year not in count_dict:
                    count_dict[loc_year] = {f'{m:02d}': 0 for m in range(1, 13)}  # Start counts for all months

                # Go through subfolders to find month folders
                for month_fold in os.listdir(year_loc_path):
                    month_fold_path = os.path.join(year_loc_path, month_fold)
                    if os.path.isdir(month_fold_path):
                        try:
                            # Get date part from folder name
                            date_part = month_fold.split('_')[-1]  # Get date part ('2020-03-26')
                            file_month = date_part.split('-')[1]  # Get month part (eg.'03'
                            
                            # Count .wav files in month folder
                            wav_count = len([f for f in os.listdir(month_fold_path) if f.endswith('.wav')])
                            if file_month in count_dict[loc_year]:
                                count_dict[loc_year][file_month] += wav_count
                        except IndexError:
                            pass

# Turn dictionary into df
data_frame = pd.DataFrame(count_dict).T

# Make sure all months are in the df, fill missing with zero
months = [f'{m:02d}' for m in range(1, 13)]
data_frame = data_frame.reindex(columns=months, fill_value=0)

# Rename columns to months
month_names = [month_name[int(m)] for m in months]
data_frame.columns = month_names

# Transpose DataFrame for heatmap
data_frame = data_frame.T

# Plot heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(data_frame, cmap="YlGnBu", annot=True, fmt='d', linewidths=.5, cbar_kws={'label': 'File Count'})
plt.title('File Count Heatmap per Month for Each Location/Year', fontweight='bold')
plt.xlabel('Location/Year')
plt.ylabel('Month')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

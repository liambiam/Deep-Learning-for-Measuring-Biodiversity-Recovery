import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths to the CSV files
results_dir = 'E:/Results/acoustic_indices'
pam_csv_files = {
    'PAM_1': os.path.join(results_dir, 'acoustic_indices_PAM_1.csv'),
    'PAM_2': os.path.join(results_dir, 'acoustic_indices_PAM_2.csv'),
    'PAM_3': os.path.join(results_dir, 'acoustic_indices_PAM_3.csv'),
    'PAM_4': os.path.join(results_dir, 'acoustic_indices_PAM_4.csv')
}

# Colors and styles for each PAM
pam_colors = {
    'PAM_1': 'blue',
    'PAM_2': 'yellow',
    'PAM_3': 'green',
    'PAM_4': 'red'
}
pam_styles = {
    'PAM_1': 'o',
    'PAM_2': 'o',
    'PAM_3': 'o',
    'PAM_4': 'o'
}

# Load and process data
monthly_averages = {}
for pam, csv_file in pam_csv_files.items():
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_avg = df.groupby('YearMonth').mean().reset_index()
    monthly_averages[pam] = monthly_avg

# Plotting all indices in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
index_names = ['ACI', 'ADI', 'NDSI', 'Bioacoustic Index']
axs = axs.ravel()

for ax, index_name in zip(axs, index_names):
    for pam, monthly_avg in monthly_averages.items():
        ax.scatter(monthly_avg['YearMonth'].astype(str), monthly_avg[index_name], 
                   color=pam_colors[pam], edgecolor='black' if pam == 'PAM_2' else None,
                   marker=pam_styles[pam], s=50)  # Reduced point size to 50
    
    ax.set_title(index_name)
    ax.set_xlabel('Month')
    ax.set_ylabel(index_name)
    
    # Set x-ticks to show fewer dates
    ax.set_xticks(ax.get_xticks()[::3])  # Show every 3rd tick
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

plt.tight_layout()
plt.show()

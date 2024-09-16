import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

# Define paths to the CSV files
results_dir = 'E:/Results/acoustic_indices'
pam_csv_files = {
    'PAM_2': os.path.join(results_dir, 'acoustic_indices_PAM_2.csv'),
    'PAM_3': os.path.join(results_dir, 'acoustic_indices_PAM_3.csv')
}

# Colors for each PAM
pam_colors = {
    'PAM_2': 'yellow',
    'PAM_3': 'green'
}

# Load and process data
monthly_averages = {}
index_names = ['ACI', 'ADI', 'NDSI', 'Bioacoustic Index']

for pam, csv_file in pam_csv_files.items():
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_avg = df.groupby('YearMonth')[index_names].mean().reset_index()
    monthly_averages[pam] = monthly_avg

# Align data
aligned_data = pd.merge(monthly_averages['PAM_2'], monthly_averages['PAM_3'], on='YearMonth', suffixes=('_PAM2', '_PAM3'))

# Plotting function
def plot_comparison(ax, index_name):
    x = aligned_data['YearMonth'].astype(str)
    y2 = aligned_data[f'{index_name}_PAM2']
    y3 = aligned_data[f'{index_name}_PAM3']
   
    ax.scatter(x, y2, color=pam_colors['PAM_2'], alpha=0.7, edgecolors='black')
    ax.scatter(x, y3, color=pam_colors['PAM_3'], alpha=0.7, edgecolors='none')
   
    # Calculate regression lines and metrics
    slope2, intercept2, r_value2, _, _ = stats.linregress(range(len(x)), y2)
    slope3, intercept3, r_value3, _, _ = stats.linregress(range(len(x)), y3)
   
    line2 = slope2 * np.arange(len(x)) + intercept2
    line3 = slope3 * np.arange(len(x)) + intercept3
   
    ax.plot(x, line2, color='black', linestyle='--')
    ax.plot(x, line3, color='green', linestyle='--')
   
    # Add R² and correlation coefficient to the plot
    r_squared2 = r_value2**2
    r_squared3 = r_value3**2
    ax.set_xlabel('Date')
    ax.set_ylabel(index_name)
    ax.set_title(f'{index_name}: PAM 2 vs PAM 3')
    ax.grid(True)
   
    # Show dates every 6 months
    date_ticks = x[::6]
    ax.set_xticks(range(0, len(x), 6))
    ax.set_xticklabels(date_ticks, rotation=45, ha='right')

    # Print R² and correlation coefficient
    print(f"{index_name}:")
    print(f"PAM 2: R² = {r_squared2:.3f}, r = {r_value2:.3f}")
    print(f"PAM 3: R² = {r_squared3:.3f}, r = {r_value3:.3f}")
    print()

# Plotting all indices in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(16, 14))
axs = axs.ravel()

for ax, index_name in zip(axs, index_names):
    plot_comparison(ax, index_name)

plt.tight_layout()
plt.show()
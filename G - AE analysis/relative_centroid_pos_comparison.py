import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# File paths for CSV files with averaged centroid distances
centroid_files = {
    'PAM1 vs PAM3': r'e:\Results\cluster distances\centroid_positions1v3_monthly_avg_centroid_distance.csv',
    'PAM2 vs PAM3': r'e:\Results\cluster distances\centroid_positions2v3_monthly_avg_centroid_distance.csv',
    'PAM4 vs PAM3': r'e:\Results\cluster distances\centroid_positions4v3_monthly_avg_centroid_distance.csv'
}

# Define colors for each plot
plot_colors = {
    'PAM1 vs PAM3': 'blue',
    'PAM2 vs PAM3': 'gold',
    'PAM4 vs PAM3': 'red'
}

# Create figure and set up 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (label, file_path) in zip(axes, centroid_files.items()):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Plot centroid distances
    color = plot_colors[label]
    if color == 'gold':
        ax.plot(data['date'], data['centroid_distance'], marker='o', linestyle='None', color=color, markeredgecolor='black', markeredgewidth=1, label='Average Distance')
    else:
        ax.plot(data['date'], data['centroid_distance'], marker='o', linestyle='None', color=color, label='Average Distance')
    
    # Compute trend line
    numeric_dates = (data['date'] - data['date'].min()).dt.days  # Dates to numeric values (days from start)
    slope, intercept, r_val, p_val, std_err = linregress(numeric_dates, data['centroid_distance'])
    trend_line = slope * numeric_dates + intercept
    
    # Plot trend line
    ax.plot(data['date'], trend_line, color=color, linestyle='-', label=f'Trend Line')
    
    # Calculate and show R² and correlation coefficient
    r_squared = r_val ** 2
    corr_coef = np.corrcoef(numeric_dates, data['centroid_distance'])[0, 1]
    ax.text(0.05, 0.95, f'$R^2$={r_squared:.2f}\n$\\rho$={corr_coef:.2f}', transform=ax.transAxes,
            verticalalignment='top', color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Set subplot titles and labels
    ax.set_title(f'Avg Centroid Distance by Month: {label}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg Centroid Distance')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# Adjust layout and save figure to file
plt.tight_layout()
output_path = r'e:\Results\cluster distances\combined_monthly_avg_centroid_distances.png'
plt.savefig(output_path)
plt.show()

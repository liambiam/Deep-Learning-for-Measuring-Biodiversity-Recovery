import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# File paths for the monthly average CSVs
monthly_avg_paths = {
    'PAM1 vs PAM3': r"E:\Results\AE clusters\monthly_averages\PAM1_vs_PAM3_monthly_avg.csv",
    'PAM2 vs PAM3': r"E:\Results\AE clusters\monthly_averages\PAM2_vs_PAM3_monthly_avg.csv",
    'PAM4 vs PAM3': r"E:\Results\AE clusters\monthly_averages\PAM4_vs_PAM3_monthly_avg.csv"
}

# Define colors for the plots
colors = {
    'PAM1 vs PAM3': 'blue',
    'PAM2 vs PAM3': 'gold',
    'PAM4 vs PAM3': 'red',
    'PAM3': 'green'  # Green for PAM3
}

# Create subplots (3 plots side by side)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (title, path) in zip(axes, monthly_avg_paths.items()):
    # Load the monthly average data
    monthly_avg = pd.read_csv(path)
    
    # Convert 'date' column to datetime
    monthly_avg['date'] = pd.to_datetime(monthly_avg['date'])
    
    # Plot the Cluster Distances
    color = colors[title]
    if color == 'gold':
        # Plot gold dots with black circles
        ax.scatter(monthly_avg['date'], monthly_avg['cluster_1_mean_distance'], 
                   facecolors=color, edgecolors='black', s=50, linewidths=1, zorder=3)
    else:
        ax.plot(monthly_avg['date'], monthly_avg['cluster_1_mean_distance'], 
                marker='o', linestyle='None', color=color)
    
    ax.plot(monthly_avg['date'], monthly_avg['cluster_2_mean_distance'], 
            marker='o', linestyle='None', color=colors['PAM3'])
    
    # Perform Linear Regression for Cluster 1
    valid_mask1 = ~np.isnan(monthly_avg['cluster_1_mean_distance'])
    X1 = np.arange(len(monthly_avg))[valid_mask1].reshape(-1, 1)
    y1 = monthly_avg['cluster_1_mean_distance'].dropna().values
    model1 = LinearRegression().fit(X1, y1)
    trend1 = model1.predict(np.arange(len(monthly_avg)).reshape(-1, 1))
    
    # Plot regression line for Cluster 1
    ax.plot(monthly_avg['date'], trend1, color=color, linestyle='-', linewidth=1.5)
    
    # Calculate R2 and CC for Cluster 1
    r2_1 = r2_score(y1, model1.predict(X1))
    cc_1 = np.corrcoef(X1[:, 0], y1)[0, 1]
    
    # Perform Linear Regression for PAM3 (Cluster 2)
    valid_mask2 = ~np.isnan(monthly_avg['cluster_2_mean_distance'])
    X2 = np.arange(len(monthly_avg))[valid_mask2].reshape(-1, 1)
    y2 = monthly_avg['cluster_2_mean_distance'].dropna().values
    model2 = LinearRegression().fit(X2, y2)
    trend2 = model2.predict(np.arange(len(monthly_avg)).reshape(-1, 1))
    
    # Plot regression line for PAM3 (Cluster 2)
    ax.plot(monthly_avg['date'], trend2, color=colors['PAM3'], linestyle='-', linewidth=1.5)
    
    # Calculate R2 and CC for PAM3 (Cluster 2)
    r2_2 = r2_score(y2, model2.predict(X2))
    cc_2 = np.corrcoef(X2[:, 0], y2)[0, 1]
    
    # Annotate R² and CC values for Cluster 1
    ax.text(0.05, 0.9, f'Cluster 1 R²: {r2_1:.2f}', transform=ax.transAxes, color=color)
    ax.text(0.05, 0.85, f'Cluster 1 CC: {cc_1:.2f}', transform=ax.transAxes, color=color)
    
    # Annotate R² and CC values for PAM3 (Cluster 2)
    ax.text(0.05, 0.8, f'PAM3 R²: {r2_2:.2f}', transform=ax.transAxes, color=colors['PAM3'])
    ax.text(0.05, 0.75, f'PAM3 CC: {cc_2:.2f}', transform=ax.transAxes, color=colors['PAM3'])
    
    # Set titles and labels
    ax.set_title(f'Mean Cluster Distances Over Time: {title}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Distance from Centroid')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

# Adjust layout and save the figure
plt.tight_layout()
save_path = r'e:\Results\AE clusters\combined_cluster_mean_distances_monthly_avg.png'
plt.savefig(save_path)
plt.show()
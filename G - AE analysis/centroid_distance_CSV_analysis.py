import pandas as pd
import matplotlib.pyplot as plt

# File paths for the CSVs
centroid_csv_path = r'E:\Results\AE6a\centroid_positions.csv'
distance_csv_path = r'E:\Results\AE6a\cluster_distances.csv'

# Load the data from the CSV files
centroid_df = pd.read_csv(centroid_csv_path)
distance_df = pd.read_csv(distance_csv_path)

# Convert 'Month' column to datetime for better plotting
centroid_df['Month'] = pd.to_datetime(centroid_df['Month'])
distance_df['Month'] = pd.to_datetime(distance_df['Month'])

# Sort data by date just in case
centroid_df = centroid_df.sort_values(by='Month')
distance_df = distance_df.sort_values(by='Month')

# Plot Centroid Distances over Time
plt.figure(figsize=(10, 6))
plt.plot(centroid_df['Month'], centroid_df['Centroid_Distance'], marker='o', linestyle='-', color='b')
plt.title('Distance Between Centroids Over Time')
plt.xlabel('Date')
plt.ylabel('Centroid Distance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'E:\Results\AE6a\centroid_distance_over_time.png')
plt.show()

# Plot Mean Cluster Distances over Time
plt.figure(figsize=(10, 6))
plt.plot(distance_df['Month'], distance_df['Cluster_1_Mean_Distance'], marker='o', linestyle='-', color='r', label='Cluster 1 Mean Distance')
plt.plot(distance_df['Month'], distance_df['Cluster_2_Mean_Distance'], marker='o', linestyle='-', color='g', label='Cluster 2 Mean Distance')
plt.title('Mean Cluster Distances Over Time')
plt.xlabel('Date')
plt.ylabel('Mean Distance from Centroid')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'E:\Results\AE6a\cluster_mean_distances_over_time.png')
plt.show()

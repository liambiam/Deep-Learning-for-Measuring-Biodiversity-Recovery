import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

# Path to the CSV file
csv_file_path = 'E:/Code/Results/predictions.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# # Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# # Calculate the mean confidence score for 'Confidence_PAM_1' for each date
mean_confidence_per_date = df.groupby('Date').mean().reset_index()

# # Plot the average confidence scores for Class 0 (PAM_1) over time
# plt.figure(figsize=(12, 6))
# plt.plot(mean_confidence_per_date['Date'], mean_confidence_per_date['Confidence_3'], marker='o', linestyle='-', label='Average Confidence_0')
# plt.title('Average Confidence Scores for Class 0 (PAM_1) Over Time')
# plt.xlabel('Date')
# plt.ylabel('Average Confidence Score')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Path to save the plot
# avg_confidence_plot_path = 'E:/Code/Results/avg_confidence_scores_class_3_over_time.png'
# plt.savefig(avg_confidence_plot_path)
# plt.close()

# print(f"Saved plot: {avg_confidence_plot_path}")

import seaborn as sns

# Create a pivot table for the heatmap
pivot_table = df.pivot_table(index='Date', values=['Confidence_0', 'Confidence_1', 'Confidence_2', 'Confidence_3'])

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt=".2f")
plt.title('Heatmap of Confidence Scores Over Time')
plt.xlabel('Class')
plt.ylabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()

# Path to save the plot
heatmap_plot_path = 'E:/Code/Results/heatmap_confidence_scores.png'
plt.savefig(heatmap_plot_path)
plt.close()

print(f"Saved plot: {heatmap_plot_path}")

# Prepare the data for the trajectory plot
trajectory_data = mean_confidence_per_date[['Confidence_0', 'Confidence_1']].values

# Plot the trajectory in a 3D space
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_data[:, 0], trajectory_data[:, 1], marker='o')

ax.set_title('Trajectory of Confidence Scores Over Time')
ax.set_xlabel('Confidence_PAM_1')

# Path to save the plot
trajectory_plot_path = 'E:/Code/Results/trajectory_confidence_scores.png'
plt.savefig(trajectory_plot_path)
plt.close()

print(f"Saved plot: {trajectory_plot_path}")
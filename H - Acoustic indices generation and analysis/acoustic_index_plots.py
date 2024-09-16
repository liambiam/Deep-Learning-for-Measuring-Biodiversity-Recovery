import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file
csv_path = 'E:/Results/acoustic_indices/acoustic_indices_PAM_1.csv'

# Load the data
df = pd.read_csv(csv_path)

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the monthly average for each index
df['Month'] = df['Date'].dt.to_period('M')
indices = ['ACI', 'ADI', 'NDSI', 'Bioacoustic Index', 'Hybrid']
df_monthly_avg = df.groupby('Month')[indices].mean().reset_index()

# Convert 'Month' back to datetime for plotting
df_monthly_avg['Month'] = df_monthly_avg['Month'].dt.to_timestamp()

# Plotting
for index in indices:
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly_avg['Month'], df_monthly_avg[index], marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel(f'Average {index} Score')
    plt.title(f'Monthly Average {index} Score Over Time')
    plt.grid(True)
    
    # Save each plot
    plot_path = os.path.join('E:/Results/acoustic_indices', f'PAM_1_average_{index.lower()}_over_time.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved plot: {plot_path}")

print("All plots have been generated and saved.")

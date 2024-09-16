import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Path to the CSV file
csv_path = 'E:/Results/CNN2secondConf/predictions_PAM_3.csv'

# Load the data
df = pd.read_csv(csv_path)

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the monthly average confidence score for each class
df['Month'] = df['Date'].dt.to_period('M')
confidence_columns = [col for col in df.columns if col.startswith('Conf')]
df_monthly_avg_conf = df.groupby('Month')[confidence_columns].mean().reset_index()

# Convert 'Month' back to datetime for plotting
df_monthly_avg_conf['Month'] = df_monthly_avg_conf['Month'].dt.to_timestamp()

# Plotting
plt.figure(figsize=(12, 6))
for conf in confidence_columns:
    # Scatter plot of the monthly average confidence scores
    plt.scatter(df_monthly_avg_conf['Month'], df_monthly_avg_conf[conf], label=conf)
    
    # Fit a polynomial trend line (degree 3) to the data
    p = Polynomial.fit(df_monthly_avg_conf.index, df_monthly_avg_conf[conf], deg=3)
    trend_line = p(df_monthly_avg_conf.index)
    
    # Plot the trend line
    plt.plot(df_monthly_avg_conf['Month'], trend_line, linestyle='--')

plt.xlabel('Month')
plt.ylabel('Average Confidence Score')
plt.title('Monthly Average Confidence Score for Each Class')
plt.legend(title='Class', loc='upper left')
plt.tight_layout()

# Save plot
results_dir = 'E:/Results/CNN2secondConf'
plot_path = os.path.join(results_dir, 'PAM3polyaverage_confidence_scores_over_time_.png')
plt.savefig(plot_path)
plt.close()

print(f"Saved plot: {plot_path}")
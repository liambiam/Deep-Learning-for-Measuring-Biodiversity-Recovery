import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Path to CSV file
csv_file = 'E:/Results/CNN2secondConf2/predictions_PAM_2.csv'
output_folder = 'E:/Results/CNN2secondConf2'

# Load data
data = pd.read_csv(csv_file)

# Make sure 'Date' is datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Compute monthly average confidence scores for each class
data['Month'] = data['Date'].dt.to_period('M')
confidence_cols = [col for col in data.columns if col.startswith('Conf')]
monthly_avg_conf = data.groupby('Month')[confidence_cols].mean().reset_index()

# Convert 'Month' to datetime for plotting
monthly_avg_conf['Month'] = monthly_avg_conf['Month'].dt.to_timestamp()

# Plotting
plt.figure(figsize=(12, 6))
for conf in confidence_cols:
    # Scatter plot of monthly average confidence scores
    plt.scatter(monthly_avg_conf['Month'], monthly_avg_conf[conf], label=conf)
    
    # Fit linear trend line
    X = np.array(range(len(monthly_avg_conf))).reshape(-1, 1)
    y = monthly_avg_conf[conf].values
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    trend_line = lr_model.predict(X)
    
    # Plot trend line
    plt.plot(monthly_avg_conf['Month'], trend_line, linestyle='--', label=f'{conf} trend')

plt.xlabel('Month')
plt.ylabel('Average Confidence Score')
plt.title('Monthly Average Confidence Score for Each Class')
plt.legend(title='Class', loc='upper left')
plt.tight_layout()

# Save plot
plot_file = os.path.join(output_folder, 'PAM_3_avg_conf_scores_trend.png')
plt.savefig(plot_file)
plt.close()

print(f"Saved plot: {plot_file}")

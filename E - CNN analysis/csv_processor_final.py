import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# Path to the CSV file
csv_path = 'E:/Results/CNN2secondConf/predictions_PAM_2.csv'
results_dir = 'E:/Results/CNN2secondConf'

# Load the data
df = pd.read_csv(csv_path)

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the monthly statistics for each class
df['Month'] = df['Date'].dt.to_period('M')
confidence_columns = [col for col in df.columns if col.startswith('Conf')]

# Calculate different metrics: max, median, standard deviation, and mean
metrics = {
    'max': df.groupby('Month')[confidence_columns].max().reset_index(),
    'median': df.groupby('Month')[confidence_columns].median().reset_index(),
    'std': df.groupby('Month')[confidence_columns].std().reset_index(),
    'mean': df.groupby('Month')[confidence_columns].mean().reset_index()
}

# Convert 'Month' back to datetime for plotting
for metric_name, metric_df in metrics.items():
    metric_df['Month'] = metric_df['Month'].dt.to_timestamp()

    # Export metrics to CSV
    csv_export_path = os.path.join(results_dir, f'PAM_2_{metric_name}_confidence_scores.csv')
    metric_df.to_csv(csv_export_path, index=False)
    print(f"Saved CSV: {csv_export_path}")

    # Plotting
    plt.figure(figsize=(12, 6))
    for conf in confidence_columns:
        # Scatter plot of the monthly metric scores
        plt.scatter(metric_df['Month'], metric_df[conf], label=f'{conf} ({metric_name})')
        
        # Fit a linear trend line to the data
        X = np.array(range(len(metric_df))).reshape(-1, 1)
        y = metric_df[conf].values
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        
        # Plot the trend line
        plt.plot(metric_df['Month'], trend_line, linestyle='--', label=f'{conf} {metric_name} trend')

    plt.xlabel('Month')
    plt.ylabel(f'Monthly {metric_name.capitalize()} Confidence Score')
    plt.title(f'Monthly {metric_name.capitalize()} Confidence Score for Each Class')
    plt.legend(title='Class', loc='upper left')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f'PAM_2_{metric_name}_confidence_scores_over_time_trend_linear.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot: {plot_path}")

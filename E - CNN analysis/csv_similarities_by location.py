import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to CSV file
csv_file = 'E:/Results/CNN2secondConf/predictions_PAM_2.csv'

# Load data
data = pd.read_csv(csv_file)

# Convert 'Date' to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Get monthly average confidence for each class
data['Month'] = data['Date'].dt.to_period('M')
conf_cols = [col for col in data.columns if col.startswith('Conf')]
monthly_avg_conf = data.groupby('Month')[conf_cols].mean().reset_index()

# Convert 'Month' to datetime for plotting
monthly_avg_conf['Month'] = monthly_avg_conf['Month'].dt.to_timestamp()

# PAM_1 confidence
pam_1_conf = monthly_avg_conf[conf_cols[0]]

# Calculate absolute differences between PAM_1 and other PAMs
diffs = {}
for i, conf in enumerate(conf_cols[1:], start=2):
    diffs[f'PAM_1_vs_PAM_{i}'] = abs(pam_1_conf - monthly_avg_conf[conf])

# Convert differences to DataFrame
differences_df = pd.DataFrame(diffs)
differences_df['Month'] = monthly_avg_conf['Month']

# Plot differences over time
plt.figure(figsize=(12, 6))
for col in diffs.keys():
    plt.plot(differences_df['Month'], differences_df[col], label=col)

plt.xlabel('Month')
plt.ylabel('Absolute Difference in Confidence Scores')
plt.title('Similarity Between PAM 1 and Other PAMs Over Time')
plt.legend(title='PAM Pairs', loc='upper left')
plt.tight_layout()

# Save plot
results_folder = 'E:/Results/CNN2secondConf'
plot_file = os.path.join(results_folder, 'PAM_similarity_over_time.png')
plt.savefig(plot_file)
plt.close()

print(f"Saved plot: {plot_file}")

import pandas as pd
import matplotlib.pyplot as plt

# Load the monthly averages CSV file
monthly_csv_path = r"E:\Results\CNN2secondConf\monthly_average_confidence_scores.csv"
df_monthly = pd.read_csv(monthly_csv_path, parse_dates=['Date'], index_col='Date')

# Plotting setup
fig, ax = plt.subplots(figsize=(12, 8))

# Plot dots for monthly averages with accurate gold color
ax.plot(df_monthly.index, df_monthly['Conf1'], 'o', label='Conf1', color='blue', alpha=0.7)
ax.plot(df_monthly.index, df_monthly['Conf2'], 'o', label='Conf2', color='gold', alpha=0.7)  # Gold color
ax.plot(df_monthly.index, df_monthly['Conf3'], 'o', label='Conf3', color='red', alpha=0.7)
ax.plot(df_monthly.index, df_monthly['Conf4'], 'o', label='Conf4', color='green', alpha=0.7)

# Customize plot
ax.set_title('Monthly Average Confidence Scores')
ax.set_xlabel('Date')
ax.set_ylabel('Average Confidence Score')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

# Improve layout and show plot
plt.tight_layout()
plt.show()

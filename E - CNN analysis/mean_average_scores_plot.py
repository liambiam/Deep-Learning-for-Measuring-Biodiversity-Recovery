import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import datetime

# Path to monthly averages CSV file
monthly_csv = r"E:\Results\CNN2secondConf\monthly_average_confidence_scores2.csv"

# Load data
df_monthly = pd.read_csv(monthly_csv, parse_dates=['Date'], index_col='Date')

# Convert dates to numbers for fitting
df_monthly['Date_num'] = (df_monthly.index - df_monthly.index.min()).days

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Function to fit and plot trends
def plot_trends(ax, df, column_name, label_text, color_text):
    # Plot scatter of data
    ax.scatter(df.index, df[column_name], label=f'{label_text} Data', color=color_text, alpha=0.7)

    # Fit 2nd-degree polynomial
    p = Polynomial.fit(df['Date_num'], df[column_name], 2)
    
    # Get trend line values
    x_fit = np.linspace(df['Date_num'].min(), df['Date_num'].max(), 100)
    y_fit = p(x_fit)
    
    # Convert x_fit to dates
    dates_fit = df.index.min() + pd.to_timedelta(x_fit, unit='D')
    
    # Plot trend line
    ax.plot(dates_fit, y_fit, color=color_text, linestyle='--', label=f'{label_text} Trend')
    
    # Print trend line info
    print(f"Trend line for {label_text}:")
    print(f"x_fit from {x_fit.min()} to {x_fit.max()}")
    print(f"y_fit from {y_fit.min()} to {y_fit.max()}")
    print(f"dates_fit from {dates_fit.min()} to {dates_fit.max()}")
    print("---")

# Plot for each confidence score with trend lines
for col, color in zip(['Conf1', 'Conf2', 'Conf3', 'Conf4'], ['blue', '#FFD700', 'green', 'red']):
    plot_trends(ax, df_monthly, col, col, color)

# Customize plot
ax.set_title('Monthly Average Confidence Scores with Polynomial Trends')
ax.set_xlabel('Date')
ax.set_ylabel('Avg Confidence Score')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Print DataFrame info for debugging
print(df_monthly.info())
print(df_monthly.head())
print(df_monthly.tail())

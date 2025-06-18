import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

filename = "optuna_tuning_results.csv"

df = pd.read_csv(filename)

# Convert relevant columns to appropriate data types
df['datetime_start'] = pd.to_datetime(df['datetime_start'])
df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])
df['duration'] = df['datetime_complete'] - df['datetime_start']

# Filter for complete trials
complete_trials = df[df['state'] == 'COMPLETE'].copy()

print("--- Optuna Hyperparameter Tuning Results Analysis ---")
print("\nOverall Statistics for Complete Trials:")
print(complete_trials['value'].describe())

# Find the best trial (minimum 'value' as it's likely a loss metric)
best_trial = complete_trials.loc[complete_trials['value'].idxmin()]
print("\nBest Trial Found:")
print(best_trial)

print("\nHyperparameters of the Best Trial:")
# Extract only parameter columns for the best trial
params_cols = [col for col in df.columns if col.startswith('params_')]
print(best_trial[params_cols])


print("\nVisualizations")

# Distribution of 'value' for complete trials
plt.figure(figsize=(10, 6))
sns.histplot(complete_trials['value'], kde=True)
plt.title('Distribution of Objective Values (Loss)')
plt.xlabel('Objective Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Relationship between learning rate and objective values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=complete_trials, x='params_lr', y='value', hue='params_loss', s=100, alpha=0.7)
plt.xscale('log') # Learning rates are often log-distributed
# plt.title('Objective Value vs. Learning Rate (by Loss)')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('MAE')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Parameters')
plt.tight_layout()
plt.show()

# Objective value by loss function
plt.figure(figsize=(10, 6))
sns.boxplot(data=complete_trials, x='params_loss', y='value')
plt.title('Objective Value by Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Objective Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print("Plots generation complete!")

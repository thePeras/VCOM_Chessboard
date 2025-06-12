import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np # Import numpy for potential NaN handling

filename = "optuna_tuning_results_run4.csv"

df = pd.read_csv(filename)

# Convert relevant columns to appropriate data types
df['datetime_start'] = pd.to_datetime(df['datetime_start'])
df['datetime_complete'] = pd.to_datetime(df['datetime_complete'])
df['duration'] = df['datetime_complete'] - df['datetime_start']

# Filter for complete trials
complete_trials = df[df['state'] == 'COMPLETE'].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Analysis ---

print("--- Optuna Hyperparameter Tuning Results Analysis ---")
print("\n1. Overall Statistics for Complete Trials:")
print(complete_trials['value'].describe())

# Find the best trial (minimum 'value' as it's likely a loss metric)
best_trial = complete_trials.loc[complete_trials['value'].idxmin()]
print("\n2. Best Trial Found:")
print(best_trial)

print("\n3. Hyperparameters of the Best Trial:")
# Extract only parameter columns for the best trial
params_cols = [col for col in df.columns if col.startswith('params_')]
print(best_trial[params_cols])

# --- Visualizations ---

print("\n--- Visualizations ---")

# 1. Distribution of 'value' for complete trials
plt.figure(figsize=(10, 6))
sns.histplot(complete_trials['value'], kde=True)
plt.title('Distribution of Objective Values (Loss)')
plt.xlabel('Objective Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 2. Relationship between learning rate and objective values
style = "params_optimizer" if "params_optimizer" in df.columns else None
plt.figure(figsize=(10, 6))
sns.scatterplot(data=complete_trials, x='params_lr', y='value', hue='params_loss', style=style, s=100, alpha=0.7)
plt.xscale('log') # Learning rates are often log-distributed
plt.title('Objective Value vs. Learning Rate (by Loss and Optimizer)')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Objective Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Parameters')
plt.tight_layout()
plt.show()

# 3. Objective value by loss function
plt.figure(figsize=(10, 6))
sns.boxplot(data=complete_trials, x='params_loss', y='value')
plt.title('Objective Value by Loss Function')
plt.xlabel('Loss Function')
plt.ylabel('Objective Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

if "params_optimizer" in df.columns:
    # 4. Objective value by optimizer
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=complete_trials, x='params_optimizer', y='value')
    plt.title('Objective Value by Optimizer')
    plt.xlabel('Optimizer')
    plt.ylabel('Objective Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 5. Parallel Coordinates Plot for numerical parameters
try:
    print("\nGenerating Parallel Coordinates Plot...")
    # Select only numerical parameters and the objective value for plotting
    numerical_params_cols = [col for col in params_cols if complete_trials[col].dtype in ['float64', 'int64']]
    plot_data_numeric = complete_trials[numerical_params_cols + ['value']].copy()

    # Drop rows with NaN values in the selected columns, as parallel_coordinates can't handle them
    plot_data_numeric.dropna(inplace=True)

    # Clean up column names for plot
    plot_data_numeric.columns = [col.replace('params_', '') for col in plot_data_numeric.columns]

    if not plot_data_numeric.empty:
        plt.figure(figsize=(12, 8))
        pd.plotting.parallel_coordinates(plot_data_numeric, 'value', colormap=plt.cm.viridis)
        plt.title('Parallel Coordinates Plot of Numerical Hyperparameters and Objective Value')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Value (Normalized)')
        plt.tight_layout()
        plt.show()
    else:
        print("No complete trials with purely numerical parameters for parallel coordinates plot after dropping NaNs.")

except Exception as e: # Catch a broader exception to gracefully handle other issues
    print(f"\nError generating Parallel Coordinates Plot: {e}")
    print("This plot requires all selected columns to be numerical. Categorical parameters like 'loss' and 'optimizer' cannot be plotted directly.")
    print("Please ensure your parameters are numerical or consider alternative visualizations for categorical parameters.")


print("\nAnalysis complete. Examine the plots to gain insights into your hyperparameter tuning!")

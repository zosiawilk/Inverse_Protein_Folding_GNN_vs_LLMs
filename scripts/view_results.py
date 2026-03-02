import json
import pandas as pd

# Load results
results_file = '/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/evaluation_results.json'

with open(results_file) as f:
    data = json.load(f)

# Print overall statistics
print("="*60)
print("OVERALL STATISTICS")
print("="*60)
stats = data['overall_statistics']
print(f"Number of samples: {stats['num_samples']}")
print(f"Mean Recovery: {stats['mean_recovery']*100:.2f}% ± {stats['std_recovery']*100:.2f}%")
print(f"Median Recovery: {stats['median_recovery']*100:.2f}%")

# Create DataFrame for detailed analysis
df = pd.DataFrame(data['per_sample_results'])

# Show top 10 best predictions
print("\n" + "="*60)
print("TOP 10 BEST PREDICTIONS")
print("="*60)
print(df.nlargest(10, 'recovery')[['name', 'recovery', 'target_length', 'pred_length']])

# Show top 10 worst predictions
print("\n" + "="*60)
print("TOP 10 WORST PREDICTIONS")
print("="*60)
print(df.nsmallest(10, 'recovery')[['name', 'recovery', 'target_length', 'pred_length']])

# Recovery distribution
print("\n" + "="*60)
print("RECOVERY DISTRIBUTION")
print("="*60)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
df['recovery_bin'] = pd.cut(df['recovery'], bins=bins)
print(df['recovery_bin'].value_counts().sort_index())

# Save summary to CSV
df.to_csv('/home/zww20/rds/hpc-work/GDL/PiFold-main/results/llama_3di/evaluation_summary.csv', index=False)
print(f"\nSummary saved to: evaluation_summary.csv")

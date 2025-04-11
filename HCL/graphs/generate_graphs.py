import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the evaluation data
data = {
    "Dataset": ["FFHQ", "Places365", "ImageNet"],
    "LPIPS (Eval)": [0.0159, 0.0246, 0.0204],
    "LPIPS (No Corr)": [0.0800, 0.1323, 0.0325],
    "PSNR (Eval)": [34.37, 33.99, 34.91],
    "PSNR (No Corr)": [23.92, 24.56, 34.36],
    "SSIM (Eval)": [0.9657, 0.9712, 0.9706],
    "SSIM (No Corr)": [0.8830, 0.8717, 0.9606],
    "FID (Eval)": [1.73, 2.82, 2.09],
    "FID (No Corr)": [8.10, 13.65, 2.98],
}

# Global color palette (used in both functions)
colors = ["steelblue", "darkorange"]

df = pd.DataFrame(data)

# Function to plot and save metrics
def plot_metrics(df):
    metrics = ["LPIPS", "PSNR", "SSIM", "FID"]
    colors = ["blue", "orange"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        eval_vals = df[f"{metric} (Eval)"]
        no_corr_vals = df[f"{metric} (No Corr)"]
        
        x = range(len(df["Dataset"]))
        width = 0.35
        plt.bar([p - width/2 for p in x], eval_vals, width, label="Base Evaluation", color=colors[0])
        plt.bar([p + width/2 for p in x], no_corr_vals, width, label="No Corruption Evaluation", color=colors[1])
        
        plt.xticks(ticks=x, labels=df["Dataset"], fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"{metric} Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        for i, (val1, val2) in enumerate(zip(eval_vals, no_corr_vals)):
            offset = max(val1, val2) * 0.02  # 2% of bar height as vertical offset
            plt.text(i - 0.2, val1 + offset, f"{val1:.2f}", ha='center', va='bottom', fontsize=10)
            plt.text(i + 0.2, val2 + offset, f"{val2:.2f}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{metric.lower()}_comparison.png", dpi=300)
        plt.close()

# Bonus: Seaborn combined plot
def seaborn_overview(df):
    df_melted = df.melt(id_vars="Dataset", var_name="Metric", value_name="Score")
    df_melted["Type"] = df_melted["Metric"].apply(lambda x: "Eval" if "Eval" in x else "No Corr")
    df_melted["Metric"] = df_melted["Metric"].apply(lambda x: x.split()[0])

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Dataset", y="Score", hue="Type", errorbar=None, palette=colors)
    plt.title("Overview of Metrics on Eval vs No Corruption", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Data Type")
    plt.tight_layout()
    plt.savefig("overview_metrics_comparison.png", dpi=300)
    plt.close()

# Run the plotting functions
plot_metrics(df)
# seaborn_overview(df)

print("Plots saved as PNG files in the current directory.")

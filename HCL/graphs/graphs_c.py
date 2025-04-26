import pandas as pd
import matplotlib.pyplot as plt

# Define ImageNet-specific data
data = {
    "Condition": ["Base Eval", "No Corruption", "Snow", "Impulse Noise", "Gaussian Noise", "Shot Noise"],
    "LPIPS": [0.0204, 0.0325, 0.810, 0.263, 0.716, 0.482],
    "PSNR": [34.91, 34.36, 12.68, 21.08, 15.36, 16.877],
    "SSIM": [0.9706, 0.9606, 0.357, 0.7685, 0.386, 0.556],
    "FID": [2.09, 2.98, 274.99, 32.95, 146.97, 93.485],
}

df = pd.DataFrame(data)

# Define plotting function
def plot_imagenet_metrics(df):
    metrics = ["LPIPS", "PSNR", "SSIM", "FID"]
    colors = ["steelblue"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        values = df[metric]
        x = range(len(df["Condition"]))

        plt.bar(x, values, color=colors[0])
        plt.xticks(ticks=x, labels=df["Condition"], rotation=20, fontsize=11)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"ImageNet - {metric} Across Corruption Types", fontsize=14, fontweight="bold")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        for i, val in enumerate(values):
            offset = max(values) * 0.02
            plt.text(i, val + offset, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"imagenet_{metric.lower()}_corruption.png", dpi=300)
        plt.close()

# Run plotting
plot_imagenet_metrics(df)

print("ImageNet corruption metric plots saved.")

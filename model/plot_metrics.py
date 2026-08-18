import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    results_file = "results/predictions.csv"
    if not os.path.exists(results_file):
        print(f"Cannot find {results_file}")
        return

    df = pd.read_csv(results_file)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Accuracy vs Error Threshold Curve (Localization "PR Curve")
    # ---------------------------------------------------------
    # In localization, we use a Cumulative Error Distribution (or PCK curve)
    # instead of a standard PR curve since we don't have binary classes.
    plt.figure(figsize=(8, 6))
    thresholds = np.linspace(0, 100, 1000)
    accuracies = [(df['distance_px'] <= t).mean() * 100 for t in thresholds]
    
    plt.plot(thresholds, accuracies, color='blue', linewidth=2)
    plt.axvline(x=5, color='red', linestyle='--', label='5px Hackathon Threshold')
    plt.fill_between(thresholds, accuracies, alpha=0.1, color='blue')
    
    plt.title('Localization Accuracy vs. Error Threshold', fontsize=14)
    plt.xlabel('Error Threshold (Pixels)', fontsize=12)
    plt.ylabel('Accuracy (% of samples passing)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    curve_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {curve_path}")

    # ---------------------------------------------------------
    # 2. Performance by Architecture (Localization "Confusion Matrix")
    # ---------------------------------------------------------
    # Since we can't do a discrete confusion matrix for continuous coordinates,
    # we break down errors by architecture to see what confuses the model.
    plt.figure(figsize=(12, 6))
    
    # Calculate Pass@5px and Mean Error for each architecture
    df['pass_5px'] = df['distance_px'] <= 5.0
    arch_stats = df.groupby('architecture').agg(
        mean_error=('distance_px', 'mean'),
        pass_rate=('pass_5px', lambda x: x.mean() * 100)
    ).reset_index()
    
    # Sort by worst mean error
    arch_stats = arch_stats.sort_values('mean_error', ascending=False)
    
    sns.barplot(data=arch_stats, x='mean_error', y='architecture', palette='Reds_r')
    plt.title('Mean Error by Architecture (Where the model gets confused)', fontsize=14)
    plt.xlabel('Mean Error (Pixels)', fontsize=12)
    plt.ylabel('Wafer Architecture', fontsize=12)
    
    # Add text labels on bars
    for i, row in enumerate(arch_stats.itertuples()):
        plt.text(row.mean_error + 1, i, f"{row.mean_error:.1f}px (Pass: {row.pass_rate:.1f}%)", va='center')
        
    plt.grid(axis='x', alpha=0.3)
    
    arch_path = os.path.join(output_dir, "architecture_confusion.png")
    plt.savefig(arch_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {arch_path}")
    
    # ---------------------------------------------------------
    # 3. 2D Spatial Error Scatter (Bias check)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    df['err_x'] = df['pred_x'] - df['gt_x']
    df['err_y'] = df['pred_y'] - df['gt_y']
    
    plt.scatter(df['err_x'], df['err_y'], alpha=0.5, c='purple', s=10)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Draw circles for 5px, 20px, 50px thresholds
    for r in [5, 20, 50]:
        circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.title('2D Spatial Error Distribution', fontsize=14)
    plt.xlabel('X Error (Pixels)', fontsize=12)
    plt.ylabel('Y Error (Pixels)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(output_dir, "spatial_error.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {scatter_path}")

if __name__ == "__main__":
    main()

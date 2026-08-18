import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/predictions.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"Cannot find {args.predictions}")
        return

    df = pd.read_csv(args.predictions)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which distance column to use
    dist_col = 'hybrid_dist' if 'hybrid_dist' in df.columns else 'distance_px'

    # We will create an "Error Classification Matrix" which acts as our Confusion Matrix.
    # We bin the continuous error into categorical classes of success.
    bins = [-1, 2, 5, 20, float('inf')]
    labels = ['Perfect (<2px)', 'Good (2-5px)', 'Marginal (5-20px)', 'Failure (>20px)']
    
    df['Error Category'] = pd.cut(df[dist_col], bins=bins, labels=labels)

    # Create a cross-tabulation (Confusion Matrix)
    confusion_matrix = pd.crosstab(df['architecture'], df['Error Category'], normalize='index') * 100
    
    # Ensure columns are in the correct order even if some bins are empty
    for col in labels:
        if col not in confusion_matrix.columns:
            confusion_matrix[col] = 0.0
    confusion_matrix = confusion_matrix[labels]

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.1f', 
                cbar_kws={'label': '% of predictions'})
    
    plt.title('Prediction Quality Confusion Matrix\n(Actual Architecture vs. Error Category)', fontsize=14)
    plt.xlabel('Prediction Error Classification', fontsize=12)
    plt.ylabel('Actual Wafer Architecture', fontsize=12)
    plt.yticks(rotation=0)
    
    matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {matrix_path}")

if __name__ == "__main__":
    main()


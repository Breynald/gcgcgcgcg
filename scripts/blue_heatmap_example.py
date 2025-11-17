"""
Generate example heatmap with blue color scheme.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import nanogcg tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogcg.tools import create_heatmap


def create_blue_heatmap_example():
    """Create a single example with blue color scheme."""

    # Sample data showing increasing perplexity with table size
    sample_data = np.array([
        [12.5, 14.2, 16.8, 18.9, 21.2, 23.5, 25.8, 27.9, 29.5],
        [13.1, 14.8, 17.2, 19.1, 21.8, 24.1, 26.2, 28.3, 30.1],
        [14.2, 15.9, 18.1, 20.2, 22.8, 25.0, 27.1, 29.2, 31.0],
        [15.8, 17.2, 19.3, 21.5, 23.9, 26.1, 28.2, 30.3, 32.1],
        [17.5, 18.9, 20.8, 22.9, 25.2, 27.4, 29.5, 31.6, 33.4],
        [19.2, 20.5, 22.3, 24.4, 26.7, 28.9, 31.0, 33.1, 34.9],
        [21.0, 22.2, 23.9, 26.0, 28.3, 30.5, 32.6, 34.7, 36.5],
        [22.8, 23.9, 25.6, 27.7, 29.9, 32.1, 34.2, 36.3, 38.1],
        [24.5, 25.6, 27.2, 29.3, 31.5, 33.7, 35.8, 37.9, 39.7]
    ])

    output_path = "/work/table-fp/nanoGCG-main/assets/blue_heatmap_example.png"
    create_heatmap(sample_data, output_path, "Perplexity Heatmap for GCG Optimization (Blue Theme)")

    print("Blue heatmap example generated:")
    print(f"  Heatmap: {output_path}")
    print(f"  Best: 1x1 table (PPL: 12.5) - Lightest blue")
    print(f"  Worst: 9x9 table (PPL: 39.7) - Darkest blue")


def create_alternative_blue_heatmap():
    """Create alternative example with different blue gradient."""

    # Create sample matrix
    np.random.seed(123)
    matrix = np.random.uniform(15, 35, (9, 9))

    # Add some structure - make smaller tables better
    for i in range(9):
        for j in range(9):
            matrix[i, j] += (i + j) * 0.8  # Size penalty

    output_path = "/work/table-fp/nanoGCG-main/assets/blue_heatmap_alternative.png"

    # Create the plot manually to demonstrate different blue shades
    plt.figure(figsize=(12, 10))

    # Use different blue colormap variants
    sns.heatmap(matrix,
                annot=True,
                fmt='.1f',
                cmap='YlGnBu',  # Yellow-Green-Blue for more contrast
                square=True,
                cbar_kws={'label': 'Perplexity'},
                annot_kws={'size': 10})

    plt.title("Alternative Blue Heatmap for GCG Optimization", fontsize=16, fontweight='bold')
    plt.xlabel('Number of Columns', fontsize=12)
    plt.ylabel('Number of Rows', fontsize=12)
    plt.xticks(np.arange(9) + 0.5, range(1, 10))
    plt.yticks(np.arange(9) + 0.5, range(1, 10))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Alternative blue heatmap: {output_path}")
    print(f"  Range: {np.min(matrix):.1f} - {np.max(matrix):.1f}")


def main():
    """Generate blue heatmap examples."""
    print("=" * 60)
    print("GENERATING BLUE HEATMAP EXAMPLES")
    print("=" * 60)

    create_blue_heatmap_example()
    print()
    create_alternative_blue_heatmap()

    print("\n" + "=" * 60)
    print("BLUE HEATMAP EXAMPLES GENERATED!")
    print("=" * 60)
    print("Files saved to /work/table-fp/nanoGCG-main/assets/")
    print("Lighter blue = Lower perplexity (better)")
    print("Darker blue = Higher perplexity (worse)")


if __name__ == "__main__":
    main()
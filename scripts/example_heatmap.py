"""
Example script to generate sample heatmap for demonstration.
Creates realistic perplexity values for different table sizes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import nanogcg tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogcg.tools import create_heatmap
import seaborn as sns


def generate_sample_perplexity_matrix(max_size=9):
    """Generate realistic sample perplexity values for demonstration.

    Args:
        max_size: Maximum table size (creates max_size x max_size matrix)

    Returns:
        np.ndarray: 2D array with sample perplexity values
    """
    np.random.seed(42)  # For reproducible results

    # Base perplexity values with realistic patterns
    base_perplexity = 20.0

    # Create matrix with some realistic patterns:
    # - Smaller tables tend to have lower perplexity
    # - Square tables might perform slightly better than rectangular ones
    # - Add some noise to make it realistic
    perplexity_matrix = np.zeros((max_size, max_size))

    for i in range(max_size):
        for j in range(max_size):
            rows, cols = i + 1, j + 1

            # Base calculation: larger tables have higher perplexity
            size_factor = np.sqrt(rows * cols) / 3.0

            # Square table bonus (better performance for square shapes)
            shape_bonus = 0.0
            if abs(rows - cols) <= 1:  # Nearly square
                shape_bonus = -1.5

            # Add some realistic variation
            noise = np.random.normal(0, 0.8)

            # Final perplexity calculation
            perplexity = base_perplexity + size_factor * 3.0 + shape_bonus + noise

            # Ensure reasonable bounds
            perplexity = max(10.0, min(40.0, perplexity))

            perplexity_matrix[i, j] = perplexity

    return perplexity_matrix


def create_multiple_heatmap_examples():
    """Create multiple heatmap examples with different targets."""

    # Create output directory
    output_dir = "/work/table-fp/nanoGCG-main/assets"
    os.makedirs(output_dir, exist_ok=True)

    # Different scenarios with different patterns
    scenarios = [
        {
            "name": "copyright_optimization",
            "title": "Perplexity Heatmap for 'Copyright' Target",
            "base_perplexity": 18.0,
            "size_effect": 2.5,
            "noise_level": 0.6
        },
        {
            "name": "hello_world_optimization",
            "title": "Perplexity Heatmap for 'Hello World' Target",
            "base_perplexity": 15.0,
            "size_effect": 2.0,
            "noise_level": 0.8
        },
        {
            "name": "complex_target_optimization",
            "title": "Perplexity Heatmap for Complex Target Optimization",
            "base_perplexity": 25.0,
            "size_effect": 3.5,
            "noise_level": 1.0
        }
    ]

    for scenario in scenarios:
        np.random.seed(42)  # Keep some consistency

        max_size = 9
        perplexity_matrix = np.zeros((max_size, max_size))

        for i in range(max_size):
            for j in range(max_size):
                rows, cols = i + 1, j + 1

                # Base calculation
                size_factor = np.sqrt(rows * cols) / 3.0

                # Shape bonus
                shape_bonus = 0.0
                if abs(rows - cols) <= 1:
                    shape_bonus = -1.2

                # Noise
                noise = np.random.normal(0, scenario["noise_level"])

                # Final calculation
                perplexity = (scenario["base_perplexity"] +
                            size_factor * scenario["size_effect"] +
                            shape_bonus + noise)

                # Bounds
                perplexity = max(8.0, min(45.0, perplexity))
                perplexity_matrix[i, j] = perplexity

        # Generate heatmap
        output_path = os.path.join(output_dir, f"{scenario['name']}_heatmap.png")
        create_heatmap(perplexity_matrix, output_path, scenario["title"])

        # Also save the data
        data_path = os.path.join(output_dir, f"{scenario['name']}_data.csv")
        np.savetxt(data_path, perplexity_matrix, delimiter=',', fmt='%.2f')

        print(f"Generated {scenario['name']}:")
        print(f"  Heatmap: {output_path}")
        print(f"  Data: {data_path}")
        print(f"  Best performance: {np.min(perplexity_matrix):.2f} (PPL)")
        print(f"  Worst performance: {np.max(perplexity_matrix):.2f} (PPL)")
        print(f"  Average: {np.mean(perplexity_matrix):.2f} (PPL)")
        print()


def create_simple_example():
    """Create a single, simple example."""

    # Simple sample data
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

    output_path = "/work/table-fp/nanoGCG-main/assets/example_heatmap.png"
    create_heatmap(sample_data, output_path, "Example Perplexity Heatmap for GCG Optimization")

    print("Simple example generated:")
    print(f"  Heatmap: {output_path}")
    print(f"  Best: 1x1 table (PPL: 12.5)")
    print(f"  Worst: 9x9 table (PPL: 39.7)")


def main():
    """Generate example heatmaps."""
    print("=" * 60)
    print("GENERATING EXAMPLE HEATMAPS")
    print("=" * 60)

    # Create simple example first
    create_simple_example()

    print("\n" + "=" * 60)
    print("GENERATING MULTIPLE SCENARIO EXAMPLES")
    print("=" * 60)

    # Create multiple scenario examples
    create_multiple_heatmap_examples()

    print("=" * 60)
    print("ALL EXAMPLES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("Check the /work/table-fp/nanoGCG-main/assets/ directory for the heatmap images.")


if __name__ == "__main__":
    main()
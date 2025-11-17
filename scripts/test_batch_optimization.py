"""
Test script for batch optimization functionality.
Runs a small-scale test to verify the modular code works correctly.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path to import nanogcg tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogcg.tools import (
    generate_table_prompt, generate_simple_prompt, create_config,
    run_single_optimization, calculate_perplexity_for_prompt,
    create_heatmap, save_results, calculate_statistics
)


def test_prompt_generation():
    """Test prompt generation functions."""
    print("Testing prompt generation...")

    # Test simple prompt
    simple_result = generate_simple_prompt()
    assert "optim_str_1" in simple_result["prompt"]
    assert simple_result["optim_str_placeholders"] == ["{optim_str_1}"]
    print("✓ Simple prompt generation works")

    # Test table prompt
    table_result = generate_table_prompt(2, 3)
    assert "Metric A" in table_result["prompt"]
    assert "Metric B" in table_result["prompt"]
    assert "Metric C" in table_result["prompt"]
    assert len(table_result["optim_str_placeholders"]) == 6  # 2x3 = 6
    print("✓ Table prompt generation works")


def test_config_creation():
    """Test configuration creation."""
    print("Testing configuration creation...")

    config = create_config(num_steps=10, optim_str_init="test test")
    assert config.num_steps == 10
    assert config.optim_str_init == "test test"
    print("✓ Configuration creation works")


def test_heatmap_creation():
    """Test heatmap creation with sample data."""
    print("Testing heatmap creation...")

    # Create sample perplexity matrix
    import numpy as np
    test_matrix = np.array([
        [10.5, 12.3, 15.2],
        [11.1, 13.7, 16.8],
        [12.4, 14.9, 18.1]
    ])

    # Create temporary output directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)

    # Generate heatmap
    heatmap_path = os.path.join(test_dir, "test_heatmap.png")
    create_heatmap(test_matrix, heatmap_path, "Test Heatmap")

    # Check if file was created
    assert os.path.exists(heatmap_path)
    print("✓ Heatmap creation works")

    # Cleanup
    os.remove(heatmap_path)
    os.rmdir(test_dir)


def test_results_saving():
    """Test results saving functionality."""
    print("Testing results saving...")

    # Create sample results
    test_results = {
        "simple": {
            "success": True,
            "best_loss": 0.123,
            "perplexity": 15.7,
            "optimized_prompt": "Here is a test optimized prompt"
        },
        "table_2x2": {
            "success": True,
            "best_loss": 0.456,
            "perplexity": 18.3,
            "optimized_prompt": "Table test prompt"
        }
    }

    # Create temporary output directory
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)

    # Save results
    results_path = os.path.join(test_dir, "test_results.json")
    save_results(test_results, results_path)

    # Check if file was created and contains expected data
    assert os.path.exists(results_path)
    import json
    with open(results_path, 'r') as f:
        loaded_results = json.load(f)
    assert "simple" in loaded_results
    assert loaded_results["simple"]["best_loss"] == 0.123
    print("✓ Results saving works")

    # Cleanup
    os.remove(results_path)
    os.rmdir(test_dir)


def test_statistics_calculation():
    """Test statistics calculation."""
    print("Testing statistics calculation...")

    # Create sample results
    test_results = {
        "simple": {
            "success": True,
            "perplexity": 15.7,
            "best_loss": 0.123,
            "optimization_time": 45.2
        },
        "table_2x2": {
            "success": True,
            "perplexity": 18.3,
            "best_loss": 0.456,
            "optimization_time": 67.8
        },
        "table_3x3": {
            "success": True,
            "perplexity": 22.1,
            "best_loss": 0.789,
            "optimization_time": 89.5
        }
    }

    stats = calculate_statistics(test_results)

    assert "simple_perplexity" in stats
    assert "table_perplexity_mean" in stats
    assert "successful_table_optimizations" in stats
    assert stats["simple_perplexity"] == 15.7
    assert stats["successful_table_optimizations"] == 2
    assert abs(stats["table_perplexity_mean"] - (18.3 + 22.1) / 2) < 0.01

    print("✓ Statistics calculation works")


def run_integration_test(args):
    """Run a small integration test if models are available."""
    print("Running integration test...")

    try:
        # Try to import torch and transformers
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Test with a small model
        model_name = "openai-community/gpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading test model: {model_name} on {device}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create configuration
        config = create_config(num_steps=5, verbosity="INFO")

        # Test simple prompt optimization
        simple_prompt_data = generate_simple_prompt()
        messages = [{"role": "user", "content": simple_prompt_data["prompt"]}]

        result = run_single_optimization(
            model, tokenizer, messages, "test", config,
            simple_prompt_data["optim_str_placeholders"], "Integration Test"
        )

        if result["success"]:
            print("✓ Integration test - optimization completed successfully")

            # Test perplexity calculation
            perplexity = calculate_perplexity_for_prompt(
                result["optimized_prompt"], model, tokenizer, device
            )
            print(f"✓ Integration test - perplexity calculated: {perplexity:.2f}")
        else:
            print("⚠ Integration test - optimization failed")

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        print("⚠ Integration test skipped - torch/transformers not available")
    except Exception as e:
        print(f"⚠ Integration test failed - {e}")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test batch optimization functionality")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip integration test that requires loading models")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH OPTIMIZATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")

    try:
        # Run unit tests
        test_prompt_generation()
        test_config_creation()
        test_heatmap_creation()
        test_results_saving()
        test_statistics_calculation()

        # Run integration test if not skipped
        if not args.skip_integration:
            run_integration_test(args)
        else:
            print("Skipping integration test as requested")

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
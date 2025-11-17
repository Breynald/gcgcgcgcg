#!/usr/bin/env python3
"""
Example script demonstrating GCG thinking optimization.

This script uses the modified GCG algorithm to optimize adversarial strings
that maximize the length of thinking content between <thinking> tags.
"""

import sys
import os
import warnings
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanogcg.gcg_thinking_simple import run_gcg_thinking, GCGConfig

def main():
    # Set device - you can modify this to use different GPU
    # Options: "cuda:0", "cuda:1", "cuda:2", "cuda:3", etc.
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # If you want to use a specific GPU, you can set it like this:
    # device = "cuda:1"  # Use second GPU
    # device = "cuda:2"  # Use third GPU, etc.

    # Check available GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the current device
    if torch.cuda.is_available():
        torch.cuda.set_device(0 if device == "cuda:0" else int(device.split(":")[1]))

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "/work/models/Qwen/Qwen3-1.7B"  # You can change this to any suitable model

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map=device  # Specify GPU device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to gpt2...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure model is on the correct device
    model = model.to(device)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure the GCG thinking optimization
    config = GCGConfig(
        num_steps=50,  # Reduced number of steps for faster testing
        search_width=4,  # Reduced search width for faster evaluation
        max_generation_tokens=2000,  # Reduced generation tokens for speed
        use_prefix_cache=False,  # Disable prefix cache to avoid compatibility issues with Qwen3
        seed=None,  # Disable deterministic behavior to avoid CuBLAS warnings
        verbosity="INFO",  # Logging level

        # Custom thinking tags - you can modify these!
        thinking_start_tag="<think>",  # Start tag for thinking content
        thinking_end_tag="</think>"      # End tag for thinking content

        # Examples of other tags you could use:
        # thinking_start_tag="<reasoning>"
        # thinking_end_tag="</reasoning>"
        # thinking_start_tag="<analysis>"
        # thinking_end_tag="</analysis>"
        # thinking_start_tag="<step>"
        # thinking_end_tag="</step>"
    )

    # Define the prompt - we want the model to think extensively
    messages = [
        {
            "role": "user",
            "content": " {optim_str}. Please explain the above content in detail."
        }
    ]

    print("Starting GCG thinking optimization...")
    print("This will optimize the adversarial string to maximize thinking content.")

    # Run the thinking optimization
    result = run_gcg_thinking(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        config=config
    )

    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)

    print(f"Best loss: {result.best_loss:.4f}")
    print(f"Best optimized string: '{result.best_string}'")

    # Test the optimized string with the model
    print("\n" + "="*50)
    print("TESTING OPTIMIZED STRING")
    print("="*50)

    # Create the final prompt with the optimized string
    final_prompt = messages[0]["content"].replace("{optim_str}", result.best_string)

    # Generate response
    inputs = tokenizer(final_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 200,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated response:")
    print("-" * 30)
    print(generated_text)
    print("-" * 30)

    # Extract and analyze thinking content
    from nanogcg.gcg_thinking import extract_thinking_content

    thinking_content = extract_thinking_content(generated_text)
    if thinking_content:
        thinking_tokens = len(tokenizer.encode(thinking_content, add_special_tokens=False))
        print(f"\nExtracted thinking content ({thinking_tokens} tokens):")
        print("-" * 30)
        print(thinking_content)
        print("-" * 30)
    else:
        print("\nNo thinking content found in <thinking> tags.")

    # Show loss progression
    print(f"\nFinal loss: {result.losses[-1]:.4f}")
    print(f"Loss improvement: {result.losses[0] - result.losses[-1]:.4f}")

if __name__ == "__main__":
    main()
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def calculate_perplexity(text, model, tokenizer, device='cuda'):
    """
    Calculate the perplexity of a given text using a pre-trained language model.
    
    Args:
        text (str): The input text to calculate perplexity for
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        device (str): Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        float: Perplexity score of the text
    """
    # Tokenize the input text
    encodings = tokenizer(text, return_tensors='pt')
    
    # Move tensors to the specified device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Get the model output (logits)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Calculate negative log likelihood
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    # Flatten the tensors
    logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    labels_flat = shift_labels.view(-1)
    attention_mask_flat = shift_attention_mask.view(-1)
    
    # Calculate loss only for tokens that are part of the input (ignore padding)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits_flat, labels_flat)
    
    # Apply attention mask to ignore padding tokens
    masked_loss = loss * attention_mask_flat
    
    # Calculate perplexity
    avg_loss = masked_loss.sum() / attention_mask_flat.sum()
    perplexity = math.exp(avg_loss)
    
    return perplexity

def load_model_and_tokenizer(model_name, device='cuda'):
    """
    Load a pre-trained language model and its tokenizer.
    
    Args:
        model_name (str): Name or path of the pre-trained model
        device (str): Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity of text')
    parser.add_argument('--text', type=str, help='Input text to calculate perplexity for')
    parser.add_argument('--file', type=str, help='Path to file containing text')
    parser.add_argument('--model', type=str, default='gpt2', help='Pre-trained model name or path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the model on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    print(f"Model loaded on device: {args.device}")
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File {args.file} not found.")
            return
    else:
        print("Error: Please provide either --text or --file argument.")
        return
    
    # Calculate perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(text, model, tokenizer, args.device)
    
    # Print result
    print(f"Text: {text}")
    print(f"Perplexity: {perplexity}")

if __name__ == "__main__":
    main()
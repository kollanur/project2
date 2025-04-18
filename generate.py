# generate.py
import torch
import argparse
from pathlib import Path
from models import RNNModel, LSTMModel, TransformerModel
from tokenizer import BPETokenizer

class TextGenerator:
    def __init__(
        self,
        model_type='transformer',
        model_path=None,
        tokenizer_path='tokenizer/bpe_model.model',
        embedding_dim=768,
        hidden_dim=1024,
        num_layers=6,
        nhead=12,
        dropout=0.1,
        max_seq_length=512,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = BPETokenizer(tokenizer_path)
        
        # Initialize model
        print(f"Initializing {model_type.upper()} model...")
        vocab_size = self.tokenizer.vocab_size()
        
        if model_type.lower() == 'transformer':
            self.model = TransformerModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_length=max_seq_length
            )
        elif model_type.lower() == 'lstm':
            self.model = LSTMModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type.lower() == 'rnn':
            self.model = RNNModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        if model_path:
            print(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompt,
        max_length=100,
        temperature=0.7,
        num_return_sequences=1,
        min_length=10
    ):
        """
        Generate text from a prompt.
        
        Args:
            prompt (str): The input prompt to generate from
            max_length (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (higher = more random)
            num_return_sequences (int): Number of sequences to generate
            min_length (int): Minimum number of tokens to generate
        
        Returns:
            list: List of generated sequences
        """
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            try:
                # Generate text using the model's prompt method
                generated_text = self.model.prompt(
                    self.tokenizer,
                    prompt,
                    max_seq_length=max_length,
                    temperature=temperature
                )
                generated_sequences.append(generated_text)
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                continue
        
        return generated_sequences

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained language model')
    parser.add_argument('--model_type', type=str, default='transformer',
                      choices=['transformer', 'lstm', 'rnn'])
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/bpe_model.model',
                      help='Path to tokenizer model')
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=512)
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(
        model_type=args.model_type,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_seq_length=args.max_seq_length
    )
    
    print("\nText Generation Interface")
    print("------------------------")
    print("Type 'quit' to exit")
    print("Type 'help' for generation parameters")
    
    while True:
        try:
            # Get prompt
            prompt = input("\nEnter your prompt: ")
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("\nGeneration Parameters:")
                print("- temperature: Controls randomness (0.1-1.0, default: 0.7)")
                print("- max_length: Maximum tokens to generate (default: 100)")
                print("- num_sequences: Number of sequences to generate (default: 1)")
                continue
            
            # Get generation parameters
            temperature = float(input("Temperature (0.1-1.0, default: 0.7): ") or 0.7)
            max_length = int(input("Max length (default: 100): ") or 100)
            num_sequences = int(input("Number of sequences (default: 1): ") or 1)
            
            # Generate text
            generated_sequences = generator.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_sequences
            )
            
            # Print results
            print("\nGenerated Sequences:")
            for i, sequence in enumerate(generated_sequences, 1):
                print(f"\n--- Sequence {i} ---")
                print(sequence)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()

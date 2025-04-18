# tokenizer.py
import sentencepiece as spm
from pathlib import Path
import argparse

class BPETokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)
    
    def train(self, input_file, vocab_size=10000, model_prefix='bpe_model'):
        """Train the tokenizer on input text."""
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=3,           # Specify only pad_id
            eos_id=2,          # Specify only eos_id
            bos_id=1,          # Specify only bos_id
            unk_id=0,          # Specify only unk_id
            normalization_rule_name='nmt_nfkc',
            add_dummy_prefix=True,
            remove_extra_whitespaces=True
        )
        self.sp.load(f"{model_prefix}.model")
    
    def encode(self, text):
        """Convert text to token IDs."""
        return self.sp.encode_as_ids(text)
    
    def decode(self, tokens):
        """Convert token IDs back to text."""
        return self.sp.decode_ids(tokens)
    
    def vocab_size(self):
        """Get the vocabulary size."""
        return self.sp.get_piece_size()

def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Input text file for training')
    parser.add_argument('--vocab_size', type=int, default=10000,
                      help='Vocabulary size')
    parser.add_argument('--model_prefix', type=str, default='bpe_model',
                      help='Output model prefix')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    # First, create a single text file from the JSONL
    print("Preparing training data...")
    text_file = Path(args.input_file).parent / "train.txt"
    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(text_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                text = eval(line)['text']
                f_out.write(text + '\n')
            except:
                continue
    
    print("Training tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(
        input_file=str(text_file),
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix
    )
    
    # Clean up temporary file
    text_file.unlink()
    print("Training complete!")

if __name__ == "__main__":
    main()

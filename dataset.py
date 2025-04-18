# dataset.py
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # Load data from jsonl file
        data_path = Path(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Handle both JSON and plain text formats
                    try:
                        example = json.loads(line.strip())
                        text = example['text']
                    except json.JSONDecodeError:
                        text = line.strip()
                    
                    if text:  # Only add non-empty examples
                        self.examples.append(text)
                except Exception as e:
                    continue
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(tokens) > self.max_seq_length - 1:  # -1 for EOS token
            tokens = tokens[:self.max_seq_length - 1]
        
        # Add EOS token
        tokens.append(self.tokenizer.sp.eos_id())
        
        # Create input and target sequences
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.max_seq_length - 1 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.sp.pad_id()] * padding_length
            target_ids = target_ids + [self.tokenizer.sp.pad_id()] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

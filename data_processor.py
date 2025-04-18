# data_processor.py
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import random
import re

class TextProcessor:
    def __init__(self, input_dir, output_dir, min_length=100, max_length=1024):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.max_length = max_length
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patterns for cleaning text
        self.patterns = {
            'gutenberg_header': r'\*{3}.*?START OF .*?GUTENBERG.*?\*{3}',
            'gutenberg_footer': r'\*{3}.*?END OF .*?GUTENBERG.*?\*{3}',
            'chapter': r'CHAPTER [IVXLC\d]+',
            'multiple_newlines': r'\n{3,}',
            'multiple_spaces': r' {2,}'
        }

    def clean_text(self, text):
        """Clean raw text by removing headers, footers, and normalizing spacing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove Project Gutenberg header and footer
        text = re.sub(self.patterns['gutenberg_header'], '', text, flags=re.DOTALL)
        text = re.sub(self.patterns['gutenberg_footer'], '', text, flags=re.DOTALL)
        
        # Remove chapter headers
        text = re.sub(self.patterns['chapter'], '', text)
        
        # Normalize spacing
        text = re.sub(self.patterns['multiple_newlines'], '\n\n', text)
        text = re.sub(self.patterns['multiple_spaces'], ' ', text)
        
        return text.strip()

    def split_into_segments(self, text):
        """Split text into segments of appropriate length."""
        segments = []
        paragraphs = text.split('\n\n')
        
        current_segment = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed max_length,
            # save current segment and start a new one
            if current_length + len(paragraph) > self.max_length:
                if current_length >= self.min_length:
                    segments.append(' '.join(current_segment))
                current_segment = [paragraph]
                current_length = len(paragraph)
            else:
                current_segment.append(paragraph)
                current_length += len(paragraph)
        
        # Add the last segment if it meets minimum length
        if current_length >= self.min_length:
            segments.append(' '.join(current_segment))
        
        return segments

    def process_file(self, file_path):
        """Process a single text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Split into segments
            segments = self.split_into_segments(cleaned_text)
            
            return segments
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def process_all_files(self):
        """Process all text files in the input directory."""
        # Get all text files
        text_files = list(self.input_dir.glob('*.txt'))
        
        if not text_files:
            raise ValueError(f"No .txt files found in {self.input_dir}")
        
        all_segments = []
        
        # Process each file
        for file_path in tqdm(text_files, desc="Processing files"):
            segments = self.process_file(file_path)
            all_segments.extend(segments)
        
        # Shuffle segments
        random.shuffle(all_segments)
        
        # Split into train/val/test
        n_segments = len(all_segments)
        n_train = int(0.8 * n_segments)
        n_val = int(0.1 * n_segments)
        
        train_segments = all_segments[:n_train]
        val_segments = all_segments[n_train:n_train + n_val]
        test_segments = all_segments[n_train + n_val:]
        
        # Save to jsonl files
        self._save_segments(train_segments, 'train.jsonl')
        self._save_segments(val_segments, 'val.jsonl')
        self._save_segments(test_segments, 'test.jsonl')
        
        print(f"\nProcessed {len(text_files)} files into:")
        print(f"- {len(train_segments)} training segments")
        print(f"- {len(val_segments)} validation segments")
        print(f"- {len(test_segments)} test segments")

    def _save_segments(self, segments, filename):
        """Save segments to a JSONL file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                json_line = json.dumps({'text': segment})
                f.write(json_line + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process text files for language model training')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing raw text files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Directory to save processed files')
    parser.add_argument('--min_length', type=int, default=100,
                      help='Minimum length of text segments')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum length of text segments')
    
    args = parser.parse_args()
    
    processor = TextProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    processor.process_all_files()

if __name__ == "__main__":
    main()


## Features

- Multiple model architectures:
  - Transformer
  - LSTM
  - RNN
- BPE (Byte-Pair Encoding) tokenization
- Training with early stopping
- Learning rate scheduling with warmup
- Model evaluation (perplexity and BLEU score)
- Interactive text generation
- Training progress visualization
- Checkpoint saving and loading

## Requirements

```bash
pip install torch tqdm matplotlib seaborn nltk sentencepiece pandas
```

## Running the Training Shell Script


## Data Preparation Pipeline

Before training the models, you need to prepare the data. Follow these steps in order:

### 1. Download Books (download_books.py)

First, run the script to download books from Project Gutenberg:

```bash
python download_books.py \
    --output_dir data/raw \
    --num_books 100 \
    --min_length 10000 \
    --max_length 100000
```

Arguments:
- `--output_dir`: Directory to save downloaded books
- `--num_books`: Number of books to download
- `--min_length`: Minimum book length (in characters)
- `--max_length`: Maximum book length (in characters)
- `--start_index`: (optional) Starting index for book IDs
- `--timeout`: (optional) Download timeout in seconds

The script will:
1. Create the output directory if it doesn't exist
2. Download books from Project Gutenberg
3. Clean and preprocess the raw text
4. Save books as individual text files
5. Generate a metadata.json file with book information

### 2. Process Data (data_processor.py)

After downloading the books, process them into the format required for training:

```bash
python data_processor.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --train_split 0.8 \
    --val_split 0.1 \
    --test_split 0.1 \
    --chunk_size 512 \
    --stride 256
```

Arguments:
- `--input_dir`: Directory containing raw book files
- `--output_dir`: Directory to save processed data
- `--train_split`: Proportion of data for training
- `--val_split`: Proportion of data for validation
- `--test_split`: Proportion of data for testing
- `--chunk_size`: Size of text chunks
- `--stride`: Stride length for text chunking
- `--min_length`: (optional) Minimum chunk length
- `--workers`: (optional) Number of worker processes

The script will:
1. Read all text files from the input directory
2. Clean and normalize the text
3. Split text into chunks with specified size and stride
4. Split data into train/validation/test sets
5. Save processed data as JSONL files
6. Create train.jsonl, val.jsonl, and test.jsonl

### Data Processing Pipeline Output

After running both scripts, you should have the following directory structure:


The project includes a shell script (`train_models.sh`) that can train all model types sequentially. To use it:

1. Make the script executable:
```bash
chmod +x train_models.sh
```

2. Run the script:
```bash
./train_models.sh
```

The shell script will train the following models with their respective configurations:

### Transformer Model
```bash
python train.py \
    --model_type transformer \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --nhead 4 \
    --dropout 0.1 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 5e-3 \
    --max_seq_length 512
```

### LSTM Model
```bash
python train.py \
    --model_type lstm \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --dropout 0.1 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 5e-3 \
    --max_seq_length 512
```

### RNN Model
```bash
python train.py \
    --model_type rnn \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --dropout 0.1 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 5e-3 \
    --max_seq_length 512
```

## Text Generation

### Using the Generation Script

After training, you can generate text using any of the trained models. The generation script provides an interactive interface for text generation.

1. For Transformer Model:
```bash
python generate.py \
    --model_type transformer \
    --model_path data/processed/checkpoints/best_transformer_model.pt \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --num_layers 4 \
    --nhead 4 \
    --max_seq_length 512
```

2. For LSTM Model:
```bash
python generate.py \
    --model_type lstm \
    --model_path data/processed/checkpoints/best_lstm_model.pt \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --max_seq_length 512
```

3. For RNN Model:
```bash
python generate.py \
    --model_type rnn \
    --model_path data/processed/checkpoints/best_rnn_model.pt \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --max_seq_length 512
```

### Interactive Generation Interface

After running the generation script, you'll enter an interactive mode where you can:

1. Enter prompts for text generation
2. Customize generation parameters:
   ```
   Enter your prompt: Once upon a time
   Temperature (0.1-1.0, default: 0.7): 0.8
   Max length (default: 100): 150
   Number of sequences (default: 1): 2
   ```

3. Generation Parameters:
   - `temperature`: Controls randomness (0.1-1.0)
     - Lower values (0.1-0.5): More focused, deterministic output
     - Higher values (0.6-1.0): More creative, diverse output
   - `max_length`: Maximum number of tokens to generate
   - `num_sequences`: Number of different sequences to generate

4. Special Commands:
   - Type 'help' to see parameter information
   - Type 'quit' to exit the generator
   - Press Ctrl+C to interrupt generation

### Generation Tips

1. Temperature Selection:
   - Use 0.3-0.5 for more coherent, focused text
   - Use 0.7-0.9 for more creative, diverse output
   - Start with 0.7 and adjust based on results

2. Prompt Engineering:
   - Provide clear, specific prompts
   - Include relevant context
   - Use proper punctuation and formatting

3. Length Considerations:
   - Shorter sequences (50-100 tokens) tend to be more coherent
   - Longer sequences may require lower temperatures
   - Consider the model's training sequence length

[Rest of the README remains the same...]

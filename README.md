
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

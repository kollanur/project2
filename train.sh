#!/bin/bash

# Train RNN
echo "Training RNN model..."
python train.py \
    --model_type rnn \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 4 \
    --dropout 0.2 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 3e-4 \
    --max_seq_length 512

# Train LSTM
echo "Training LSTM model..."
python train.py \
    --model_type lstm \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 4 \
    --dropout 0.2 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 5e-4 \
    --max_seq_length 512

# Train Transformer
echo "Training Transformer model..."
python train.py \
    --model_type transformer \
    --data_path data/processed \
    --tokenizer_path tokenizer/bpe_model.model \
    --embedding_dim 384 \
    --hidden_dim 512 \
    --num_layers 4 \
    --nhead 4 \
    --dropout 0 \
    --batch_size 128 \
    --epochs 30 \
    --learning_rate 5e-3 \
    --max_seq_length 512

# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
    
    def sample_token(self, logits, temperature=1.0):
        if temperature == 0:
            # Greedy sampling
            return torch.argmax(logits, dim=-1)
        else:
            # Temperature sampling
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1)

class RNNModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask=None, temperature=0):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.output_layer(output)
        
        if temperature >= 0:
            return self.sample_token(logits[:, -1, :], temperature)
        return logits

    def prompt(self, tokenizer, text, max_seq_length=100, temperature=0):
        self.eval()
        tokens = tokenizer.encode(text)
        device = next(self.parameters()).device
        
        for _ in range(max_seq_length):
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                next_token = self.forward(input_ids, temperature=temperature).item()
            tokens.append(next_token)
            if next_token == tokenizer.sp.eos_id():
                break
        
        return tokenizer.decode(tokens)

class LSTMModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask=None, temperature=0):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.output_layer(output)
        
        if temperature >= 0:
            return self.sample_token(logits[:, -1, :], temperature)
        return logits

    def prompt(self, tokenizer, text, max_seq_length=100, temperature=0):
        self.eval()
        tokens = tokenizer.encode(text)
        device = next(self.parameters()).device
        
        for _ in range(max_seq_length):
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                next_token = self.forward(input_ids, temperature=temperature).item()
            tokens.append(next_token)
            if next_token == tokenizer.sp.eos_id():
                break
        
        return tokenizer.decode(tokens)

class TransformerModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim=256, nhead=8, num_layers=4, 
                 dropout=0.1, max_seq_length=512):
        super().__init__(vocab_size, embedding_dim)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize position embeddings
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * 
                           (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_seq_length, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embedding.data = pe
    
    def forward(self, x, attention_mask=None, temperature=0):
        embedded = self.embedding(x)
        embedded = embedded + self.pos_embedding[:, :x.size(1)]
        embedded = self.dropout(embedded)
        
        if attention_mask is not None:
            padding_mask = attention_mask == 0
        else:
            padding_mask = None
        
        output = self.transformer(embedded, src_key_padding_mask=padding_mask)
        logits = self.output_layer(output)
        
        if temperature >= 0:
            return self.sample_token(logits[:, -1, :], temperature)
        return logits

    def prompt(self, tokenizer, text, max_seq_length=100, temperature=0):
        self.eval()
        tokens = tokenizer.encode(text)
        device = next(self.parameters()).device
        
        for _ in range(max_seq_length):
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                next_token = self.forward(input_ids, temperature=temperature).item()
            tokens.append(next_token)
            if next_token == tokenizer.sp.eos_id():
                break
        
        return tokenizer.decode(tokens)

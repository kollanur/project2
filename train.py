import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from nltk.translate.bleu_score import corpus_bleu
import nltk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

try:
    nltk.download('punkt')
except:
    pass

from models import RNNModel, LSTMModel, TransformerModel
from tokenizer import BPETokenizer
from dataset import TextDataset

class MetricsTracker:
    def __init__(self, model_type: str, plots_dir: Path):
        self.model_type = model_type
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.batch_losses = []
        
        # Initialize plot style
        plt.style.use('default')
        
        # Set default plot parameters for better visibility
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
            
    def update_batch(self, loss: float):
        self.batch_losses.append(loss)
        
    def update_epoch(self, train_loss: float, val_loss: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
            
    def plot_training_curves(self, epoch: int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, 'b-', label='Train Loss', marker='o')
        plt.plot(self.val_losses, 'r-', label='Validation Loss', marker='o')
        plt.title(f'{self.model_type} Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = self.plots_dir / f'{self.model_type}_training_curve_epoch_{epoch}_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot batch losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.batch_losses[-100:], 'b-')  # Plot last 100 batches
        plt.title(f'{self.model_type} Batch Losses (Last 100 batches)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        save_path = self.plots_dir / f'{self.model_type}_batch_losses_epoch_{epoch}_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_metrics(self, save_dir: Path):
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save as JSON
        with open(save_dir / f'{self.model_type}_training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=2, reduction='sum')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(input_ids, attention_mask, temperature=-1)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))

            num_tokens = (target_ids != 2).sum().item()
            total_tokens += num_tokens
            total_loss += loss.item()

    if total_tokens == 0:
        print("⚠️ Warning: No valid tokens for perplexity calculation.")
        return float('inf')

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()

def compute_bleu(model, tokenizer, test_data, device):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for text in tqdm(test_data, desc="Computing BLEU score"):
            try:
                ref_tokens = tokenizer.encode(text)
                references.append([ref_tokens])
                
                input_text = text[:50]
                hyp_tokens = tokenizer.encode(model.prompt(tokenizer, input_text))
                hypotheses.append(hyp_tokens)
            except Exception as e:
                print(f"Error processing text for BLEU score: {str(e)}")
                continue
    
    return corpus_bleu(references, hypotheses)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, metrics_tracker):
    model.train()
    total_loss = 0
    total_steps = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        try:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, temperature=-1)
            
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            metrics_tracker.update_batch(batch_loss)
            
            total_loss += batch_loss
            total_steps += 1
            
            progress_bar.set_postfix({
                'loss': f'{total_loss/total_steps:.4f}'
            })
            
        except Exception as e:
            print(f"Error in batch: {str(e)}")
            continue
    
    return total_loss / total_steps

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            try:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                output = model(input_ids, attention_mask, temperature=-1)
                loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                total_steps += 1
                
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
    return total_loss / total_steps

def train(args):
    # Setup directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(args.data_path)
    checkpoint_dir = base_dir / 'checkpoints'
    plots_dir = base_dir / 'plots'
    metrics_dir = base_dir / 'metrics'
    
    for dir_path in [checkpoint_dir, plots_dir, metrics_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.model_type, plots_dir)
    
    print(f"Using device: {device}")
    print(f"Saving checkpoints to: {checkpoint_dir}")
    print(f"Saving plots to: {plots_dir}")
    print(f"Saving metrics to: {metrics_dir}")
    
    # Initialize tokenizer and datasets
    print("Loading tokenizer...")
    tokenizer = BPETokenizer(args.tokenizer_path)
    
    print("Loading datasets...")
    train_dataset = TextDataset(f"{args.data_path}/train.jsonl", tokenizer, args.max_seq_length)
    val_dataset = TextDataset(f"{args.data_path}/val.jsonl", tokenizer, args.max_seq_length)
    test_dataset = TextDataset(f"{args.data_path}/test.jsonl", tokenizer, args.max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    print(f"Initializing {args.model_type.upper()} model...")
    vocab_size = tokenizer.vocab_size()
    model_classes = {
        'transformer': TransformerModel,
        'lstm': LSTMModel,
        'rnn': RNNModel
    }
    
    # Base configuration shared by all models
    base_config = {
        'vocab_size': vocab_size
    }
    
    # Model-specific configurations
    if args.model_type == 'transformer':
        model_config = {
            **base_config,
            'embedding_dim': args.embedding_dim,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'max_seq_length': args.max_seq_length
        }
    else:  # LSTM and RNN models
        model_config = {
            **base_config,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    
    print("Model configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
        
    model = model_classes[args.model_type](**model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    max_patience = 3
    best_metrics = {}
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, 
                               device, epoch + 1, metrics_tracker)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update metrics
        metrics_tracker.update_epoch(train_loss, val_loss)
        
        # Plot metrics
        metrics_tracker.plot_training_curves(epoch + 1)
        
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            
            # Compute metrics for best model
            test_perplexity = compute_perplexity(model, test_loader, device)
            test_bleu = compute_bleu(model, tokenizer, test_dataset.examples[:100], device)
            
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'test_perplexity': test_perplexity,
                'test_bleu': test_bleu
            }
            
            checkpoint_path = checkpoint_dir / f'best_{args.model_type}_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': best_metrics
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered")
                break
    
    # Save final metrics
    metrics_tracker.save_metrics(metrics_dir)
    
    # Save best model metrics
    with open(metrics_dir / f'{args.model_type}_best_metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    print("\nBest Model Metrics:")
    print(f"Epoch: {best_metrics['epoch']}")
    print(f"Validation Loss: {best_metrics['val_loss']:.4f}")
    print(f"Test Perplexity: {best_metrics['test_perplexity']:.2f}")
    print(f"Test BLEU Score: {best_metrics['test_bleu']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--model_type', type=str, default='transformer',
                      choices=['transformer', 'lstm', 'rnn'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_seq_length', type=int, default=512)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()

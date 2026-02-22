import torch
from src.models.transformer import Transformer
from src.data.dataset import get_tiny_shakespeare, create_dataloader
from src.training.trainer import train
import logging
import os

# Configure logging to also write to a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training.log"),
                        logging.StreamHandler()
                    ])

def main():
    # Hyperparameters
    vocab_size = 65 # For Tiny Shakespeare
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 512
    max_seq_len = 128
    dropout_p = 0.1
    batch_size = 32
    epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    text = get_tiny_shakespeare()
    dataloader = create_dataloader(text, batch_size=batch_size, seq_len=max_seq_len)
    logging.info("Data loaded and dataloader created.")

    # Initialize model
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout_p)
    logging.info("Transformer model initialized.")

    # Train the model
    logging.info("Starting training...")
    train(model, dataloader, epochs=epochs, device=device)
    logging.info("Training finished.")

    # Save a dummy model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/transformer_final.pth")
    logging.info("Dummy model checkpoint saved to checkpoints/transformer_final.pth")

if __name__ == "__main__":
    main()

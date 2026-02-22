import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os

class TextDataset(Dataset):
    def __init__(self, text, vocab_size=10000, seq_len=128):
        self.text = text
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        src = torch.tensor(self.data[idx:idx + self.seq_len])
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1])
        return src, target

def get_tiny_shakespeare(path="tiny_shakespeare.txt"):
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(path, "w") as f:
            f.write(response.text)
    with open(path, "r") as f:
        text = f.read()
    return text

def create_dataloader(text, batch_size=32, seq_len=128):
    dataset = TextDataset(text, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

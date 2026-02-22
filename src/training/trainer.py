import torch
import torch.nn as nn
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for src, target in tqdm(dataloader):
            src, target = src.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, None)
            loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, target in dataloader:
                src, target = src.to(self.device), target.to(self.device)
                output = self.model(src, None)
                loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)

def train(model, dataloader, epochs=10, device="cuda"):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, scheduler, criterion, device)
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        logging.info(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

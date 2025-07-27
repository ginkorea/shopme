# backend/emotic/pipeline/trainer.py

import torch
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, save_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        self.criterion_emotions = torch.nn.BCEWithLogitsLoss()
        self.criterion_vad = torch.nn.MSELoss()

        os.makedirs(save_dir, exist_ok=True)

    def _prepare_batch(self, batch):
        """
        Move batch to device and extract inputs and targets.
        """
        img = batch['img'].to(self.device)
        y_emotions = batch['emotions'].to(self.device).float()
        y_vad = batch['vad'].to(self.device).float()
        return img, y_emotions, y_vad

    def _compute_loss(self, output, y_emotions, y_vad):
        """
        Compute multi-task loss.
        """
        loss_emotions = self.criterion_emotions(output['emotions'], y_emotions)
        loss_vad = self.criterion_vad(output['vad'], y_vad)
        return loss_emotions + 0.5 * loss_vad

    def train(self, num_epochs=10, save_every=1):
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                img, y_emotions, y_vad = self._prepare_batch(batch)
                output = self.model(img)
                loss = self._compute_loss(output, y_emotions, y_vad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.4f}")

            if self.val_loader:
                self.validate(epoch)

            if save_every and (epoch % save_every == 0):
                self.save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                img, y_emotions, y_vad = self._prepare_batch(batch)
                output = self.model(img)
                loss = self._compute_loss(output, y_emotions, y_vad)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_loss:.4f}")

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"✅ Checkpoint saved: {path}")


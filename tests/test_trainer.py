# tests/test_trainer.py

import torch
from torch.utils.data import DataLoader, Dataset
from backend.emotic.model import EmoticNeuroNet
from backend.emotic.pipeline import Trainer


# 🔹 Dummy dataset for testing
class DummyEmoticDataset(Dataset):
    def __len__(self):
        return 10  # Small number for fast test

    def __getitem__(self, idx):
        return {
            'img': torch.randn(3, 224, 224),
            'emotions': torch.randint(0, 2, (26,)).float(),
            'vad': torch.randn(3)
        }


def test_trainer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize dummy data loaders
    train_loader = DataLoader(DummyEmoticDataset(), batch_size=2)
    val_loader = DataLoader(DummyEmoticDataset(), batch_size=2)

    # Model and optimizer
    model = EmoticNeuroNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Wrap trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir="checkpoints_test"
    )

    # Patch save method for safety
    def dummy_save(path, optimizer, epoch):
        print(f"✅ Dummy checkpoint triggered at epoch {epoch}: {path}")

    model.save = dummy_save

    trainer.train(num_epochs=1)


if __name__ == "__main__":
    test_trainer()

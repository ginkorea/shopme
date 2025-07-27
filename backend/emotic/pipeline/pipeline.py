# backend/emotic/pipeline/pipeline.py

import torch
from torch.utils.data import DataLoader

from backend.emotic.pipeline.dataset import EmoticDataset
from backend.emotic.pipeline.inference import Inference
from backend.emotic.pipeline.trainer import Trainer
from backend.emotic.model import EmoticNeuroNet

class EmoticPipeline:
    def __init__(
        self,
        csv_paths,
        data_root,
        batch_size=32,
        lr=1e-4,
        device=None,
        num_epochs=10,
        save_dir="checkpoints"
    ):
        """
        Args:
            csv_paths (dict): Keys: 'train', 'val' (optional: 'test'). Values: path to CSV files.
            data_root (str): Path to EMOTIC data root.
            batch_size (int): Batch size for DataLoaders.
            lr (float): Learning rate.
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
            num_epochs (int): Number of epochs.
            save_dir (str): Directory to save checkpoints.
        """
        self.csv_paths = csv_paths
        self.data_root = data_root
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.save_dir = save_dir

        self._prepare()

    def _prepare(self):
        # Load datasets
        train_dataset = EmoticDataset(self.csv_paths['train'], self.data_root)
        val_dataset = EmoticDataset(self.csv_paths['val'], self.data_root)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.model = EmoticNeuroNet()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            device=self.device,
            save_dir=self.save_dir
        )

    def run(self):
        self.trainer.train(num_epochs=self.num_epochs)

    def load_and_eval(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Basic val eval
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['img'].to(self.device)
                y_emotions = batch['emotions'].to(self.device)
                y_vad = batch['vad'].to(self.device)
                predictions = self.model(images)
                print(predictions)  # Replace with metric calc later
                break

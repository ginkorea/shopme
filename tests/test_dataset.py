# tests/test_dataset.py

from backend.emotic.pipeline.dataset import EmoticDataset
from torch.utils.data import DataLoader
import os

def test_dataset():
    # Automatically resolve project root relative to this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "backend", "emotic", "data", "annots_arrs", "annot_arrs_train.csv")
    data_root = os.path.join(project_root, "backend", "emotic", "data")

    dataset = EmoticDataset(csv_path=csv_path, data_root=data_root, crop_type="crop")

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]

    print("Sample keys:", sample.keys())
    print("Image shape:", sample['img'].shape)
    print("Emotions shape:", sample['emotions'].shape)
    print("VAD:", sample['vad'])

    assert 'img' in sample
    assert 'emotions' in sample
    assert 'vad' in sample
    assert sample['img'].ndim == 3  # Expecting HWC or CHW
    assert sample['emotions'].shape[0] == 26
    assert sample['vad'].shape[0] == 3
    print("✅ Dataset test passed.")

if __name__ == "__main__":
    test_dataset()

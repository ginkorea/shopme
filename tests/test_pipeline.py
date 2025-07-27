# tests/test_pipeline.py

import os
import pandas as pd
import numpy as np

from backend.emotic.pipeline import EmoticPipeline

def generate_dummy_csv(csv_path, data_root, num_samples=10):
    """
    Generates a dummy CSV and matching .npy files for pipeline testing.
    """
    os.makedirs(os.path.join(data_root, "img_arrs"), exist_ok=True)
    rows = []

    for i in range(num_samples):
        # Create dummy image file
        img = np.random.randn(224, 224, 3).astype(np.float32)
        img_file = f"img_arr_{i}.npy"
        np.save(os.path.join(data_root, "img_arrs", img_file), img)

        # Construct dummy row
        row = {
            'Filename': f"img_{i}.jpg",
            'Width': 224, 'Height': 224,
            'Age': 'Adult', 'Gender': 'Male',
            'Valence': 5, 'Arousal': 5, 'Dominance': 5,
            'Crop_name': img_file,
            'Arr_name': img_file,
        }
        # Add 26 emotion fields
        for emotion in [
            'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement', 'Confidence',
            'Happiness', 'Pleasure', 'Excitement', 'Surprise', 'Sympathy', 'Doubt/Confusion',
            'Disconnection', 'Fatigue', 'Embarrassment', 'Yearning', 'Disapproval', 'Aversion',
            'Annoyance', 'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear',
            'Pain', 'Suffering'
        ]:
            row[emotion] = np.random.randint(0, 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def test_pipeline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(project_root, "backend", "emotic", "data", "dummy")
    os.makedirs(data_root, exist_ok=True)

    train_csv = os.path.join(data_root, "train.csv")
    val_csv = os.path.join(data_root, "val.csv")

    # Generate dummy data
    generate_dummy_csv(train_csv, data_root)
    generate_dummy_csv(val_csv, data_root)

    csv_paths = {'train': train_csv, 'val': val_csv}

    pipeline = EmoticPipeline(
        csv_paths=csv_paths,
        data_root=data_root,
        batch_size=2,
        lr=1e-4,
        num_epochs=1,
        save_dir=os.path.join(project_root, "checkpoints_dummy")
    )

    pipeline.run()

    # Find the latest checkpoint and test load_and_eval
    ckpt_path = os.path.join(project_root, "checkpoints_dummy", "emotic_epoch1.pt")
    if os.path.exists(ckpt_path):
        pipeline.load_and_eval(ckpt_path)

if __name__ == "__main__":
    test_pipeline()

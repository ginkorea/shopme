import torch
from backend.emotic.model import EmoticNeuroNet

def test_emotic_model():
    model = EmoticNeuroNet()
    model.eval()

    # Dummy batch of RGB images [B, C, H, W]
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4

    with torch.no_grad():
        output = model(dummy_input)

    print("✅ Forward pass output keys:", output.keys())
    assert 'emotions' in output
    assert 'emotions_prob' in output
    assert 'vad' in output

    assert output['emotions'].shape == (4, 26)
    assert output['emotions_prob'].shape == (4, 26)
    assert output['vad'].shape == (4, 3)
    print("✅ Output shapes are correct.")

    # Test infer method
    results = model.infer(dummy_input)
    print("Infer output:", results)
    assert isinstance(results['emotions'], list)
    assert isinstance(results['vad'], list)
    assert len(results['emotions']) == 4
    assert len(results['vad']) == 4
    print("✅ Inference method works.")

if __name__ == "__main__":
    test_emotic_model()

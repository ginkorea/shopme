import torch
import torch.nn.functional as F
import json


class Inference:
    def __init__(self, model, device='cpu', emotion_labels=None):
        """
        Args:
            model (EmoticNeuroNet): Trained model
            device (str or torch.device): Device to use
            emotion_labels (List[str], optional): Human-readable emotion labels
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.labels = emotion_labels or [f"emo_{i}" for i in range(model.num_emotions)]

    def run(self, x: torch.Tensor, threshold: float = 0.5, return_probs: bool = False):
        """
        Perform inference on input tensor.

        Args:
            x (Tensor): Input tensor (B, 3, 224, 224)
            threshold (float): Classification threshold for emotion prediction
            return_probs (bool): Whether to return raw probabilities too

        Returns:
            List[dict]: One dict per input with emotion labels and VAD values
        """
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x, detach_emotion_for_vad=True)
            probs = F.sigmoid(output['emotions'])

            results = []
            for i in range(x.size(0)):
                emotion_vector = (probs[i] > threshold).int().tolist()
                emotion_names = [self.labels[j] for j, active in enumerate(emotion_vector) if active]

                sample_result = {
                    "emotions": emotion_names,
                    "vad": [round(v, 3) for v in output['vad'][i].tolist()]
                }

                if return_probs:
                    sample_result["emotion_probs"] = [round(p, 4) for p in probs[i].tolist()]

                results.append(sample_result)

        return results


    @staticmethod
    def save_results(results, path="inference_results.json"):
        """
        Save results to a JSON file.

        Args:
            results (List[dict]): Inference outputs
            path (str): Output file path
        """
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📝 Saved results to {path}")

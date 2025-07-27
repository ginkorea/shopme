import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmoticNeuroNet(nn.Module):
    """
    Multi-task neural network for emotion recognition and valence/arousal/dominance (VAD) regression.

    Components:
    - EfficientNet-B0 backbone (pretrained)
    - Shared FC layer
    - Emotion head (multi-label classification, 26 outputs)
    - VAD head (regression, 3 outputs)
    - Emotion logits feed into VAD head for contextual dependency

    Args:
        num_emotions (int): Number of emotion classes (default: 26)
        unfreeze_from (int): EfficientNet block index to unfreeze (0–6). Default: 6 (last block)
        dropout_p (float): Dropout probability in FC layers
    """
    def __init__(self, num_emotions=26, unfreeze_from=6, dropout_p=0.3):
        super().__init__()
        self.num_emotions = num_emotions

        # Load EfficientNet-B0 backbone
        base_model = models.efficientnet_b0(pretrained=True)
        self.backbone = base_model.features

        # Freeze all blocks initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze from specified block onward
        for idx, block in enumerate(self.backbone):
            if idx >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # Shared representation layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        # Emotion head: multi-label classification
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_emotions)
        )

        # VAD head: regression, uses shared features + emotion logits
        self.vad_head = nn.Sequential(
            nn.Linear(512 + num_emotions, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, detach_emotion_for_vad: bool = False):
        """
        Forward pass.

        Args:
            x (Tensor): Input image tensor (B, 3, 224, 224)
            detach_emotion_for_vad (bool): Whether to detach emotion logits before VAD head
                                           to prevent backpropagation through emotion head

        Returns:
            dict: {
                'emotions': raw logits (B, num_emotions),
                'emotions_prob': sigmoid output (B, num_emotions),
                'vad': continuous predictions (B, 3)
            }
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        shared = self.shared_fc(x)

        emotion_logits = self.emotion_head(shared)
        emotion_input = emotion_logits.detach() if detach_emotion_for_vad else emotion_logits

        vad_input = torch.cat([shared, emotion_input], dim=1)
        vad_output = self.vad_head(vad_input)

        return {
            'emotions': emotion_logits,
            'emotions_prob': torch.sigmoid(emotion_logits),
            'vad': vad_output
        }

    def freeze_backbone(self):
        """
        Freeze all EfficientNet backbone layers.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreeze all layers in the model (backbone + heads).
        """
        for param in self.parameters():
            param.requires_grad = True

    def infer(self, x, threshold: float = 0.5, detach_emotion_for_vad: bool = True):
        """
        Inference method for single or batched input.

        Args:
            x (Tensor): Input tensor (B, 3, 224, 224)
            threshold (float): Threshold for multi-label emotion classification
            detach_emotion_for_vad (bool): Prevent gradients flowing into VAD head

        Returns:
            dict:
                - emotions (List[List[int]]): Binary list of active emotion class indices
                - vad (List[List[float]]): Predicted VAD scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, detach_emotion_for_vad=detach_emotion_for_vad)

            probs = F.sigmoid(output['emotions'])  # shape: (B, 26)
            emotions = (probs > threshold).int().tolist()
            vad = output['vad'].tolist()

            return {
                'emotions': emotions,
                'vad': vad
            }


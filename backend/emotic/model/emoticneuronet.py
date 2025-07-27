import torch
import torch.nn as nn
from torchvision import models

class EmoticNeuroNet(nn.Module):
    def __init__(self, num_emotions=26, unfreeze_from=6, dropout_p=0.3):
        super().__init__()

        # Backbone
        base_model = models.efficientnet_b0(pretrained=True)
        self.backbone = base_model.features
        for param in self.pbackbone.parameters():
            param.requires_grad = False
        for idx, block in enumerate(self.backbone):
            if idx >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # Shared
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_emotions)
        )

        # VAD head: emotion logits + shared representation
        self.vad_head = nn.Sequential(
            nn.Linear(512 + num_emotions, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        shared = self.shared_fc(x)

        emotion_logits = self.emotion_head(shared)

        # Concatenate emotion logits with shared features
        vad_input = torch.cat([shared, emotion_logits], dim=1)
        vad_output = self.vad_head(vad_input)

        return {
            'emotions': emotion_logits,
            'vad': vad_output
        }

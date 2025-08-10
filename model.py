import torch
import torch.nn as nn
import timm

class CategoricalCNN(nn.Module):
    """Model architecture from mobilenet_v2_l.ipynb"""
    def __init__(self, num_classes, num_categories=3, embedding_dim=16, backbone_name='mobilenet_edgetpu_v2_l'):
        super().__init__()
        self.num_classes = num_classes
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            in_chans=embedding_dim
        )
        num_features = self.backbone.feature_info.channels(-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.long()
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.permute(0, 3, 1, 2)
        features = self.backbone(embedded_x)
        last_feature_map = features[-1]
        pooled_features = self.global_pool(last_feature_map).flatten(1)
        output = self.classifier(pooled_features)
        return output

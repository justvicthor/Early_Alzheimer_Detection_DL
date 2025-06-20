import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=3,
                 expansion=8,
                 feature_dim=1024,
                 nhid=512,
                 norm_type='Instance',
                 crop_size=96):
        super(ClassifierCNN, self).__init__()

        self.expansion   = expansion
        self.feature_dim = feature_dim
        self.nhid        = nhid
        self.num_classes = num_classes
        self.norm_type   = norm_type
        self.crop_size = crop_size

        # ---- Dynamically choosing normalization type --------------
        Norm3d = nn.InstanceNorm3d if norm_type.lower() == 'instance' else nn.BatchNorm3d

        # ---- Feature extractor ----------------------------------
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 4*expansion, kernel_size=1),
            Norm3d(4*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(4*expansion, 32*expansion, kernel_size=3, dilation=2),
            Norm3d(32*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2),
            Norm3d(64*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2),
            Norm3d(64*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(5, stride=2)
        )

        
        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, crop_size, crop_size, crop_size)  
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        
        self.fc6 = nn.Linear(flat_dim, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, nhid),
            nn.Linear(nhid, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.classifier(x)
        return x
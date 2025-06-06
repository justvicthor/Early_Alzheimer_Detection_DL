import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(ClassifierCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 15 * 18 * 15, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 


class ClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, expansion=8, feature_dim=1024, nhid=512):
        super(ClassifierCNN, self).__init__()
        self.expansion = expansion
        self.feature_dim = feature_dim
        self.nhid = nhid
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 4*expansion, kernel_size=1),
            nn.InstanceNorm3d(4*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(4*expansion, 32*expansion, kernel_size=3, dilation=2),
            nn.InstanceNorm3d(32*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2),
            nn.InstanceNorm3d(64*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2),
            nn.InstanceNorm3d(64*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=5, stride=2)
        )
        # Calculate output dimension of  conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 120, 144, 120) 
            out = self.conv(dummy)
            print('Conv output shape:', out.shape)
            flat_dim = out.view(1, -1).size(1)
            
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
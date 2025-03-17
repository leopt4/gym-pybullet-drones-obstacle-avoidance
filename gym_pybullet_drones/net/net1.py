import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ObstacleAvoidanceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim=1)

        self.extractors = nn.ModuleDict()
        total_concat_size = 0  # Total feature size after concatenation

        for key, subspace in observation_space.spaces.items():
            if key == "image1":
                # CNN for depth image processing
                self.extractors[key] = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, stride=1),  
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(8, 16, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(16, 25, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling → (batch, 25, 1, 1)
                )
                total_concat_size += 25  # CNN outputs (batch, 25, 1, 1)

            elif key == "states":
                # No FCN, just reshape from (batch, 8, 1) to (batch, 8, 1, 1)
                self.extractors[key] = nn.Identity()  # Keep state as is
                total_concat_size += 8  # State has 8 features (batch, 8, 1, 1)

        # Update the features_dim based on the extracted sizes
        self._features_dim = total_concat_size

        # Fully connected layers after concatenating extracted features
        # self.fc1 = nn.Linear(self._features_dim, 128)
        # self.fc2 = nn.Linear(128, features_dim)  # Output final features_dim (default: 128)

    def forward(self, observations) -> th.Tensor:
        encoded_tensors = []

        for key, extractor in self.extractors.items():
            encoded_tensors.append(extractor(observations[key]))

        # Flatten image tensor (if it's 4D)
        # if encoded_tensors[0].dim() == 4:  # Image features have shape (batch, 25, 1, 1)
        #     encoded_tensors[0] = encoded_tensors[0].view(encoded_tensors[0].size(0), -1)  # Flatten

        # Flatten state tensor (if it's 3D: batch, 1, 7)
            if encoded_tensors[0].dim() == 3:  # States are (batch, 1, 7)
                encoded_tensors[0] = encoded_tensors[0].view(encoded_tensors[0].size(0), -1)  # Flatten to (batch, 7)
        
        # for i in range(len(encoded_tensors)):
        #     print("len ", i, encoded_tensors[i].size())
        # Concatenate along channel dimension: (batch, 25, 1, 1) + (batch, 8, 1, 1) → (batch, 33, 1, 1)
        x = th.cat(encoded_tensors, dim=1)

        # Flatten before fully connected layers
        x = x.view(x.shape[0], -1)  # (batch, 33)

        # Fully connected layers
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # print("size x", x.size())
        return x

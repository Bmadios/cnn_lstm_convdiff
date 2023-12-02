import torch
import torch.nn as nn
import torch.nn.functional as F

# Étape 1: Définir l'architecture CNN
class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Ajustez la taille d'entrée ici pour correspondre à la sortie des couches convolutives
        self.fc1 = nn.Linear(20000, output_size)  # Ajustez cette taille en fonction de votre calcul

    def forward(self, x):
        # Votre logique de forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # Aplatir les caractéristiques pour le FC
        x = self.fc1(x)
        return x


# Étape 2: Intégrer le CNN avec votre LSTM
class CNNLSTM(nn.Module):
    def __init__(self, cnn, lstm):
        super(CNNLSTM, self).__init__()
        self.cnn = cnn
        self.lstm = lstm

    def forward(self, x):
        batch_size, _, timesteps, H, W = x.size()
        # Redimensionner pour traiter chaque pas de temps séparément dans le CNN
        c_in = x.view(batch_size * timesteps, 1, H, W)  # Assurez-vous que '1' est la dimension des canaux
        c_out = self.cnn(c_in)
        # Redimensionner la sortie pour le LSTM
        r_out = c_out.view(batch_size, timesteps, -1)
        lstm_out = self.lstm(r_out)
        return lstm_out


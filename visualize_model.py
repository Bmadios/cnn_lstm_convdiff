import torch
from torchviz import make_dot
from lstm_cnn import *
from model_lstm import *
from preprocess_data import *
import pandas as pd

# Dimensions de l'exemple d'entrée
batch_size = 1
timesteps = 10  # Nombre de pas de temps dans la séquence
height = 100
width = 100
input_channels = 1

filepath = "solution_u_implicit_data.csv"
data = pd.read_csv(filepath)

# Extrait les valeurs uniques de temps
unique_times = data['temps'].unique()

# Taille de la grille pour la transformation
grid_size = 100  # Ajustez selon vos besoins

def create_sequences(data, targets, sequence_length):
    X, Y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(targets[i+sequence_length])
    return np.array(X), np.array(Y)


sequence_length = 10

current_data = data[data['temps'] == 0.24]

inputs = current_data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
targets = current_data.iloc[:, -1].values  # Seulement la dernière colonne


X, u = create_sequences(inputs, targets, sequence_length)

print(X[:, :, 2])
u_grid = transform_to_2d(X[:, :, 1], X[:, :, 2], u, grid_size)


u_grid_tensor = torch.tensor(u_grid).unsqueeze(0).unsqueeze(0)  # Ajoute des dimensions pour batch et timesteps
# u_grid_tensor aura maintenant une forme de (1, 1, height, width)

# Créer un exemple d'entrée avec une dimension supplémentaire pour les timesteps
# example_input = torch.rand(batch_size, timesteps, input_channels, height, width)

cnn_output_size = 64  # Taille de sortie du CNN, qui sera la taille d'entrée du LSTM
hidden_dim = 96
layer_dim = 1
output_dim = 1

# Créer un exemple d'entrée
#example_input = torch.rand(batch_size, input_channels, height, width)

# Créer les modèles CNN et LSTM
cnn_model = CNN(input_channels, cnn_output_size)
lstm_model = LSTMModel(cnn_output_size, hidden_dim, layer_dim, output_dim)

# Créer le modèle CNN-LSTM
cnn_lstm_model = CNNLSTM(cnn_model, lstm_model)

# Obtenir la sortie du modèle
model_output = cnn_lstm_model(u_grid)

# Visualiser le modèle
graph = make_dot(model_output, params=dict(list(cnn_lstm_model.named_parameters())))
graph.render("cnn_lstm_model", format="png")  # Cela sauvegardera l'image en format PNG

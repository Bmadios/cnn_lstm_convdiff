from model_lstm import *
from lstm_cnn import *
from preprocess_data import *
from plotting import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import time

start_time = time.time()

filepath = "solution_u_implicit_data.csv"
data = pd.read_csv(filepath)

# Fonctions de prétraitement (tronquer les valeurs)
def truncate_two_decimals(x):
    return np.round(x, 2)

def truncate_three_decimals(x):
    return np.round(x, 3)

def is_multiple_of_10_minus_2(value, tolerance=1e-4):
    return abs(value * 100 - round(value * 100)) < tolerance

# Prétraitement des données
data["temps"] = data["temps"].apply(truncate_three_decimals)
data = data[data["temps"].apply(is_multiple_of_10_minus_2)]
data["x"] = data["x"].apply(truncate_two_decimals)
data["y"] = data["y"].apply(truncate_two_decimals)

# Séparation des entrées et des cibles
inputs = data.iloc[:, :-1].values
targets = data.iloc[:, -1].values

grid_size = 100
data_new = transform_all_sequences(inputs, targets, grid_size)

# Définir la longueur de la séquence temporelle
sequence_length = 5  # Par exemple, 5 pas de temps par séquence

# Créer des séquences temporelles
num_sequences = len(data_new) // sequence_length
data_sequences = np.array([data_new[i:i + sequence_length] for i in range(0, num_sequences * sequence_length, sequence_length)])

# Division des données en ensembles d'entraînement et de test
X_train, X_test = train_test_split(data_sequences, test_size=0.6, random_state=42)

# Conversion en tensors PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Ajouter une dimension de timesteps
X_train_tensor = X_train_tensor.unsqueeze(1)  # Ajoute une dimension de timesteps
X_test_tensor = X_test_tensor.unsqueeze(1)

# Vérifiez la forme des données
#print("Forme de X_train_tensor:", X_train_tensor.shape)
#print("Forme de X_test_tensor:", X_test_tensor.shape)

# Définir les dimensions pour le modèle CNN-LSTM
input_channels = 1 # Ajustez en fonction de vos données
cnn_output_size = 64  # Taille de sortie du CNN, qui sera la taille d'entrée du LSTM
hidden_dim = 96
layer_dim = 1
output_dim = 5*100*100

learning_rate = 0.001
epochs = 50000

# Créer les modèles CNN et LSTM
cnn_model = CNN(input_channels, cnn_output_size)
lstm_model = LSTMModel(cnn_output_size, hidden_dim, layer_dim, output_dim)

# Créer le modèle CNN-LSTM
cnn_lstm_model = CNNLSTM(cnn_model, lstm_model)

# Afficher le modèle
#print(cnn_lstm_model)

# Configuration de l'entraînement
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(epochs):
    cnn_lstm_model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = cnn_lstm_model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)  # Utilisation de X_train_tensor comme cible

    # Backward pass et optimisation
    loss.backward()
    optimizer.step()

    # Évaluation sur l'ensemble de test (utilisé ici comme validation)
    cnn_lstm_model.eval()
    val_outputs = cnn_lstm_model(X_test_tensor)
    val_loss = criterion(val_outputs, X_test_tensor)

    if epoch % 2 == 0:
        print(f"Epoch {epoch}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# Enregistrement du modèle final
torch.save(cnn_lstm_model.state_dict(), 'cnn_lstm_model_path.pth')

# Post-traitement et évaluation
unique_times = data['temps'].unique()
max_errors = []





#X_tensor = torch.tensor(data_t, dtype=torch.float32).unsqueeze(1)  # Ajoutez une dimension de timesteps si nécessaire
with torch.no_grad():
    predicted_u = cnn_lstm_model(X_test_tensor).numpy().squeeze()

"""          
for i in range(len(predicted_u)):  # Parcourir chaque séquence
    for j in range(predicted_u.shape[1]):  # Parcourir chaque pas de temps dans la séquence
        grid = predicted_u[i, j]  # Sélectionner la grille 2D pour le pas de temps j dans la séquence i
        title = f"Visualization at Sequence {i}, Time Step {j}"
        filename = f"grid_visualization_seq{i}_timestep{j}.png"
        plot_2d_grid(grid, title=title, filename=filename)

"""
for i in range(len(predicted_u)):  # Parcourir chaque séquence
    last_time_step_grid = predicted_u[i, -1]  # Sélectionner la grille 2D du dernier pas de temps dans la séquence i
    title = f"Visualization at Sequence {i}, Last Time Step"
    filename = f"grid_visualization_seq{i}.png"
    plot_2d_grid(last_time_step_grid, title=title, filename=filename)


"""average_max_error = np.mean(max_errors)
FINAL_max_error = np.max(max_errors)

print(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}")
print(f"MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error}")

# Calculer le temps d'exécution
end_time = time.time()
execution_time = end_time - start_time

# Convertir en heures, minutes, secondes
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = execution_time % 60

# Écrire dans un fichier
with open("temps_execution_TC_01.txt", "w") as file:
    file.write(f"Temps d'exécution: {hours} heure(s), {minutes} minute(s) et {seconds:.2f} seconde(s)\n")
    file.write(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}, \n MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error} \n")
    """
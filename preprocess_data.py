import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def transform_to_2d(X, Y, u, grid_size):
    """
    Transforme les données X, Y, u en une grille 2D.

    :param X: Coordonnées X
    :param Y: Coordonnées Y
    :param u: Valeurs à ces coordonnées
    :param grid_size: Taille de la grille (nombre de points dans chaque dimension)
    :return: Une matrice 2D représentant les valeurs de u sur la grille
    """
    # Créer une grille régulière
    grid_x, grid_y = np.linspace(min(X), max(X), grid_size), np.linspace(min(Y), max(Y), grid_size)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpoler les valeurs u sur cette grille
    u_grid = griddata((X, Y), u, (grid_x, grid_y), method='cubic', fill_value=0)

    return u_grid


def plot_2d_grid(grid, title="2D Grid Visualization", filename="grid_visualization.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='jet', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()  # Ajoutez cette ligne pour fermer la figure
    #plt.show()

def transform_all_sequences(inputs, targets, grid_size):
    transformed_sequences = []
    unique_time_steps = np.unique(inputs[:, 0])  # Supposons que la colonne 0 est 'time_step'

    for time_step in unique_time_steps:
        # Filtrer les données pour le time_step actuel
        indices = inputs[:, 0] == time_step
        X_current = inputs[indices, 1]  # colonne x
        Y_current = inputs[indices, 2]  # colonne y
        u_current = targets[indices]    # colonne u

        # Transformer les données en grille 2D pour ce time_step
        u_grid = transform_to_2d(X_current, Y_current, u_current, grid_size)
        transformed_sequences.append(u_grid)

    return np.array(transformed_sequences)




filepath = "solution_u_implicit_data.csv"
data = pd.read_csv(filepath)

# Extrait les valeurs uniques de temps
unique_times = data['temps'].unique()

# Taille de la grille pour la transformation
grid_size = 100  # Ajustez selon vos besoins


current_data = data[data['temps'] == 0.24]
X = current_data['x'].values
Y = current_data['y'].values
u = current_data['u'].values
u_grid = transform_to_2d(X, Y, u, grid_size)
#print(u_grid)


"""for t in unique_times:
    # Sélectionnez les données pour le temps courant
    current_data = data[data['temps'] == t]
    X = current_data['x'].values
    Y = current_data['y'].values
    u = current_data['u'].values

    # Transformez les données en grille 2D
    u_grid = transform_to_2d(X, Y, u, grid_size)

    # Visualisez la grille 2D
    plot_2d_grid(u_grid, title=f"Visualization at t={t}", filename=f"grid_visualization_t{t}.png")
    plt.close()  # Ajoutez cette ligne pour fermer la figure"""

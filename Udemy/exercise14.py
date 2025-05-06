import numpy as np
from sklearn.linear_model import LinearRegression


# 1. Clase Player
class Player:
    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
        self.name = name
        self.avg_session_time = avg_session_time
        self.avg_actions_per_min = avg_actions_per_min
        self.avg_kills_per_session = avg_kills_per_session
        self.victories = victories  # Puede ser None si es un jugador nuevo

    def to_features(self):
        return [self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session]


# 2. Clase PlayerDataset
class PlayerDataset:
    def __init__(self, players):
        # Filtrar solo jugadores con victorias definidas
        self.players = [p for p in players if p.victories is not None]

    def get_feature_matrix(self):
        return np.array([player.to_features() for player in self.players])  # Asegura que es un array 2D

    def get_target_vector(self):
        return np.array([player.victories for player in self.players])  # Vector 1D


# 3. Clase VictoryPredictor
class VictoryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False  # Bandera para saber si ya entrenamos

    def train(self, dataset: PlayerDataset):
        X = dataset.get_feature_matrix()
        y = dataset.get_target_vector()
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, player: Player):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Llama primero a train().")
        X_new = np.array(player.to_features()).reshape(1, -1)  # Correcto: matriz 2D (1 muestra, n features)
        return self.model.predict(X_new)[0]


# Uso de ejemplo
players = [
    Player("Alice", 40, 50, 6, 20),
    Player("Bob", 30, 35, 4, 10),
    Player("Charlie", 50, 60, 7, 25),
    Player("Diana", 20, 25, 2, 5),
    Player("Eve", 60, 70, 8, 30)
]

dataset = PlayerDataset(players)
predictor = VictoryPredictor()
predictor.train(dataset)

test_player = Player("TestPlayer", 45, 55, 5)
predicted = predictor.predict(test_player)
print(f"Victorias predichas para {test_player.name}: {predicted:.2f}")

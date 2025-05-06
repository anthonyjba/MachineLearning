'''
Tareas que debes realizar:

1. Crea una clase App que represente cada app con sus atributos.

2. Crea una clase RevenuePredictor que:

- Reciba una lista de objetos App.
- Extraiga las características relevantes para entrenar un modelo.
- Entrene un modelo de regresión lineal para predecir los ingresos (revenue).
- Permita predecir los ingresos de una nueva app con datos similares.

3. Entrena el modelo con los datos proporcionados (puedes usar una lista de ejemplo en el código).

4. Prueba el modelo prediciendo los ingresos estimados de una nueva app ficticia.
'''

import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Clase App
class App:
    def __init__(self, app_name, downloads, rating, size_mb, reviews, revenue=None):
        '''
        app_name: Nombre de la app
        downloads: Número de descargas (en miles)
        rating: Valoración media de los usuarios (de 1 a 5)
        size_mb: Tamaño de la app (en MB)
        reviews: Número de valoraciones escritas
        revenue: Ingresos generados (en miles de dólares) → variable a predecir
        '''
        self.app_name = app_name
        self.downloads = downloads
        self.rating = rating
        self.size_mb = size_mb
        self.reviews = reviews
        self.revenue = revenue

    def __str__(self):
        return (f'name: {self.app_name}, downloads: {self.downloads}, rating; {self.rating}, '
                f'size_mb: {self.size_mb}, reviews: {self.reviews}, revenue {self.revenue} ')
        
    def to_features(self):
        return [self.downloads, self.rating, self.size_mb, self.reviews]


#2 Clase RevenuePredictor
class RevenuePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False  # Bandera para saber si ya entrenamos

    def fit(self, dataset):
        X = np.array([app.to_features() for app in dataset])
        Y = np.array([app.revenue for app in dataset])
        self.model.fit(X, Y)
        self.is_trained = True

    def predict(self, app: App):
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Llama primero a train().")
        X_new = np.array(app.to_features()).reshape(1, -1)  # Correcto: matriz 2D (1 muestra, n features)
        return self.model.predict(X_new)[0]
		
		
'''
# Datos simulados de entrenamiento
training_apps = [
	App("TaskPro", 200, 4.2, 45.0, 1800, 120.0),
	App("MindSpark", 150, 4.5, 60.0, 2100, 135.0),
	App("WorkFlow", 300, 4.1, 55.0, 2500, 160.0),
	App("ZenTime", 120, 4.8, 40.0, 1700, 140.0),
	App("FocusApp", 180, 4.3, 52.0, 1900, 130.0),
	App("BoostApp", 220, 4.0, 48.0, 2300, 145.0),
]

predictor = RevenuePredictor()
predictor.fit(training_apps)

# Nueva app para predecir
new_app = App("FocusMaster", 250, 4.5, 50.0, 3000)
predicted_revenue = predictor.predict(new_app)
print(f"Ingresos estimados para {new_app.app_name}: ${predicted_revenue:.2f}K")
'''
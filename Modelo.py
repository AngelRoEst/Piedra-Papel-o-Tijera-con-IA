
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#Cargamos nuestro modelo entrenado
rf = joblib.load('modelo_entrenado.pkl')




def prediccionMovimiento(jugadas_humano, jugadas_ia, estado):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        "JugadasHumano": [jugadas_humano],
        "JugadasIA": [jugadas_ia],
        "Estado": [estado]
    })
    # Make a prediction using the trained model
    predicted_move = rf.predict(input_data)
    return predicted_move[0]


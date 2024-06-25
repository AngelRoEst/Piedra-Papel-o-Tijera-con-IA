# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def entrenarModelo():
    historialJugadas = pd.read_excel("DataFrameTestRPIA.xlsx")

    tiempoEstandar = '2024-06-23 16:25:00'
    tiempoEstandar = pd.to_datetime(tiempoEstandar)

    rf = RandomForestClassifier(n_estimators=25, min_samples_split=3, random_state=1)
    train = historialJugadas[historialJugadas["tiempo"] < tiempoEstandar]
    test = historialJugadas[historialJugadas["tiempo"] > tiempoEstandar]
    predictors = ["JugadasHumano", "JugadasIA", "Estado"]
    rf.fit(train[predictors], train["JugadasHumano"])

    # Save the trained model to a file
    joblib.dump(rf, 'modelo_entrenado.pkl')

if __name__ == "__main__":
    entrenarModelo()



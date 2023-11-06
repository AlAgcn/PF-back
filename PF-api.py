from flask import Flask, request, jsonify
import pickle
import pandas as pd

    # Cargar el modelo entrenado desde un archivo
with open('modelo_entrenado2.pkl', 'rb') as f:
    modelo_entrenado = pickle.load(f)

# Crear la aplicación Flask
app = Flask(__name__)

def acomodar_caracteristicas(datos_json):
    
    orden = ['P1_Q', 'P1_MC', 'P1_EA', 'P1_MT', 'P1_S', 'P1_EO', 'P1_I', 'P1_C', 
       'P2_C', 'P2_EO', 'P2_S', 'P2_MT', 'P2_Q', 'P2_MC', 'P2_I', 'P2_EA', 
       'P3_MT', 'P3_EO', 'P3_I', 'P3_C', 'P3_S', 'P3_EA', 'P3_Q', 'P3_MC', 
       'P4_MC', 'P4_MT', 'P4_EO', 'P4_I', 'P4_C', 'P4_Q', 'P4_S', 'P4_EA', 
       'P5_S', 'P5_EO', 'P5_C', 'P5_I', 'P5_MT', 'P5_MC', 'P5_Q', 'P5_EA']
    
    #print("Datos len: ", len(datos))
    #print("Caracteristicas len: ", len(caracteristicas))

    df = pd.DataFrame(datos_json, index=[0])

    df = df.reindex(columns=orden)    
    return df

# Definir la ruta de la API
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de entrada del usuario
    datos = request.get_json()  

    # Realizar la predicción utilizando el modelo de Random Forest
    prediction = modelo_entrenado.predict(acomodar_caracteristicas(datos))

    # Devolver la predicción al usuario
    output = {'prediction': str(prediction[0])}
    return jsonify(output), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)

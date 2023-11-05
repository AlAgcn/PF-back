import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#Para que se visualice correctamente
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#Archivo CSV a mostrar
ruta = r'A:\\Downloaded\PF-V1.4.csv'

datos = pd.read_csv(ruta)
datos = datos.drop(['Marca temporal', 'email', 'cursado'], axis=1)


print(datos.groupby(['ingenieria']).size())

#Eliminamos estudiantes que no están conformes con su carrera
datos = datos[datos['satisfaccion'] != 'No']
datos = datos.drop(['satisfaccion'], axis=1)

#Codificamos a Ordinal las columnas y escalamos las preguntas si o no
encoder = OrdinalEncoder()
reemplazos = {'Si': 1, 'No': 0, 'Más o menos': 0.5, 'Sí':1}

#col_enc = datos.columns[1:2]
col_sca = datos.columns[1:]
#datos[col_enc] = encoder.fit_transform(datos[col_enc])
datos[col_sca] = datos[col_sca].map(lambda x: reemplazos.get(x, x))


# Separe los datos en conjuntos de entrenamiento y prueba
col_obj = 'ingenieria'
col_caracts = datos.columns[1:]
x_train = datos.drop('ingenieria', axis=1)
y_train = datos['ingenieria']
#X_train, X_test, y_train, y_test = train_test_split(datos[col_caracts], datos[col_obj], test_size=0.2)

# Cree una instancia del clasificador Random Forest con los hiperparámetros sugeridos
clf = RandomForestClassifier(n_estimators=45, max_depth=8, random_state=101)

# Ajuste el modelo utilizando los datos de entrenamiento
clf.fit(x_train, y_train)

print(x_train.columns)
# Guardar el modelo entrenado en un archivo
with open('modelo_entrenado2.pkl', 'wb') as f:
    pickle.dump(clf, f)
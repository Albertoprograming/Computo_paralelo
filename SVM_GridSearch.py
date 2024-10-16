#  13/10/2024
#  OASM
#  Este programa realiza la busqueda de hiperparametros usando nivelacion de cargas e implementacion de procesos en hilos
import itertools
import multiprocess
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

n_cores = 3

def Nivelacion_de_Cargas(n_cores, lista_inicial):
    lista_final = []
    longitud_li = len(lista_inicial)
    carga = longitud_li // n_cores 
    salidas = longitud_li % n_cores
    contador = 0

    for i in range(n_cores):
        if i < salidas:
            carga2 = contador + carga + 1
        else:
            carga2 = contador + carga
        lista_final.append(lista_inicial[contador:carga2])
        contador = carga2
    return lista_final

# Definir parámetros
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 0.001, 0.01]
}


keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

def evaluate_set(hyperparameter_set, mejor_result, lock):

    df = pd.read_csv('Data_for_UCI_named.csv')

    df['stabf'] = df['stabf'].map({'unstable': 0, 'stable': 1})

    # Características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'])
        clf.fit(X_train, y_train)

        # Predecir con el conjunto de prueba
        y_pred = clf.predict(X_test)

        proce_accuracy = accuracy_score(y_test, y_pred)
        proce_recall = recall_score(y_test, y_pred)
        lock.acquire()
        print(f"Parámetros: {s}, Accuracy: {accuracy_score(y_test, y_pred):}")
        lock.release()
        # Bloqueo para actualizar el mejor resultado global
        lock.acquire()
        if proce_accuracy > mejor_result['accuracy']:
            mejor_result['accuracy'] = proce_accuracy
            mejor_result['recall'] = proce_recall
            mejor_result['params'] = s
        lock.release()

if __name__ == '__main__':
    threads = []
    N_THREADS = n_cores  
    splits = Nivelacion_de_Cargas(N_THREADS, combinations_svm)  
    lock = multiprocess.Lock()

    # Usar Manager para compartir el mejor resultado entre procesos
    with multiprocess.Manager() as manager:
        mejor_result = manager.dict({'accuracy': 0, 'recall': 0, 'params': None})

        start_time = time.perf_counter()
        # Cargar el dataset y mostrar las características (columnas)
        df = pd.read_csv('Data_for_UCI_named.csv')
        print(f"Características del dataset: {df.columns.tolist()}")
        # Crear y ejecutar los procesos
        for i in range(N_THREADS):
            threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], mejor_result, lock)))
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        finish_time = time.perf_counter()

      
        print(f"\nMejor accuracy es: {mejor_result['accuracy']}, con recall: {mejor_result['recall']} y parámetros: {mejor_result['params']}, número de nucleos: {n_cores}")
        print(f"\nProgram finished in {finish_time - start_time:.2f} seconds")

import numpy as np


#?Metodo que recibe el subconjunto del sistema candidato y me saca en una lista los elementos en t y en otra los elementos en t+1 sin repetidos
def obtenerElementosEnT(subconjuntoSistemaCandidato):
    elementosEnT = []
    elementosEnTmas1 = []

    for i in range(len(subconjuntoSistemaCandidato)):
        elementos = subconjuntoSistemaCandidato[i].split('-')
        if elementos[0] not in elementosEnT:
            elementosEnT.append(elementos[0])
        if elementos[1] not in elementosEnTmas1:
            elementosEnTmas1.append(elementos[1])

    return elementosEnT, elementosEnTmas1

#? Metodo que me recibe un vector de aristas, y crea una matriz llena de 1's, y luego recorre las aristas y pone 0's en la matriz
#? segun la arista que se recorre
def crearMatrizDeAdyacencia(aristas, subconjuntoSistemaCandidato):
    #Obtenemos los nodos 
    elementosEnT, elementosEnTmas1 = obtenerElementosEnT(subconjuntoSistemaCandidato)

    #? Creo una matriz de 1's
    matriz = np.ones((len(elementosEnT), len(elementosEnTmas1)), dtype=int)

    #? Recorro las aristas y voy poniendo 0's en la matriz
    for i in range(len(aristas)):
        elementos = aristas[i].split('-')
        matriz[elementosEnT.index(elementos[0])][elementosEnTmas1.index(elementos[1])] = 0
    return matriz


#? Metodo que recibe una matriz y me verifica si es bipartita

def esBipartita(matriz):
    matriz = np.array(matriz)

    # Verificar filas en ceros
    filas_ceros = np.sum(np.all(matriz == 0, axis=1))

    # Verificar columnas en ceros
    columnas_ceros = np.sum(np.all(matriz == 0, axis=0))

    # Determinar si hay una fila o una columna con ceros
    hay_ceros = filas_ceros == 1 or columnas_ceros == 1
    k = 0
    if(hay_ceros):
        k = 2

    if filas_ceros > 1 or columnas_ceros > 1 or (filas_ceros + columnas_ceros) > 1:
        hay_ceros = False
        k = (filas_ceros + columnas_ceros) + 1


    return {
        'esBipartita': hay_ceros,
        'filas_ceros': filas_ceros,
        'columnas_ceros': columnas_ceros,
        'k-particones':  k
    }




subconjuntoSistemaCandidato = np.array([
    'at-at+1', 'at-bt+1', 'at-ct+1', 'bt-at+1', 'bt-bt+1', 'bt-ct+1', 'ct-at+1', 'ct-bt+1', 'ct-ct+1'
])

aristas = ['at-at+1', 'bt-at+1', 'ct-at+1']
aristas2 = ['at-at+1', 'at-bt+1', 'at-ct+1', 'bt-at+1', 'bt-bt+1', 'bt-ct+1', 'ct-at+1']

matriz = (crearMatrizDeAdyacencia(aristas2, subconjuntoSistemaCandidato))
print(matriz)

print(esBipartita(matriz))
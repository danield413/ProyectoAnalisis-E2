import numpy as np
from scipy.stats import wasserstein_distance

def generarMatrizPresenteInicial(n):
    print(n)
    # Generar un array de números de 0 a 2^n - 1
    combinaciones = np.arange(2 ** n, dtype=np.uint32)
    
    # Crear una matriz para almacenar los bits
    binario_array = np.zeros((2 ** n, n), dtype=np.uint8)
    
    # Llenar la matriz bit a bit en formato little-endian
    for bit_pos in range(n):
        # Desplazar los números y obtener el bit en la posición correspondiente
        binario_array[:, bit_pos] = (combinaciones >> bit_pos) & 1
    
    return binario_array

def generarMatrizFuturoInicial(matriz):
    # Convertir la lista de listas a una matriz NumPy y luego transponerla
    return np.array(matriz).T


def elementosNoSistemaCandidato(estadoActualElementos, subconjuntoElementos):
    return [elemento for elemento in reversed(estadoActualElementos) if next(iter(elemento)) not in subconjuntoElementos]

#* Función que calcula el producto tensorial entre n vectores
def producto_tensorial_n(vectores: list[np.ndarray]) -> np.ndarray:
    
    if len(vectores) == 1:
        return vectores[0]
    
    if len(vectores) == 0:
        return np.array([0])
    
    resultado = vectores[0]
    for vector in vectores[1:]:
        resultado = np.kron(resultado, vector).flatten()
    
    return resultado

def producto_tensorial(a: np.ndarray ,b: np.ndarray):
    return np.kron(a,b).flatten()

#* Función para calcular la distancia EMD
def calcularEMD(a: np.ndarray, b: np.ndarray):
    emd_value = wasserstein_distance(a, b)
    return emd_value

def obtenerParticion(elementos):
    elementosT = [elem for elem in elementos if 't' in elem and 't+1' not in elem]
    elementosT1 = [elem for elem in elementos if 't+1' in elem]
    return (elementosT1, elementosT)

def generarCombinacionesEstadosIniciales(n):
    #* generar un arreglo con todas las combinaciones para n elementos en binario
    combinaciones = []
    for i in range(2**n):
        combinaciones.append(bin(i)[2:].zfill(n))
    return combinaciones

def encontrarParticionEquilibrioComplemento(particion1, subconjuntoElementos):
    elementos = subconjuntoElementos
    for x in range(len(elementos)):
        if elementos[x] == 't':
            elementos[x] = 't+1'
            
    faltantesT1 = []
    for i in elementos:
        if i+'+1' not in particion1[0]:
            faltantesT1.append(i+'+1')
            
    faltantesT = []
    for i in elementos:
        if i not in particion1[1]:
            faltantesT.append(i)
    
    p2 = (faltantesT1, faltantesT)

    return p2

def particionComplemento(particion1, subconjuntoSistemaCandidato):
    
    #* calcular elementos que hacen falta en t+1 basandose en el subconjunto del sistema candidato
    faltantesT1 = []
    faltantesT = []
    
    for i in subconjuntoSistemaCandidato:
        if 't+1' in i:
            if i not in particion1[0]:
                faltantesT1.append(i)
        
        if 't' in i and 't+1' not in i:
            if i not in particion1[1]:
                faltantesT.append(i)
        
    return (faltantesT1, faltantesT)
import copy
from utilidades.utils import producto_tensorial_n
import numpy as np

def encontrarVectorProbabilidades(particion, matricesPresentes, matricesFuturas, matricesTPM, estadoActualElementos, subconjuntoElementos, indicesElementosT, nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT):
    # print("******* ---- PARTICION:" , particion)
    
    vectorProbabilidades = []
    lista_nueva, lista_anterior = particion
    subDivisiones = [([elem], lista_anterior) for elem in lista_nueva]
  

    #? CASO 1: cuando el futuro es vacio
    #* Para estos casos es mas conveniente usar la matriz presente, la matriz futura y la tpm original sin partir

    if particion[0] == []:
        
        elementosPresente = particion[1] #* elementos en t
        valores = []
        
        #* la idea es marginalizar solo las columnas del presente que no están en la particion[1]
        elementosAMarginalizar = [elem for elem in subconjuntoElementos if elem not in elementosPresente]

        if elementosAMarginalizar == []:
            return [1]
        
        if elementosAMarginalizar != []:
            return [1]
        

    #? CASO 2: cuando el presente es vacio
    if particion[1] == []:
        #* uso la matriz presente, la matriz futura y la tpm original partidas
        valores = []
        elementosFuturo = particion[0]
        for elemento in elementosFuturo:
            #* selecciono la matriz futura y tpm correspondiente

            copiaMatricesTPM = copy.deepcopy(matricesTPM)

            matrizTpm = copiaMatricesTPM[elemento]

            #* sumar el valor de cada columna de la tpm
            matrizTpm = matrizTpm.T
            val = []
            for i in range(len(matrizTpm)):
                suma = sum(matrizTpm[i])
                val.append(suma)
            for i in range(len(val)):
                val[i] = val[i] / len(matrizTpm[0])

            valores.append(val)

        x = producto_tensorial_n(valores)
        return x

    #? CASO 3: cuando el futuro y el presente no son vacios
    for subDivision in subDivisiones:
        # print("         ---- SUBDIVISION:" , subDivision)
        #*Sacar cada elemento del lado izquierdo
        ladoIzquierdo = subDivision[0][0]

        matrizPresenteVector = matricesPresentes
        tpmVector = matricesTPM[ladoIzquierdo]
        
        # print("LA TPM QUE SE USARA ES: ", tpmVector)

        #*Si la longitud del lado derecho de la subdivision es menor que la longitud del subconjunto de elementos, hay que marginalizar por filas
        # print('-----------', ladoIzquierdo ,'-----------')
        # if len(subDivision[1]) < len(subconjuntoElementos):

        ordenColumnasPresente = subDivision[1]
        # print("ORDEN COLUMNAS PRESENTE: ", ordenColumnasPresente)
    
        #*Marginalizar por filas
        #*Crear un arreglo de indices con la longitud del subconjunto de elementos, desde 0 hasta la longitud del subconjunto de elementos
        indicesIniciales = np.arange(len(subconjuntoElementos))

        #*Crear un arreglo con los indices del lado derecho de la subdivision
        indicesPresente = [indicesIniciales[i] for i in range(len(indicesIniciales)) if subconjuntoElementos[i] in subDivision[1]]
        
        #*Hacer la diferencia entre los indices iniciales y los indices presente
        indicesMarginalizar = np.setdiff1d(indicesIniciales, indicesPresente)
        

        #*Eliminar esos indices de la matriz presente
        #*Transponemos la matriz para eliminar las filas con esos indices
        matrizPresenteVector = matrizPresenteVector.T
        #*Eliminar las columnas con esos indices
        matrizPresenteVector = np.delete(matrizPresenteVector, indicesMarginalizar, axis=0)

        # print("matrizPresenteVector", matrizPresenteVector)
        
        #*Transponer la matriz para dejarla como estaba

        #* identificar los grupos que se repiten en columnas
        arreglo = [[] for i in range(len(matrizPresenteVector[0]))]
        for fila in matrizPresenteVector:
            for idx, valor in enumerate(fila):
                arreglo[idx].append(valor)

        
        #* recorrer los grupos
        subarreglos_repetidos = {}

        for i, subarreglo in enumerate(arreglo):
            subarreglo_tuple = tuple(subarreglo)  # Convertir el subarreglo a tupla (para ser hashable)
            if subarreglo_tuple in subarreglos_repetidos:
                subarreglos_repetidos[subarreglo_tuple].append(i)
            else:
                subarreglos_repetidos[subarreglo_tuple] = [i]

        repetidos_con_indices = {k: v for k, v in subarreglos_repetidos.items() if len(v) > 1}

        for subarreglo, indices in repetidos_con_indices.items():
            menorIndice = min(indices)
            #* recorre [0,1,16]
            for i in indices:
                if i != menorIndice:
                    for k in range(len(tpmVector[i])):
                        tpmVector[menorIndice][k] += tpmVector[i][k]
                        tpmVector[i][k] = 99

                if i != menorIndice:
                    for k in matrizPresenteVector:
                        k[i] = 77

            numero_columnas = len(tpmVector[0])
            for k in range(numero_columnas):
                division = tpmVector[menorIndice][k] / len(indices)
                tpmVector[menorIndice][k] = division
                
        matrizPresenteVector = matrizPresenteVector.T

        filas_a_eliminar = []
        for i in range(len(matrizPresenteVector)):
            if 77 in matrizPresenteVector[i]:
                filas_a_eliminar.append(i)

        matrizPresenteVector = np.delete(matrizPresenteVector, filas_a_eliminar, axis=0)

        filas_a_eliminar = []
        for i in range(len(tpmVector)):
            if 99 in tpmVector[i]:
                filas_a_eliminar.append(i)

        tpmVector = np.delete(tpmVector, filas_a_eliminar, axis=0)

        estadosAcutales = []
        for i in estadoActualElementos:
            if list(i.keys())[0] in ordenColumnasPresente:
                estadosAcutales.append(list(i.values())[0])
        
        indiceVector = -1
        for i in range(len(matrizPresenteVector)):
            if matrizPresenteVector[i].tolist() == estadosAcutales:
                indiceVector = i
                break

        vectorProbabilidades.append(tpmVector[indiceVector])
            
   
    productoTensorialParticion = producto_tensorial_n(vectorProbabilidades)
    return productoTensorialParticion
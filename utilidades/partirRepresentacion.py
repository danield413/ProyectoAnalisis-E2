import copy
import numpy as np

def partirRepresentacion(nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT1, indicesElementosT):

    #* Matrices resultantes del proceso de representacion
    matricesPresentes = nuevaMatrizPresente
    matricesFuturas = dict()
    matricesTPM = dict()

    elementosT1Revisados = np.array([])
    
    for elementoT1 in elementosT1:
        # print("elementoT1", elementoT1)

        # print("COPIA: nuevaMatrizFuturo", nuevaMatrizFuturo)
        copiaMatrizFuturo = copy.deepcopy(nuevaMatrizFuturo)
        copiaTPM = copy.deepcopy(nuevaTPM)

        #* si el elemento futuro no se ha revisado
        if (elementoT1 not in elementosT1Revisados):
            elementosT1Revisados = np.append(elementosT1Revisados, elementoT1)
            
            #* buscar el indice del elemento (si es por ejm at+1, buscar at) en el estado actual
            tieneIndice = False
            for elemento in indicesElementosT:
                if elementoT1[:-2] in elemento:
                    tieneIndice = True
                    break
            if not tieneIndice:
                continue
            
            print(len(nuevaMatrizPresente))
            # print("indicesElementosT", indicesElementosT)
            # # print("len(copiaMatrizFuturo)", len(copiaMatrizFuturo))
            indice = indicesElementosT[elementoT1[:-2]+'+1']
            # print("elemento", elementoT1, "indice:",indice)
            #* borrar las filas de la matriz futuro excepto la fila indice
            
            
            copiaMatrizFuturo = np.delete(copiaMatrizFuturo, [i for i in range(len(copiaMatrizFuturo)) if i != indice], axis=0)
            

            #? proceso
            #* identificar los grupos que se repiten en columnas
            arreglo = [[] for i in range(len(copiaMatrizFuturo[0]))]
            for fila in copiaMatrizFuturo:
                for idx, valor in enumerate(fila):
                    arreglo[idx].append(valor)

            #* recorrer los grupos
            subarreglos_repetidos = {}

            # Iterar sobre el arreglo y buscar repetidos
            for i, subarreglo in enumerate(arreglo):
                subarreglo_tuple = tuple(subarreglo)  # Convertir el subarreglo a tupla (para ser hashable)
                if subarreglo_tuple in subarreglos_repetidos:
                    subarreglos_repetidos[subarreglo_tuple].append(i)
                else:
                    subarreglos_repetidos[subarreglo_tuple] = [i]

            # Filtrar solo los subarreglos que están repetidos (es decir, que tienen más de un índice)
            repetidos_con_indices = {k: v for k, v in subarreglos_repetidos.items() if len(v) > 1}

            # print("Repetidos")
            # print(repetidos_con_indices)

            for subarreglo, indices in repetidos_con_indices.items():
                menorIndice = min(indices)
                #* recorre [0,1,16]
                for i in indices:
                    #* i != 0
                    if i != menorIndice:
                        
                        #* si es menor recorro las fila de TPM
                        #* recorrer la tpm y sumar a la fila menorIndice la fila i
                        for k in copiaTPM:
                            k[menorIndice] += k[i]
                            #* el valor de la columna de la fila k
                            k[i] = 99

                        for k in copiaMatrizFuturo:
                            k[i] = 77

            #* Transponer la matriz para eliminar las columnas con 99
            copiaTPM = copiaTPM.T
            copiaMatrizFuturo = copiaMatrizFuturo.T

            #* Eliminar las columnas con 99
            filas_a_eliminar = []
            for i in range(len(copiaTPM)):
                if 99 in copiaTPM[i]:
                    filas_a_eliminar.append(i)

            copiaTPM = np.delete(copiaTPM, filas_a_eliminar, axis=0)

            #* Eliminar las columnas con 77
            filas_a_eliminar = []
            for i in range(len(copiaMatrizFuturo)):
                if 77 in copiaMatrizFuturo[i]:
                    filas_a_eliminar.append(i)

            copiaMatrizFuturo = np.delete(copiaMatrizFuturo, filas_a_eliminar, axis=0)

            #* Transponer la matriz para dejarla como estaba
            copiaTPM = copiaTPM.T
            copiaMatrizFuturo = copiaMatrizFuturo.T

            matricesFuturas[elementoT1] = copiaMatrizFuturo
            matricesTPM[elementoT1] = copiaTPM

    return matricesPresentes, matricesFuturas, matricesTPM
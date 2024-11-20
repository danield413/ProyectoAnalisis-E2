import numpy as np

def aplicarMarginalizacion(nuevaMatrizFuturo, nuevaTPM, elementosBackground, estadoActualElementos, nuevaMatrizPresente):

    indicesElementosEliminados = []

    # print("ELEMENTOS DE BACKGROUND", elementosBackground)
    
    for elemento in elementosBackground:
        llave = list(elemento.keys())[0]
        # print("Elemento", llave)
        for idx, elem in enumerate(estadoActualElementos):
            if llave in elem:
                indicesElementosEliminados.append({
                    llave+'+1': idx
                })
                break
        
    # print("Indices de elementos eliminados en marginalizacion", indicesElementosEliminados)
    
    #*Ahora recorro todos los elementos en t
    nuevosIndicesElementos = {}
    counter = 0
    for elemento in estadoActualElementos:
        llave = list(elemento.keys())[0]
        nombreElemento = llave+'+1'

        #* Si el elemento no está en los elementos eliminados
        #* el siguiente elemento tiene la posicion counter
        if nombreElemento not in [list(elem.keys())[0] for elem in indicesElementosEliminados]:
            nuevosIndicesElementos[nombreElemento] = counter
            counter += 1
        
    # print("Nuevos indices de los elementos en la matriz futuro", nuevosIndicesElementos)    

    if len(elementosBackground) > 0:

        indicesCondicionesBackGround = {list(elem.keys())[0]: idx for idx, elem in enumerate(estadoActualElementos)}

        for elemento in elementosBackground:
            
            llave = list(elemento.keys())[0]  

            indice = indicesCondicionesBackGround[llave]


            #* Eliminar la fila #indice de la matriz futuro
            nuevaMatrizFuturo = np.delete(nuevaMatrizFuturo, indice, axis=0)
            
            #* identificar los grupos que se repiten en columnas
            arreglo = [[] for i in range(len(nuevaMatrizFuturo[0]))]
            for fila in nuevaMatrizFuturo:
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

            # Mostrar los subarreglos repetidos y sus respectivos índices

            for subarreglo, indices in repetidos_con_indices.items():
                menorIndice = min(indices)
                #* recorre [0,1,16]
                for i in indices:
                    #* i != 0
                    if i != menorIndice:
                        
                        #* si es menor recorro las fila de TPM
                        #* recorrer la tpm y sumar a la fila menorIndice la fila i
                        for k in nuevaTPM:
                            k[menorIndice] += k[i]
                            #* el valor de la columna de la fila k
                            k[i] = 99

                        for k in nuevaMatrizFuturo:
                            k[i] = 77

            #* Transponer la matriz para eliminar las columnas con 99
            nuevaTPM = nuevaTPM.T
            nuevaMatrizFuturo = nuevaMatrizFuturo.T

            #* Eliminar las columnas con 99
            filas_a_eliminar = []
            for i in range(len(nuevaTPM)):
                if 99 in nuevaTPM[i]:
                    filas_a_eliminar.append(i)

            nuevaTPM = np.delete(nuevaTPM, filas_a_eliminar, axis=0)

            #* Eliminar las columnas con 77
            filas_a_eliminar = []
            for i in range(len(nuevaMatrizFuturo)):
                if 77 in nuevaMatrizFuturo[i]:
                    filas_a_eliminar.append(i)

            nuevaMatrizFuturo = np.delete(nuevaMatrizFuturo, filas_a_eliminar, axis=0)

            #* Transponer la matriz para dejarla como estaba
            nuevaTPM = nuevaTPM.T
            nuevaMatrizFuturo = nuevaMatrizFuturo.T


    return nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, nuevosIndicesElementos
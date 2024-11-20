import numpy as np

#? Parámetros
#? matrizPresente: matriz presente en t
#? matrizFuturo: matriz futuro en t+1
#? TPM: matriz de transición de probabilidad
#? elementosBackground: elementos del background {elemento: valor inicial}
def aplicarCondicionesBackground(nuevaMatrizPresente, nuevaTPM, elementosBackground, nuevaMatrizFuturo, estadoActualElementos):
    
    # print("elementosBackground", elementosBackground)
    
    if elementosBackground == []:
        return nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM
    
    if len(elementosBackground) > 0:

        #* Extraemos primero los indices con su respectivo elemento del estado actual
        #* Crear un diccionario que mapea los nombres a su índice
        indicesCondicionesBackGround = {list(elem.keys())[0]: idx for idx, elem in enumerate(estadoActualElementos)}
        #print(indicesCondicionesBackGround)

        for elemento in elementosBackground:

            llave = list(elemento.keys())[0]  

            indice = indicesCondicionesBackGround[llave]

            valorActualElemento = elemento[llave]
            
            #* Ya sabemos la posicion del elemento
            #* Ahora buscamos en la matriz presente y futura la fila y columna correspondiente
            #* Si el valor actual del elemento es 0 dejamos las filas que tengan 0
            #* Si el valor actual del elemento es 1 dejamos las filas que tengan 1
            for i in range(len(nuevaMatrizPresente)):
                if nuevaMatrizPresente[i][indice] == (1 if valorActualElemento == 0 else 0):
                    nuevaMatrizPresente[i].fill(99)
                    
            filas_a_eliminar = []
            for i in range(len(nuevaMatrizPresente)):
                if 99 in nuevaMatrizPresente[i]:
                    filas_a_eliminar.append(i)

            #* Ahora eliminamos las filas de la matriz presente usando los índices acumulados
            nuevaMatrizPresente = np.delete(nuevaMatrizPresente, filas_a_eliminar, axis=0)

            #* Ahora eliminamos las columnas de la matriz presente que estén en la posición indice
            nuevaMatrizPresente = np.delete(nuevaMatrizPresente, indice, axis=1)

            #* Ahora eliminamos las columnas de la matriz futura que estén en la posición indice
            nuevaTPM = np.delete(nuevaTPM, filas_a_eliminar, axis=0)

        return nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM
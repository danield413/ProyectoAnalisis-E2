import copy
import math
import numpy as np
from utilidades.utils import buscarValorUPrima, producto_tensorial_n, generarMatrizPresenteInicial

def obtenerVector(conjunto, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente, listaDeUPrimas):

    print("conjunto", conjunto)
    # print("partirMatricesPresentes", partirMatricesPresentes)

    # Diccionario donde se guardarán las relaciones
    relaciones = {}

    #* Crear diccionario de relaciones
    #* Contiene la matriz futuro que debe ser marginalizada y los elementos presentes que se deben marginalizar
    
    nuevoConjunto = []
    for arista in conjunto:
        if 'u' in arista:
            valor = buscarValorUPrima(listaDeUPrimas, arista)
            for val in valor:
                nuevoConjunto.append(val)
        else:
            nuevoConjunto.append(arista)
        
    
    for arista in nuevoConjunto:
        #* Separar presente y futuro
        
       
        x = arista.split('-')
        presente = x[0]
        futuro = x[1]
        
        #* Agregarlas
        if futuro not in relaciones:
            relaciones[futuro] = []
        relaciones[futuro].append(presente)
    
    #* ahora que tengo las relaciones sé en que matrices t+1 se marginalizan los elementos en t
    # print(relaciones)
    
    matricesResultado = []
    #* Se recorren las relaciones (es decir, las matrices a marginalizar)
    for matrizAMarginalizar in relaciones:
        #* Se toman sus elementos presentes a marginalizar
        elementosPresentesAMarginalizar = relaciones[matrizAMarginalizar]
        print("elementosPresentesAMarginalizar", elementosPresentesAMarginalizar)
        
        #*Cuando se marginalizan todas las columnas de una matriz
        if len(elementosPresentesAMarginalizar) == len(partirMatricesPresentes[0]):
            
            val = []
            
            tpmActual = copy.deepcopy(partirMatricesTPM[matrizAMarginalizar])
            
            tpmActual = tpmActual.T
            for i in range(len(tpmActual)):
                suma = sum(tpmActual[i])
                val.append(suma)
            for i in range(len(val)):
                val[i] = val[i] / len(tpmActual[0])

            
            matricesResultado.append({
                matrizAMarginalizar: [val]
            })
            continue
        
        print("ELEMENTOS T", elementosT)
        #* Se toman los elementos presentes que no se van a marginalizar
        elementosPresentesNOAMarginalizar = [elem for elem in elementosT if elem not in elementosPresentesAMarginalizar]
        
        #* Se toman las matrices a partir y se hace una copia de cada una
        mPresente = copy.deepcopy(partirMatricesPresentes)
        mFuturo = copy.deepcopy(partirMatricesFuturas[matrizAMarginalizar])
        tpmActual = copy.deepcopy(partirMatricesTPM[matrizAMarginalizar])
        
        #* Se toman los índices de los elementos presentes a marginalizar
        indicesElementosPresentesAMarginalizar = [indicesElementosT[elemento] for elemento in elementosPresentesAMarginalizar]
        indicesElementosPresentesNOAMarginalizar = [indicesElementosT[elemento] for elemento in elementosPresentesNOAMarginalizar]
        print("indicesElementosPresentesNOAMarginalizar", indicesElementosPresentesNOAMarginalizar)
        print("elementosPresentesAMarginalizar", elementosPresentesAMarginalizar)
        print("indicesElementosPresentesAMarginalizar", indicesElementosPresentesAMarginalizar)

        nuevosIndices = []
        count = 0
        for elem in elementosT:
            if elem in elementosPresentesAMarginalizar:
                nuevosIndices.append(count)
            count += 1
            
        # print("nuevosIndices", nuevosIndices) 
        
        #* Se borran las columnas de la matriz presente que se van a marginalizar
        mPresente = mPresente.T
        mPresente = np.delete(mPresente, nuevosIndices, axis=0)
        mPresente = mPresente.T
        
        # print("mPresente", mPresente)
        # print("tpmactual", tpmActual)
        
        #* Ya se eliminaron la(s) columna(s) de la matriz presente, ahora identificar los grupos que se repiten en filas
        
        grupos_filas = {}

        #* Recorrer la matriz para identificar las filas y sus índices
        for idx, fila in enumerate(mPresente):
            fila_tupla = tuple(fila)
            if fila_tupla in grupos_filas:
                grupos_filas[fila_tupla].append(idx)
            else:
                grupos_filas[fila_tupla] = [idx]

        #* Filtrar solo los grupos que se repiten (más de un índice)
        grupos_repetidos = {fila: indices for fila, indices in grupos_filas.items() if len(indices) > 1}
        
        for fila, indices in grupos_repetidos.items():
            # print(f"Grupo: {fila} - Indices: {indices}")
            menorIndice = min(indices)
            # print("menorIndice", menorIndice)
            for i in indices:
                #* i != 0
                if i != menorIndice:
                    for k in range(len(tpmActual[i])): #* <-- columnas
                        #* Sumar las columnas de las filas repetidas en la fila con el menor índice
                        tpmActual[menorIndice][k] += tpmActual[i][k]
                    #* Marcar la fila que ya se sumó
                    tpmActual[i] = [99] * len(tpmActual[i])
        
        #* Dividir las columnas de la fila con el menor índice entre la cantidad de índices     
        for fila, indices in grupos_repetidos.items():
            menorIndice = min(indices)
            tpmActual[menorIndice] = tpmActual[menorIndice] / len(indices)
            
        #* Eliminar las filas que ya se sumaron y no se usarán más                   
        filas_a_eliminar = []
        for i in range(len(tpmActual)):
            if(99 in tpmActual[i]):
                filas_a_eliminar.append(i)

        tpmActual = np.delete(tpmActual, filas_a_eliminar, axis=0)
        mPresente = np.delete(mPresente, filas_a_eliminar, axis=0)
        
        print("mPresente", mPresente)
        print("tpmactual", tpmActual)

        #* Tomar cada valor de la tpm y asociarlo a su respectiva fila de la TPM
        valores = {}
        for i in range(len(tpmActual)):
            valores[f'{mPresente[i]}'.replace(" ", "")] = tpmActual[i]
            
        #* Expandir la matriz tpmActual si su longitud es menor a la de las otras tpm
        longitudATener = len(partirMatricesTPM[futuro])
        filasExpandir = longitudATener - len(tpmActual)
        
        x = math.log2(longitudATener)
        
        matrizPresenteExpandida = generarMatrizPresenteInicial(int(x))
        
        #* Elementos que no se van a marginalizar
        elementosNo = {}
        for i in range(len(elementosPresentesNOAMarginalizar)):
            elementosNo[elementosPresentesNOAMarginalizar[i]] = indicesElementosPresentesNOAMarginalizar[i]
        
        #* Crear un string tipo plantilla para saber como se deben llenar los campos de la TPM usando los elementos no marginalizados y los estados actuales de los elementos
        plantilla = ''
        for i in range(len(matrizPresenteExpandida[0])):
            plantilla += 'x'

        # print("plantilla", plantilla)
        # print("elementosNo", elementosNo)
        
        elementosUtiles = elementosPresentesAMarginalizar + elementosPresentesNOAMarginalizar
        # print("elementosUtiles", elementosUtiles)
        # print("subconjuntoelementos", subconjuntoElementos)
        
        #* cambiar el indice
        for elem in elementosNo:
            indice = -1
            for i in range(len(elementosUtiles)):
                if elementosUtiles[i] == elem:
                    indice = i
                    break
                
            elementosNo[elem] = indice
        
        for i in elementosNo:
            val = elementosNo[i]
            #* convertir plantilla en lista
            plantilla = list(plantilla)
            
            v = -1
            for j in estadoActualElementos:
                if list(j.keys())[0] == i:
                    v = list(j.values())[0]
                    break
            
            plantilla[val] = str(v)
            
            plantilla = ''.join(plantilla)
            
        #* Matriz a comparar: se convierte la matriz expandida en un arreglo de strings
        matrizComparar = []
        for i in range(len(matrizPresenteExpandida)):
            matrizComparar.append( str(matrizPresenteExpandida[i]).replace('[','').replace(']','').replace(' ','').replace(',','') )
                
        #* Se comparan las plantillas con la matriz a comparar
        resultado = []
        for elemento in matrizComparar:
            transformado = ''.join(c if c == 'x' else e for c, e in zip(plantilla, elemento))
            resultado.append(transformado)
                    
        #* Se eliminan las x de los strings
        resultado_sin_x = [elemento.replace('x', '') for elemento in resultado]
        tpmActualCopia = []
        
        print("valores", valores)
        
        #* Se toman los valores de la tpm que coinciden con los resultados sin x
        for i in range(len(resultado_sin_x)):
            tpmActualCopia.append(valores[f'[{resultado_sin_x[i]}]'])
        
        #* Se convierte la tpm a un arreglo de numpy
        tpmActualCopia = np.array(tpmActualCopia)
        
        tpmActual = tpmActualCopia
        
        #* Se agrega la matriz a las matrices resultado
        matricesResultado.append({
            matrizAMarginalizar: tpmActual
        })
    
    #* Se agregan las matrices TPM que no se marginalizaron
    for i in partirMatricesTPM:
        if i not in relaciones:
            matricesResultado.append({
                i: partirMatricesTPM[i]
            })
            
    #* Se toman los estados actuales y se obtiene el indice respectivo de la TPM de donde se tomará el vector
    estadosActuales = []
    ordenColumnasPresente = []
    for i in subconjuntoElementos:
        ordenColumnasPresente.append(i)
        
    print("ordenColumnasPresente", ordenColumnasPresente)
    ordenColumnas = []
    for i in ordenColumnasPresente:
        if i in elementosT:
            ordenColumnas.append(i)
            
    print("ordernColumnas", ordenColumnas)
        
    for i in estadoActualElementos:
        if list(i.keys())[0] in ordenColumnas:
            estadosActuales.append(list(i.values())[0])
            
    print("estadosActuales", estadosActuales)
    print("elementosT", elementosT)
    
    print("nuevaMatrizPresente", nuevaMatrizPresente)
    indiceVector = -1
    for i in range(len(nuevaMatrizPresente)):
        if nuevaMatrizPresente[i].tolist() == estadosActuales:
            indiceVector = i
            break
    
    print("indiceVector", indiceVector)
    
    #* Se toman los vectores de las matrices resultado usando el indiceVector
    vectoresUtilizar = []
    for matriz in matricesResultado:
        vectoresUtilizar.append(matriz[list(matriz.keys())[0]])
        
    print("vectoresUtilizar", vectoresUtilizar)
        
    #* Se toman los vectores que se van a multiplicar
    #* Si solo hay un vector, se toma directamente ese (sucede cuando se marginalizan todas las columnas de una matriz)
    vectoresFinales = []
    for matriz in vectoresUtilizar:
        if len(matriz) == 1:
            vectoresFinales.append(matriz[0])
        else:
            vectoresFinales.append(matriz[indiceVector])
        
    #* Se realiza el producto tensorial de los vectores
    vectorFinal = producto_tensorial_n(vectoresFinales)
    
    #* Se retorna el vector final
    return vectorFinal


#* Metodo recubridor para obtener el vector de probabilidad
def obtenerVectorProbabilidad(aristas, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente, listaDeUPrimas):
    matrizPresente = copy.deepcopy(partirMatricesPresentes)
    matricesFuturo = copy.deepcopy(partirMatricesFuturas)
    matricesTPM = copy.deepcopy(partirMatricesTPM)
    estadoActual = copy.deepcopy(estadoActualElementos)
    subconjunto = copy.deepcopy(subconjuntoElementos)    
    elementosT = copy.deepcopy(elementosT)
    indicesElementosT = copy.deepcopy(indicesElementosT)
    
    return obtenerVector(aristas, matrizPresente, matricesFuturo, matricesTPM, estadoActual, subconjunto, elementosT, indicesElementosT, nuevaMatrizPresente, listaDeUPrimas)
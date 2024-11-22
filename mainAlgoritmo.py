import copy
import numpy as np
import time
import math

from data.cargarData import obtenerInformacionCSV
from utilidades.evaluarParticionesFinales import evaluarParticionesFinales
from utilidades.background import aplicarCondicionesBackground
from utilidades.marginalizacionInicial import aplicarMarginalizacion
from utilidades.organizarCandidatas import buscarValorUPrima, organizarParticionesCandidatasFinales
from utilidades.utils import encontrarParticionEquilibrioComplemento, generarMatrizPresenteInicial, obtenerParticion, particionComplemento, producto_tensorial_n
from utilidades.utils import generarMatrizFuturoInicial
from utilidades.utils import elementosNoSistemaCandidato
from utilidades.utils import producto_tensorial
from utilidades.partirRepresentacion import partirRepresentacion
from utilidades.comparaciones import compararParticion
from utilidades.vectorProbabilidad import encontrarVectorProbabilidades

#? ----------------- ENTRADAS DE DATOS ---------------------------------

# from data.matrices import TPM
from data.matrices import subconjuntoSistemaCandidato
from data.matrices import subconjuntoElementos
from data.matrices import estadoActualElementos
from data.matrices import TPM

# print("TPM", TPM)
# print("subconjuntoSistemaCandidato", subconjuntoSistemaCandidato)
# print("subconjuntoElementos", subconjuntoElementos)
# print("estadoActualElementos", estadoActualElementos)

#? ----------------- MATRIZ PRESENTE Y MATRIZ FUTURO ---------------------------------

matrizPresente = generarMatrizPresenteInicial( len(estadoActualElementos) )
matrizFuturo = generarMatrizFuturoInicial(matrizPresente)

# print("matrizPresente", matrizPresente)
# print("matrizFuturo", matrizFuturo)


#? ----------------- APLICAR CONDICIONES DE BACKGROUND ---------------------------------

#? Elementos que no hacen parte del sistema cantidato
elementosBackground = elementosNoSistemaCandidato(estadoActualElementos, subconjuntoElementos)

# #? Realizar una copia de las matrices para no modificar las originales
nuevaTPM = np.copy(TPM)
nuevaMatrizPresente = np.copy(matrizPresente)
nuevaMatrizFuturo = np.copy(matrizFuturo)


#? Ejecución de las condiciones de background
nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM = aplicarCondicionesBackground(matrizPresente, nuevaTPM, elementosBackground, nuevaMatrizFuturo, estadoActualElementos)

# print("nuevaMatrizPresente", nuevaMatrizPresente, len(nuevaMatrizPresente))
# print("nuevaMatrizFuturo", nuevaMatrizFuturo, len(nuevaMatrizFuturo))
# print("nuevaTPM", nuevaTPM, len(nuevaTPM))

#? ----------------- APLICAR MARGINALIZACIÓN INICIAL ---------------------------------

nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, nuevosIndicesElementos = aplicarMarginalizacion(nuevaMatrizFuturo, nuevaTPM, elementosBackground, estadoActualElementos, nuevaMatrizPresente)

# print("nuevaMatrizPresente", nuevaMatrizPresente)
# print("nuevaMatrizFuturo", nuevaMatrizFuturo)
# print("nuevaTPM", nuevaTPM)

#?  ------------------------ DIVIDIR EN LA REPRESENTACION -----------------------------------
#? P(ABC t | ABC t+1) = P(ABC t | A t+1) X P(ABC t | B t+1) X P(ABC t | C t+1)

#* tomar el subconjunto de elementos (los de t y t+1) con su indice

#En esta parte se tomaron los elementos de t y t+1 a partir de las aristas
elementosOrdenados = []
for arista in subconjuntoSistemaCandidato:
    #separar los elementos de t y t+1
    x = arista.split('-')
    if x[0] not in elementosOrdenados:
        elementosOrdenados.append(x[0])
    if x[1] not in elementosOrdenados:
        elementosOrdenados.append(x[1])

elementosT = [elem for elem in elementosOrdenados if 't' in elem and 't+1' not in elem]
elementosT1 = [elem for elem in elementosOrdenados if 't+1' in elem]

# print(elementosT, elementosT1)

indicesElementosT = {list(elem.keys())[0]: idx for idx, elem in enumerate(estadoActualElementos) if list(elem.keys())[0] in elementosT}

# print("elementosT1", elementosT1)
# print("indicesElementosT viejos y nuevos", indicesElementosT,  nuevosIndicesElementos)

# #? Ejecución de la representación
# print("------ REPRESENTACIÓN -----------")
partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM = partirRepresentacion(nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT1, nuevosIndicesElementos)

#*ojo hacer las copias
def obtenerVector(conjunto, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos):

    # #* miramos el conjunto y definimos lo que vamos a marginalizar y en qué matriz
    # if len(conjunto) == 1:
    #     x = conjunto[0].split('-')
    #     presente = x[0]
    #     futuro = x[1]

    #     tpmActual = partirMatricesTPM[futuro]

    #     #*voy a marginalizar lo que está en presente en la matriz futuro correspondiente
    #     #? proceso
    #     #* obtener inidice del elemento presente a marginalizar
    #     indice = indicesElementosT[presente]
    #     #* borrar la columna de la matriz presente
    #     partirMatricesPresentes = partirMatricesPresentes.T
    #     partirMatricesPresentes = np.delete(partirMatricesPresentes, indice, axis=0)
    #     partirMatricesPresentes = partirMatricesPresentes.T

    #     #* ya se eliminó la columna de la matriz presente, ahora identificar los grupos que se repiten en filas
    #     # Diccionario para agrupar los índices de las filas
    #     grupos_filas = {}

    #     # Recorrer la matriz para identificar las filas y sus índices
    #     for idx, fila in enumerate(partirMatricesPresentes):
    #         # Convertir la fila en tupla para usar como clave
    #         fila_tupla = tuple(fila)
    #         if fila_tupla in grupos_filas:
    #             grupos_filas[fila_tupla].append(idx)
    #         else:
    #             grupos_filas[fila_tupla] = [idx]

    #     # Filtrar solo los grupos que se repiten (más de un índice)
    #     grupos_repetidos = {fila: indices for fila, indices in grupos_filas.items() if len(indices) > 1}

    #     # Imprimir los resultados
    #     for fila, indices in grupos_repetidos.items():
    #         # print(f"Grupo: {fila} - Indices: {indices}")
    #         menorIndice = min(indices)
    #         for i in indices:
    #             #* i != 0
    #             if i != menorIndice:

    #                 for k in range(len(tpmActual[i])):
    #                     if k == indice:
    #                         for j in range(len(tpmActual[i])):
    #                             tpmActual[menorIndice][j] += tpmActual[i][j]
    #                         tpmActual[i].fill(99)

    #     for fila, indices in grupos_repetidos.items():
    #         menorIndice = min(indices)
    #         tpmActual[menorIndice] = tpmActual[menorIndice] / len(indices)

    #     filas_a_eliminar = []
    #     for i in range(len(tpmActual)):
    #         if(99 in tpmActual[i]):
    #             filas_a_eliminar.append(i)

    #     tpmActual = np.delete(tpmActual, filas_a_eliminar, axis=0)
    #     partirMatricesPresentes = np.delete(partirMatricesPresentes, filas_a_eliminar, axis=0)

    #     valores = {}
    #     for i in range(len(tpmActual)):
    #         valores[f'{partirMatricesPresentes[i]}'] = tpmActual[i]
            
    #     #* expandir la matriz tpmActual si su longitud es menor a la de las otras tpm
    #     # print(partirMatricesTPM)
    #     longitudATener = len(partirMatricesTPM[futuro])
    #     filasExpandir = longitudATener - len(tpmActual)
        
    #     x = math.log2(longitudATener)
        
    #     matrizPresenteExpandida = generarMatrizPresenteInicial(int(x))
        
    #     for i in range(filasExpandir):
    #         tpmActual = np.vstack([tpmActual, [88 for i in range(len(tpmActual[0]))]]) 
            
            
    #     for i in range(len(matrizPresenteExpandida)):
    #         print([matrizPresenteExpandida[i][indice]], valores[f'[{matrizPresenteExpandida[i][indice]}]'])
    #         tpmActual[i] = valores[f'[{matrizPresenteExpandida[i][indice]}]']
        
    #     estadosActuales = []
    #     ordenColumnasPresente = []
    #     for i in subconjuntoElementos:
    #         ordenColumnasPresente.append(i)

    #     for i in estadoActualElementos:
    #         if list(i.keys())[0] in ordenColumnasPresente:
    #             estadosActuales.append(list(i.values())[0])
                
    #     # print("estadosActuales", estadosActuales)
    #     # print("ordenColumnasPresente", ordenColumnasPresente)
        
    #     indiceVector = -1
    #     for i in range(len(matrizPresenteExpandida)):
    #         if matrizPresenteExpandida[i].tolist() == estadosActuales:
    #             indiceVector = i
    #             break
        
    #     #* agrego el de la tpm actual calculada
    #     vectores = []
    #     vectores.append(tpmActual[indiceVector])
        
    #     #* agrego los otros vectores de las otras tpm
    #     for i in partirMatricesTPM:
    #         if i != futuro:
    #             vectores.append(partirMatricesTPM[i][indice])
        
    #     #* producto tensorial de los vectores
    #     productoTensorial = producto_tensorial_n(np.array(vectores))
    #     print("productoTensorial", productoTensorial)
        
    #     return productoTensorial
    
    # else:
        
    # Diccionario donde se guardarán las relaciones
    relaciones = {}

    # Procesar cada arista del conjunto
    for arista in conjunto:
        # Separar presente y futuro
        x = arista.split('-')
        presente = x[0]
        futuro = x[1]
        
        # Agregar presente al futuro en el diccionario
        if futuro not in relaciones:
            relaciones[futuro] = []
        relaciones[futuro].append(presente)
    

    #todo: mirar cuando se marginalizan todas las columnas que es lo que se hace
    #* ahora que tengo las relaciones sé en que matrices t+1 se marginalizan los elementos en t
    print(relaciones)
    
    matricesResultado = []
    
    for matrizAMarginalizar in relaciones:
        print(matrizAMarginalizar)
        elementosPresentesAMarginalizar = relaciones[matrizAMarginalizar]
        
        elementosPresentesNOAMarginalizar = [elem for elem in elementosT if elem not in elementosPresentesAMarginalizar]
        
        # print(matrizAMarginalizar, elementosPresentesAMarginalizar)
        
        #* para cada elemento a marginalizar en la matriz
        mPresente = copy.deepcopy(partirMatricesPresentes)
        mFuturo = copy.deepcopy(partirMatricesFuturas[matrizAMarginalizar])
        tpmActual = copy.deepcopy(partirMatricesTPM[matrizAMarginalizar])
        
        # print("mPresente", mPresente)
        # print("mFuturo", mFuturo)
        # print("tpmActual", tpmActual)
            
        indicesElementosPresentesAMarginalizar = [indicesElementosT[elemento] for elemento in elementosPresentesAMarginalizar]
        indicesElementosPresentesNOAMarginalizar = [indicesElementosT[elemento] for elemento in elementosPresentesNOAMarginalizar]
        # print("indicesElementosPresentesAMarginalizar", indicesElementosPresentesAMarginalizar)
        # print("indicesElementosPresentesNOAMarginalizar", indicesElementosPresentesNOAMarginalizar)
        
        #* borrar las columnas de la matriz presente
        mPresente = mPresente.T
        mPresente = np.delete(mPresente, indicesElementosPresentesAMarginalizar, axis=0)
        mPresente = mPresente.T
        
        #* ya se eliminó la columna de la matriz presente, ahora identificar los grupos que se repiten en filas
        
        grupos_filas = {}

        # Recorrer la matriz para identificar las filas y sus índices
        for idx, fila in enumerate(mPresente):
            # Convertir la fila en tupla para usar como clave
            fila_tupla = tuple(fila)
            if fila_tupla in grupos_filas:
                grupos_filas[fila_tupla].append(idx)
            else:
                grupos_filas[fila_tupla] = [idx]

        # Filtrar solo los grupos que se repiten (más de un índice)
        grupos_repetidos = {fila: indices for fila, indices in grupos_filas.items() if len(indices) > 1}
        
        print("grupos_repetidos", grupos_repetidos)
        
        # print("> DEBUGEANDO <")
        
        print("presente \n", mPresente)
        print("actual \n", tpmActual)
        
        for fila, indices in grupos_repetidos.items():
            # print(f"Grupo: {fila} - Indices: {indices}")
            menorIndice = min(indices)
            # print("menorIndice", menorIndice)
            for i in indices:
                #* i != 0
                if i != menorIndice:
                    for k in range(len(tpmActual[i])): #* <-- columnas
                        tpmActual[menorIndice][k] += tpmActual[i][k]
                    
        print("actual \n", tpmActual)
                    
        for fila, indices in grupos_repetidos.items():
            menorIndice = min(indices)
            tpmActual[menorIndice] = tpmActual[menorIndice] / len(indices)
                                
        filas_a_eliminar = []
        for i in range(len(tpmActual)):
            if(99 in tpmActual[i]):
                filas_a_eliminar.append(i)

        tpmActual = np.delete(tpmActual, filas_a_eliminar, axis=0)
        mPresente = np.delete(mPresente, filas_a_eliminar, axis=0)
        
        # print("presente \n", mPresente)
        # print("actual \n", tpmActual)

        valores = {}
        for i in range(len(tpmActual)):
            valores[f'{mPresente[i]}'.replace(" ", "")] = tpmActual[i]
            
        #* expandir la matriz tpmActual si su longitud es menor a la de las otras tpm
        # print(partirMatricesTPM)
        longitudATener = len(partirMatricesTPM[futuro])
        filasExpandir = longitudATener - len(tpmActual)
        
        x = math.log2(longitudATener)
        
        matrizPresenteExpandida = generarMatrizPresenteInicial(int(x))
        
        # print("matrizPresenteExpandida \n", matrizPresenteExpandida)
        # print("tpmActual \n", tpmActual)
        # print("valores \n", valores)
        
        # print(estadoActualElementos)
        
        # print(elementosPresentesNOAMarginalizar)
        # print(indicesElementosPresentesNOAMarginalizar)
        
        elementosNo = {}
        for i in range(len(elementosPresentesNOAMarginalizar)):
            elementosNo[elementosPresentesNOAMarginalizar[i]] = indicesElementosPresentesNOAMarginalizar[i]
            
        # print("elementosNo", elementosNo)
        
        # elementosNo = {
        #     'at': 0,
        #     'ct': 2
        # }
        
        plantilla = ''
        for i in range(len(matrizPresenteExpandida[0])):
            plantilla += 'x'
        
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
        
        # print("plantilla", plantilla)
        
        matrizComparar = []
        for i in range(len(matrizPresenteExpandida)):
            matrizComparar.append( str(matrizPresenteExpandida[i]).replace('[','').replace(']','').replace(' ','').replace(',','') )
                
        # print("matrizComparar", matrizComparar)
        
        
        resultado = []
        for elemento in matrizComparar:
            transformado = ''.join(c if c == 'x' else e for c, e in zip(plantilla, elemento))
            resultado.append(transformado)
                    
            
        resultado_sin_x = [elemento.replace('x', '') for elemento in resultado]
        # print("resultado_sin_x", resultado_sin_x)
        
        tpmActualCopia = []
        # print("tpmActualCopia", tpmActualCopia)
        # print("valores", valores)
        
        # valores = {'[0]': np.array([0.3, 0.7]), '[1]': np.array([0.5, 0.5])}
        
        for i in range(len(resultado_sin_x)):
            tpmActualCopia.append(valores[f'[{resultado_sin_x[i]}]'])
        
        tpmActualCopia = np.array(tpmActualCopia)
        
        tpmActual = tpmActualCopia
        # print("presente expandida \n", matrizPresenteExpandida)
        # print("tpmActual \n", tpmActual)
        
        matricesResultado.append({
            matrizAMarginalizar: tpmActual
        })
    
    
    for i in partirMatricesTPM:
        if i not in relaciones:
            matricesResultado.append({
                i: partirMatricesTPM[i]
            })
    
    print("partirMatricesTPM", partirMatricesTPM)
    print("matricesResultado", matricesResultado)
            

# obtenerVector(['at-at+1', 'bt-at+1'], partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM)
obtenerVector(['at-at+1'], partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos)
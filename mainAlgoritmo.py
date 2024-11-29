import copy
import numpy as np
import random
import time

from utilidades.background import aplicarCondicionesBackground
from utilidades.marginalizacionInicial import aplicarMarginalizacion
from utilidades.utils import filtrar_diccionarios_unicos, generarMatrizPresenteInicial, producto_tensorial_n, seleccionarCandidata
from utilidades.utils import generarMatrizFuturoInicial
from utilidades.utils import elementosNoSistemaCandidato
from utilidades.utils import producto_tensorial
from utilidades.utils import obtenerVectorProbabilidadTPM
from utilidades.utils import calcularEMD
from utilidades.partirRepresentacion import partirRepresentacion
from utilidades.vectorProbabilidad import obtenerVectorProbabilidad
from utilidades.verificarParticiones import esBipartita
from utilidades.verificarParticiones import crearMatrizDeAdyacencia
from utilidades.utils import buscarValorUPrima
from data.cargarData import obtenerInformacionCSV

#? ----------------- ENTRADAS DE DATOS ---------------------------------

# from data.matrices import TPM
from data.matrices import subconjuntoSistemaCandidato
from data.matrices import subconjuntoElementos
from data.matrices import estadoActualElementos
# from data.matrices import TPM

_,_, TPM = obtenerInformacionCSV('csv/red5.csv')

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
print("indicesElementosT", indicesElementosT)

# print("elementosT1", elementosT1)
# print("indicesElementosT viejos y nuevos", indicesElementosT,  nuevosIndicesElementos)

# #? Ejecución de la representación
# print("------ REPRESENTACIÓN -----------")
partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM = partirRepresentacion(nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT1, nuevosIndicesElementos)

# print("partirMatricesPresentes", partirMatricesPresentes)
# print("partirMatricesFuturas", partirMatricesFuturas)
# print("partirMatricesTPM", partirMatricesTPM)
   
#? Hacer solo cuando los elementos del presente sean menos que los elementos del sistema candidato

#* comparar si el tamaño de elementosT es menor que el tamaño de subconjuntoElementos
if len(elementosT) < len(subconjuntoElementos):
    #* buscar el indice de los elementos de subconjuntoElementos que no estén en elementosT
    # print("elementosT", elementosT)
    # print("subconjuntoElementos", subconjuntoElementos)
    elementosNoPresentes = [elem for elem in subconjuntoElementos if elem not in elementosT]
    # print("elementosNoPresentes", elementosNoPresentes)
    
    indices = []
    for elem in elementosNoPresentes:
        indices.append(subconjuntoElementos.tolist().index(elem))
        
    # print("indices", indices)
    
    mPresente = nuevaMatrizPresente
    
    #* Se borran las columnas de la matriz presente que se van a marginalizar
    mPresente = mPresente.T
    mPresente = np.delete(mPresente, indices, axis=0)
    mPresente = mPresente.T
    
    # print("Matriz presente marginalizada", mPresente)
    
    filas_eliminar_presente = []
    
    for mtpm in partirMatricesTPM:
        tpmActual = partirMatricesTPM[mtpm]
        
        grupos_filas = {}

        #* Recorrer la matriz para identificar las filas y sus índices
        for idx, fila in enumerate(mPresente):
            fila_tupla = tuple(fila)
            if fila_tupla in grupos_filas:
                grupos_filas[fila_tupla].append(idx)
            else:
                grupos_filas[fila_tupla] = [idx]
                
        # print("Grupos filas", grupos_filas)

        #* Filtrar solo los grupos que se repiten (más de un índice)
        grupos_repetidos = {fila: indices for fila, indices in grupos_filas.items() if len(indices) > 1}
        
        for fila, indices in grupos_repetidos.items():
            # print(f"Grupo: {fila} - Indices: {indices}")
            menorIndice = min(indices)
            # print("menorIndice", menorIndice)
            for i in indices:
                #* i != 0
                if i != menorIndice:
                    for k in range(len(partirMatricesTPM[elementosT1[0]][0])): #* <-- columnas
                        #* Sumar las columnas de las filas repetidas en la fila con el menor índice
                        tpmActual[menorIndice][k] += tpmActual[i][k]
                    tpmActual[i] = [99] * len(tpmActual[i])
                        
        
        #* Dividir las columnas de la fila con el menor índice entre la cantidad de índices     
        for fila, indices in grupos_repetidos.items():
            menorIndice = min(indices)
            tpmActual[menorIndice] = tpmActual[menorIndice] / len(indices)
        
        # print("mPresente", mPresente)    
        # print("TPM actual", tpmActual)
        
        # Eliminar las filas que ya se sumaron y no se usarán más                   
        filas_a_eliminar = []
        for i in range(len(tpmActual)):
            if(99 in tpmActual[i]):
                filas_a_eliminar.append(i)

        # print("Filas a eliminar", filas_a_eliminar)
        filas_eliminar_presente = filas_a_eliminar

        tpmActual = np.delete(tpmActual, filas_a_eliminar, axis=0)
        # mPresente = np.delete(mPresente, filas_a_eliminar, axis=0)
        
        partirMatricesTPM[mtpm] = tpmActual
        
        
    mPresente = np.delete(mPresente, filas_eliminar_presente, axis=0)

    partirMatricesPresentes = mPresente


listaDeUPrimas = []
subconjuntoSistemaCandidatoCopia = copy.deepcopy(subconjuntoSistemaCandidato)

iteraciones_k_particionesGeneradas = []

particionesCandidatas = []

def algoritmo(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos):
    #* ESCRIBIR TODA LA LOGICA DEL ALGORITMO AQUI
    A = subconjuntoSistemaCandidato
    
    W = []
    for i in range(len(A)+1):
        W.append([])
    W[0] = []
    W[1] = [ A[0] ]
    
    restas = []
    
    for i in range( 2, len(A) + 1 ):
        
        print("\n Para la iteracion exterior i", i, "\n")
        
        elementosRecorrer = [elem for elem in A if elem not in W[i-1]]
        print("Elementos a recorrer", elementosRecorrer)
        for elemento in elementosRecorrer:
            wi_1Uelemento = W[i-1] + [elemento]
            
            u = [elemento]
            
            if 'u' in elemento:
                valor = buscarValorUPrima(listaDeUPrimas, elemento)
                # print(elemento, "es u", u)
                wi_1Uelemento = W[i-1] + valor
                u = valor
                
            
            # Calcula EMD(W[i-1] U {u})
            vectorProbabilidadUnion = obtenerVectorProbabilidad(wi_1Uelemento, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente, listaDeUPrimas)
            vectorProbabilidadUnionTPM = obtenerVectorProbabilidadTPM(estadoActualElementos, nuevaTPM, subconjuntoElementos, nuevaMatrizPresente)
            emdUnion = calcularEMD(vectorProbabilidadUnion, vectorProbabilidadUnionTPM) 
            # Calcula EMD({u})
            
            
            vectorProbabilidadU = obtenerVectorProbabilidad(u, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente, listaDeUPrimas)
            vectorProbabilidadUTPM = obtenerVectorProbabilidadTPM(estadoActualElementos, nuevaTPM, subconjuntoElementos, nuevaMatrizPresente)
            emdU = calcularEMD(vectorProbabilidadU, vectorProbabilidadUTPM)
            
            # calcular diferencia
            # EMD(W[i-1] U {u}) - EMD({u})
            valorEMDFinal = emdUnion - emdU    
            
            # Verificar si se genera biparticion (le pase el W)
            # print("antes de llamar m ady", subconjuntoSistemaCandidato)
            print("W[i-1] U {u}", wi_1Uelemento)
            
            nuevoWi_1Uelemento = []
            for arista in wi_1Uelemento:
                if 'u' in arista:
                    valor = buscarValorUPrima(listaDeUPrimas, arista)
                    for val in valor:
                        nuevoWi_1Uelemento.append(val)
                else:
                    nuevoWi_1Uelemento.append(arista)
            
            matrizAdyacencia = crearMatrizDeAdyacencia(nuevoWi_1Uelemento, subconjuntoSistemaCandidatoCopia)
            biparticion = esBipartita(matrizAdyacencia)
            # guardar elemento, emdw_1_u y emdu , diferencia
            info = {
                'RE1': {
                    'aristas': wi_1Uelemento,
                    'emd': emdUnion
                },
                'RE2': {
                    'aristas': u,
                    'emd': emdU
                },
                'elemento': elemento,
                'resultado': valorEMDFinal,
                'distribucion': vectorProbabilidadUnion,
                'biparticion': biparticion,
                "matrizConexiones": matrizAdyacencia
            }
            
            restas.append(info)
            
        #* Verificar si hay BIPARTICIONES
        restasBiparticiones = [d for d in restas if d['biparticion']['esBipartita'] == True]
        # print("Restas con biparticiones", restasBiparticiones)
        
        for elem in restasBiparticiones:
            particionesCandidatas.append(elem)
            
        #* Verificar si hay K-PARTICIONES para k > 2 (Guardar en una lista su indice de iteración)
        restasKParticiones = [d for d in restas if d['biparticion']['esBipartita'] == False and d['biparticion']['k-particiones'] > 2]
        
        for elem in restasKParticiones:
            iteraciones_k_particionesGeneradas.append((elem, i))
        
        #* Aplicar criterios
        
        elegido = None
        
        #* 1. Verificar el resultado
        
        if len(restas) == 0:
            continue
        #sacar las restas que tengan la menor diferencia (pueden ser varias)
        min_diferencia = min(d['resultado'] for d in restas)
        restas_min_diferencia = [d for d in restas if d['resultado'] == min_diferencia]
        
        #* hay repetidos
        if len(restas_min_diferencia) > 1:
            
            #* 2. Verificar el RE1 (Unión)
            min_diferencia_RE1 = min(d['RE1']['emd'] for d in restas_min_diferencia)
            # print("Minima diferencia RE1", min_diferencia_RE1)
            
            restas_min_diferencia_RE1 = [d for d in restas_min_diferencia if d['RE1']['emd'] == min_diferencia_RE1]
            # print("Restas con minima diferencia RE1", restas_min_diferencia_RE1)
            
            elegido = restas_min_diferencia_RE1[0]
            
        #* solo hay uno
        else:
            elegido = restas_min_diferencia[0]
            
        #* El elemento elegido se agrega a W[i]
        
        valoresI = copy.deepcopy(W[i-1])
        valoresI.append(elegido["elemento"])
        W[i] = valoresI
        print("Se escogió", elegido["elemento"])
        restas = []
        
        if i == len(A):
            SecuenciaResultante = []
            for x in W:
                if x == []:
                    continue
                #*agregar el elemento de la ultima posicion de x
                SecuenciaResultante.append(x[-1])
            print("SECUENCIA RESULTANTE", SecuenciaResultante)
            parCandidato = (SecuenciaResultante[-2], SecuenciaResultante[-1])
            print("Par candidato", parCandidato)
    
            uActual = [SecuenciaResultante[-2], SecuenciaResultante[-1]]
            nombreU = ""
            if(len(listaDeUPrimas) == 0):
                nombreU = "u1"
            else:
                nombreU = "u" + str(len(listaDeUPrimas) + 1)
            listaDeUPrimas.append({nombreU: uActual})
            
            # print("Lista de U'", listaDeUPrimas)

            #* nuevoA = los elementos de A que no son el par candidato + nombre del uActual
            nuevoA = []
            nuevoA = [elem for elem in A if elem not in parCandidato]
            nuevoA = nuevoA + [nombreU]
            
            print("Nuevo A", nuevoA)
            if len(nuevoA) >= 2:
                algoritmo(nuevaTPM, subconjuntoElementos, nuevoA, estadoActualElementos)
       

algoritmo(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos)

# print()
# print()
particionesCandidatasFinales =  filtrar_diccionarios_unicos(particionesCandidatas)

# # particionFinal = seleccionarCandidata(particionesCandidatasFinales)
# # print("Partición final")
# # print(particionFinal)

def obtenerRepresentacion(particionFinal, elementosT, elementosT1):
    
    matriz = copy.deepcopy( particionFinal['matrizConexiones'] )
    
    # print(elementosT)
    # print(elementosT1)
    
    particion1 = []
    
    #* Primero para filas
    for i in range(len(matriz)):
        if np.all(matriz[i] == 0):
            print("Fila", i, "es cero")
            particion1.append(elementosT[i])
            
    #* Luego para columnas
    matriz = matriz.T
    for i in range(len(matriz)):
        if np.all(matriz[i] == 0):
            print("Columna", i, "es cero")
            particion1.append(elementosT1[i])
            
    tuplaParticion1 = ( [], [] )
    for elemento in particion1:
        if 't+1' in elemento:
            tuplaParticion1[0].append(elemento)
        elif 't' in elemento and 't+1' not in elemento:
            tuplaParticion1[1].append(elemento)
            
    print("Particion 1", tuplaParticion1)
    
    particionComplemento = ( [], [] )
    for elemento in elementosT:
        if elemento not in tuplaParticion1[1]:
            particionComplemento[1].append(elemento)
    
    for elemento in elementosT1:
        if elemento not in tuplaParticion1[0]:
            particionComplemento[0].append(elemento)
            
    # print("Particion complemento", particionComplemento)
    
    return {
        'particion1': tuplaParticion1,
        'particion2': particionComplemento,
        'matriz': matriz,
        'informacion': particionFinal
    }
    
    
# #* generar numero random entre 0 y len(particionesCandidatasFinales)
random_index = random.randint(0, len(particionesCandidatasFinales) - 1)
print("random_index", random_index)

for a in particionesCandidatasFinales:
    print(a)

x = obtenerRepresentacion(particionesCandidatasFinales[random_index], elementosT, elementosT1)
print(x)
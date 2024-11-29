import copy
import numpy as np
import time
import math

from utilidades.background import aplicarCondicionesBackground
from utilidades.marginalizacionInicial import aplicarMarginalizacion
from utilidades.utils import generarMatrizPresenteInicial, producto_tensorial_n
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


   
# vector1 = obtenerVectorProbabilidad(['at-at+1', 'bt-at+1', 'ct-at+1'], partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente)

# vector2 = obtenerVectorProbabilidadTPM(estadoActualElementos, nuevaTPM, subconjuntoElementos, nuevaMatrizPresente)

# print("Vector 1", vector1)
# print("Vector 2", vector2)

# print("Diferencia", calcularEMD(vector1, vector2))

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
            vectorProbabilidadUnion = obtenerVectorProbabilidad(wi_1Uelemento, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente)
            vectorProbabilidadUnionTPM = obtenerVectorProbabilidadTPM(estadoActualElementos, nuevaTPM, subconjuntoElementos, nuevaMatrizPresente)
            emdUnion = calcularEMD(vectorProbabilidadUnion, vectorProbabilidadUnionTPM) 
            # Calcula EMD({u})
            
            
            vectorProbabilidadU = obtenerVectorProbabilidad(u, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, elementosT, indicesElementosT, nuevaMatrizPresente)
            vectorProbabilidadUTPM = obtenerVectorProbabilidadTPM(estadoActualElementos, nuevaTPM, subconjuntoElementos, nuevaMatrizPresente)
            emdU = calcularEMD(vectorProbabilidadU, vectorProbabilidadUTPM)
            
            # calcular diferencia
            # EMD(W[i-1] U {u}) - EMD({u})
            valorEMDFinal = emdUnion - emdU    
            
            # Verificar si se genera biparticion (le pase el W)
            # print("antes de llamar m ady", subconjuntoSistemaCandidato)
            matrizAdyacencia = crearMatrizDeAdyacencia(wi_1Uelemento, subconjuntoSistemaCandidatoCopia)
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
        #TODO: ORGANIZAR LO DE K-PARTICIONES EL EL METODO 
        restasKParticiones = [d for d in restas if d['biparticion']['esBipartita'] == False and d['biparticion']['k-particiones'] > 2]
        
        for elem in restasKParticiones:
            iteraciones_k_particionesGeneradas.append((elem, i))
        
        #* Aplicar criterios
        
        elegido = None
        
        #* 1. Verificar el resultado
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
print("Cantidad de cantidatas generadas:", len(particionesCandidatas))
print("candidatas")
for c in particionesCandidatas:
    print(c)
    
print()
print()
print("Cantidad de k-particiones generadas:", len(iteraciones_k_particionesGeneradas))
print("k-particiones")
for k in iteraciones_k_particionesGeneradas:
    print(k[0], 'iteracion', k[1])
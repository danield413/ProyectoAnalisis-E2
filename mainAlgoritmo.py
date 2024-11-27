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
from pruebaBipartito import esBipartita
from pruebaBipartito import crearMatrizDeAdyacencia
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

def algoritmo(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos):
    #* ESCRIBIR TODA LA LOGICA DEL ALGORITMO AQUI
    A = subconjuntoSistemaCandidato
    print("Llamado algoritmo", A)
    
    W = []
    for i in range(len(A)+1):
        W.append([])
    W[0] = []
    W[1] = [ A[0] ]
    
    restas = []
    
    for i in range( 2, len(A) + 1 ):
        # print("iteracion", i)
        
        elementosRecorrer = [elem for elem in A if elem not in W[i-1]]
        print("Elementos a recorrer", elementosRecorrer)
        
        for elemento in elementosRecorrer:
            wi_1Uelemento = W[i-1] + [elemento]
            
            u = [elemento]
            
            if 'u' in elemento:
                valor = buscarValorUPrima(listaDeUPrimas, elemento)
                u = valor
                print(elemento, "es u", u)
                wi_1Uelemento = W[i-1] + u
            
            
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
            matrizAdyacencia = crearMatrizDeAdyacencia(W[i-1], subconjuntoSistemaCandidatoCopia)
            biparticion = esBipartita(matrizAdyacencia)
            # guardar elemento, emdw_1_u y emdu , diferencia
            info = {
                'elemento': elemento,
                'emdW1U': emdUnion,
                'emdU': emdU,
                'diferencia': valorEMDFinal,
                'biparticion': biparticion
            }
            #TODO: MIRAR LO DE LA VERIFICACION DE LAS K-PARTICIONES PARA EVITAR SEGUIRLAS GENERANDO
            print(info)
            restas.append(info)
        
        #* Seleccionar el u que minimiza EMD(W[i-1] U {vi})
        # usando el arreglo de restas y teniendo en cuenta los criterios
        '''
        1. Diferencia
        2. EMD(W[i-1] U {u})
        3. Bipartición
        '''
        
        #* sacar las restas que tengan la menor diferencia (pueden ser varias)
        min_diferencia = min(d['diferencia'] for d in restas)
        restas_min_diferencia = [d for d in restas if d['diferencia'] == min_diferencia]

        # print("Restas con menor diferencia")
        # for x in restas_min_diferencia:
        #     print("elemento", x["elemento"], "emdw_1_u", x["emdW1U"], "EMD", x["diferencia"], "biparticion", x["biparticion"])
        
        #*Primer criterio: diferencia
        # print()
        # print("Restas con menor diferencia")
        # for x in restas_min_diferencia:
        #     print("elemento", x["elemento"], "emdw_1_u", x["emdW1U"], "EMD", x["diferencia"], "biparticion", x["biparticion"])
        
        if len(restas_min_diferencia) == 1:    
            restas_min_diferencia = restas_min_diferencia[0]
            
        #* Segundo criterio: emdW1U
        elif len(restas_min_diferencia) > 1:
            min_emdW1U = min(d['emdW1U'] for d in restas_min_diferencia)
            emdW1U_min_valor = [d for d in restas_min_diferencia if d['emdW1U'] == min_emdW1U]
            # print()
            # print("Restas con menor emdW1U")
            # for x in emdW1U_min_valor:
            #     print("elemento", x["elemento"], "emdw_1_u", x["emdW1U"], "EMD", x["diferencia"], "biparticion", x["biparticion"])
                
            #* Tercer criterio: biparticion
            
            if len(emdW1U_min_valor) == 1:
               restas_min_diferencia = emdW1U_min_valor[0] 
            elif len(emdW1U_min_valor) > 1:
                biparticion = [d for d in emdW1U_min_valor if d['biparticion']["esBipartita"] == True]
                # print()
                # print("Restas con biparticion")
                # print("Biparticion", biparticion)
                
                #* si hay varias que ya cumplen los tres criterios se elige la primera
                if len(biparticion) >= 2:
                    restas_min_diferencia = biparticion[0]
                elif len(biparticion) == 1:
                    restas_min_diferencia = biparticion[0]
                    #* si no hay ninguna que cumpla con el tercer criterio significa que las que quedan no tienen biparticion por lo cual se tiene que elegir la que tenga menor diferencia
                elif len(biparticion) == 0:
                    #* ordernar la diferencia de menor a mayor
                    emdW1U_min_valor = sorted(emdW1U_min_valor, key=lambda x: x['diferencia'])
                    #* elegir la primera
                    restas_min_diferencia = emdW1U_min_valor[0]
                    
        elif restas_min_diferencia == 1:
            restas_min_diferencia = restas_min_diferencia[0]
            
        # print("Elegida \n", restas_min_diferencia)
        
        
        valoresI = copy.deepcopy(W[i-1])
        valoresI.append(restas_min_diferencia["elemento"])
        W[i] = valoresI
        restas = []
        
        if i == len(A):
            SecuenciaResultante = []
            for x in W:
                if x == []:
                    continue
                #*agregar el elemento de la ultima posicion de x
                SecuenciaResultante.append(x[-1])
            # print("SECUENCIA RESULTANTE", SecuenciaResultante)
            parCandidato = (SecuenciaResultante[-2], SecuenciaResultante[-1])
            # print("Par candidato", parCandidato)
    
            uActual = [SecuenciaResultante[-2], SecuenciaResultante[-1]]
            nombreU = ""
            if(len(listaDeUPrimas) == 0):
                nombreU = "u1"
            else:
                nombreU = "u" + str(len(listaDeUPrimas) + 1)
            listaDeUPrimas.append({nombreU: uActual})

            #* nuevoA = los elementos de A que no son el par candidato + nombre del uActual
            nuevoA = []
            nuevoA = [elem for elem in A if elem not in parCandidato]
            nuevoA = nuevoA + [nombreU]
            
            # print("Nuevo A", nuevoA)
            # if len(nuevoA) >= 2:
            #     algoritmo(nuevaTPM, subconjuntoElementos, nuevoA, estadoActualElementos)
            
algoritmo(nuevaTPM, subconjuntoElementos, subconjuntoSistemaCandidato, estadoActualElementos)
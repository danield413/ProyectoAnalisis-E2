import copy
from utilidades.vectorProbabilidad import encontrarVectorProbabilidades
from utilidades.utils import producto_tensorial
from utilidades.comparaciones import compararParticion

def evaluarParticionesFinales(particionesFinales, partirMatricesPresentes, partirMatricesFuturas, partirMatricesTPM, estadoActualElementos, subconjuntoElementos, indicesElementosT, nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT):

    print("evaluar")
    for i in particionesFinales:
        i
        
    print(".------.")
    
    particionMenorEMD = None
    vectorParticionMenorEMD = None
    
    particionesEMD = []

    for i in particionesFinales:
        print(i)
        particion1 = i["p1"]
        particion2 = i["p2"]

        copiaMatricesPresentes = copy.deepcopy(partirMatricesPresentes)
        copiaMatricesFuturas = copy.deepcopy(partirMatricesFuturas)
        copiaMatricesTPM = copy.deepcopy(partirMatricesTPM)

        vectorp1 = encontrarVectorProbabilidades(particion1, copiaMatricesPresentes, copiaMatricesFuturas, copiaMatricesTPM, estadoActualElementos, subconjuntoElementos, indicesElementosT, nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT)
        copiaMatricesPresentes = copy.deepcopy(partirMatricesPresentes)
        copiaMatricesFuturas = copy.deepcopy(partirMatricesFuturas)
        copiaMatricesTPM = copy.deepcopy(partirMatricesTPM)

        vectorp2 = encontrarVectorProbabilidades(particion2, copiaMatricesPresentes, copiaMatricesFuturas, copiaMatricesTPM, estadoActualElementos, subconjuntoElementos, indicesElementosT, nuevaMatrizPresente, nuevaMatrizFuturo, nuevaTPM, elementosT)

        vectorResultado = producto_tensorial(vectorp1, vectorp2)

        valorEMD = compararParticion(vectorResultado, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos)
        
        particionesEMD.append((i, valorEMD))
        
        if particionMenorEMD == None:
            particionMenorEMD = (i, valorEMD)
            vectorParticionMenorEMD = vectorResultado
        else:
            if valorEMD < particionMenorEMD[1]:
                particionMenorEMD = (i, valorEMD)
                vectorParticionMenorEMD = vectorResultado

    return {
        "particionesEMD": particionesEMD,
        "particionMenorEMD": particionMenorEMD,
        "vectorParticionMenorEMD": vectorParticionMenorEMD
    }
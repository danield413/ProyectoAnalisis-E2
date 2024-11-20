from utilidades.utils import calcularEMD


#* Método que me compara el vector resultante de la partición con el vector original de la TPM con la que estoy trabajando
#* PARAMS
#* resultadoParticion: Vector resultante de la partición
#* nuevaMatrizPresente: Matriz presente
#* nuevaTPM: Matriz de transición de probabilidad
#* con la nuevvaMatrizPresente y la nuevaTPM se obtiene el vector de probabilidades original

def compararParticion(resultadoParticion, nuevaMatrizPresente, nuevaTPM, subconjuntoElementos, estadoActualElementos):
    estadosActuales = []
    ordenColumnasPresente = []
    for i in subconjuntoElementos:
        ordenColumnasPresente.append(i)

    for i in estadoActualElementos:
        if list(i.keys())[0] in ordenColumnasPresente:
            estadosActuales.append(list(i.values())[0])
            

    indiceVector = -1
    for i in range(len(nuevaMatrizPresente)):
        if nuevaMatrizPresente[i].tolist() == estadosActuales:
            indiceVector = i
            break

    vectorCompararTPM = nuevaTPM[indiceVector]

    #* Comparar distribuciones usando la distancia EMD (Earth Mover's Distance)
    valorEMD = calcularEMD(resultadoParticion, vectorCompararTPM)
    return valorEMD

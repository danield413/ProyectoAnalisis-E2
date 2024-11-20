#* metodo que me lee un archivo csv

import numpy as np
import pandas as pd

def cargarData(ruta):
    data = pd.read_csv(ruta)
    return data

def obtenerInformacionCSV(ruta):
    data = cargarData(ruta)
    # print(data)
    encabezados = data.columns
    estadosT1 = np.array([])
    for encabezado in encabezados:
        if encabezado not in estadosT1 and 't+1' in encabezado:
            estadosT1 = np.append(estadosT1, encabezado)
    # print(estadosT1)

    estadosT = np.array([])
    for estado in estadosT1:
        estadosT = np.append(estadosT, estado.replace('t+1', 't'))
    # print(estadosT)

    TPM = np.array(data.to_numpy(), dtype=float)
    # print(TPM)

    estados = np.append( estadosT, estadosT1 )
    subconjuntoSistemaCandidato = np.array( estados )

    subconjuntoElementos = np.array( estadosT )
    return subconjuntoSistemaCandidato, subconjuntoElementos, TPM

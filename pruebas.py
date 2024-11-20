import numpy as np

def generarMatrizPresenteInicial(n):
    # Generar un array de números de 0 a 2^n - 1
    combinaciones = np.arange(2 ** n, dtype=np.uint32)
    
    # Crear una matriz para almacenar los bits
    binario_array = np.zeros((2 ** n, n), dtype=np.uint8)
    
    # Llenar la matriz bit a bit en formato little-endian
    for bit_pos in range(n):
        # Desplazar los números y obtener el bit en la posición correspondiente
        binario_array[:, bit_pos] = (combinaciones >> bit_pos) & 1
    
    return binario_array

#probar

for i in generarMatrizPresenteInicial(10):
    print(i)
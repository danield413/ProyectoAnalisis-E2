
def buscarValorUPrima(listaDeU, uprima):
    for u in listaDeU:
        nombre = list(u.keys())[0]
        if nombre == uprima:
            # Llama recursivamente si algún valor contiene una 'u'
            valor_resuelto = []
            for elemento in u[nombre]:
                if 'u' in elemento:
                    # Obtener el valor sin 'u' embebidas recursivamente
                    valor_resuelto.extend(buscarValorUPrima(listaDeU, elemento))
                else:
                    valor_resuelto.append(elemento)
            return valor_resuelto
    return []  # Devuelve una lista vacía si no se encuentra el uprima

        
def organizarParticionesCandidatasFinales(particionesCandidatasFinales, listaDeU):
    
    nuevas = []

    #* remocion de las u
    for i in particionesCandidatasFinales:
        p1 = i['p1']
        p2 = i['p2']

        #* valores a reemplazar
        for x in p1[0]:
            if 'u' in x:
                valor = buscarValorUPrima(listaDeU, x)
                print(x, valor)
                p1[0].extend(valor)
                p1[0].remove(x)

        for x in p1[1]:
            if 'u' in x:
                valor = buscarValorUPrima(listaDeU, x)
                p1[1].extend(valor)
                p1[1].remove(x)

        for x in p2[0]:
            if 'u' in x:
                valor = buscarValorUPrima(listaDeU, x)
                p2[0].extend(valor)
                p2[0].remove(x)

        for x in p2[1]:
            if 'u' in x:
                valor = buscarValorUPrima(listaDeU, x)
                p2[1].extend(valor)
                p2[1].remove(x)

        nuevas.append({
            'p1': p1,
            'p2': p2
        })

    #* ahora, poner los elementos de t+1 en la primera posición de la partición y los de t en la segunda
    for i in nuevas:
        p1 = i['p1']
        p2 = i['p2']

    #* organizar t+1 en izquierda y t en derecha
    for i in nuevas:
        particion1 = i['p1']
        particion2 = i['p2']

        # Para particion1
        elementos_a_mover = []

        # Identificar elementos que contienen 't' en particion1[0]
        for elem in particion1[0]:
            if 't' in elem:
                elementos_a_mover.append(elem)

        # Mover los elementos de 'elementos_a_mover' a particion1[1] y eliminarlos de particion1[0]
        for elem in elementos_a_mover:
            particion1[0].remove(elem)
            particion1[1].append(elem)

        # Resetear elementos_a_mover
        elementos_a_mover = []

        # Identificar elementos que contienen 't+1' en particion1[1]
        for elem in particion1[1]:
            if 't+1' in elem:
                elementos_a_mover.append(elem)

        # Mover los elementos de 'elementos_a_mover' a particion1[0] y eliminarlos de particion1[1]
        for elem in elementos_a_mover:
            particion1[1].remove(elem)
            particion1[0].append(elem)

        # Repetir el mismo proceso para particion2

        elementos_a_mover = []

        # Identificar elementos que contienen 't' en particion2[0]
        for elem in particion2[0]:
            if 't' in elem:
                elementos_a_mover.append(elem)

        # Mover los elementos de 'elementos_a_mover' a particion2[1] y eliminarlos de particion2[0]
        for elem in elementos_a_mover:
            particion2[0].remove(elem)
            particion2[1].append(elem)

        # Resetear elementos_a_mover
        elementos_a_mover = []

        # Identificar elementos que contienen 't+1' en particion2[1]
        for elem in particion2[1]:
            if 't+1' in elem:
                elementos_a_mover.append(elem)

        # Mover los elementos de 'elementos_a_mover' a particion2[0] y eliminarlos de particion2[1]
        for elem in elementos_a_mover:
            particion2[1].remove(elem)
            particion2[0].append(elem)


    return nuevas
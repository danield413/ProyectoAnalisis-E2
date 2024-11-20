import tkinter as tk
from UI.interfaz import InterfazCargarDatos

#*INFO
#* El programa se ejecuta desde el archivo main.py
#* El algoritmo solo está en el archivo main-2.py, ya con la interfaz está en el archivo main.py

#? ----------------- CARGAR LA INTERFAZ ---------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazCargarDatos(root)
    root.mainloop()

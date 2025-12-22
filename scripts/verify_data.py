import torch
import matplotlib.pyplot as plt
import sys
import os

# Esto añade la carpeta actual (scripts/) al camino de búsqueda de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import RIRDataset # Importamos la clase que creamos antes

def verify_processing(data_path):
    # 1. Instanciar el dataset (ajusta la ruta a tu carpeta de datos)
    dataset = RIRDataset(data_dir=data_path)
    
    if len(dataset) == 0:
        print(f"Error: No se encontraron archivos .wav en {data_path}")
        return

    # 2. Obtener una muestra
    head, tail = dataset[0]
    
    # 3. Mostrar dimensiones (Deben ser 2400 para head y 45600 para tail)
    print(f"Dimensiones de la Cabeza (50ms): {head.shape}")
    print(f"Dimensiones de la Cola (950ms): {tail.shape}")
    
    # 4. Graficar para inspección visual
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graficar Cabeza
    axs[0].plot(head.squeeze().numpy())
    axs[0].set_title("RIR Head (Primeros 50 ms) - Entrada del Modelo")
    axs[0].set_xlabel("Muestras")
    axs[0].set_ylabel("Amplitud Normalizada")
    
    # Graficar Cola
    axs[1].plot(tail.squeeze().numpy(), color='orange')
    axs[1].set_title("RIR Tail (Siguientes 950 ms) - Objetivo (Ground Truth)")
    axs[1].set_xlabel("Muestras")
    axs[1].set_ylabel("Amplitud Normalizada")
    
    plt.tight_layout()
    plt.savefig("scripts/verificacion_data.png")
    print("Gráfica guardada en 'scripts/verificacion_data.png'. Ábrela para revisar.")
    plt.show()

if __name__ == "__main__":
    # Asegúrate de tener al menos un .wav en la carpeta 'data/'
    verify_processing(data_path="data/")
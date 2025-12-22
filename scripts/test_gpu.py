import torch
import sys

def verify_environment():
    print("--- Verificación de Entorno para DECOR ---")
    print(f"Versión de Python: {sys.version}")
    print(f"Versión de PyTorch: {torch.__version__}")
    
    # Comprobación de GPU
    cuda_available = torch.cuda.is_available()
    print(f"¿CUDA detectado por PyTorch?: {cuda_available}")
    
    if cuda_available:
        print(f"Dispositivo detectado: {torch.cuda.get_device_name(0)}")
        print(f"Versión de CUDA de la librería: {torch.version.cuda}")
    else:
        print("¡ATENCIÓN! No se detecta GPU. Entrenar DECOR en CPU será inviable.")

if __name__ == "__main__":
    verify_environment()
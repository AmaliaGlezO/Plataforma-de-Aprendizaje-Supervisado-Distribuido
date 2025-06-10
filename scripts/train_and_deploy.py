import ray
from training.distributed_trainer import execute_distributed_training

def main():
    # Iniciar Ray (en local para desarrollo)
    ray.init()
    
    print("🚀 Iniciando entrenamiento distribuido...")
    results = execute_distributed_training("../data/dataset.csv")  # Cambia la ruta según tu dataset
    
    print("\n📝 Resultados:")
    for result in results:
        print(f"- {result['model']}: {result['path']}")

if __name__ == "__main__":
    main()
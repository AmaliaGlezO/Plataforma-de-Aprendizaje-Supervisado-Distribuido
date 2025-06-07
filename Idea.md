graph TD
    A[Ray Head Node] -->|Gestiona| B[Ray Worker 1]
    A -->|Gestiona| C[Ray Worker 2]
    A -->|Gestiona| D[Ray Worker N]
    E[API Service] -->|Consulta| F[Model Registry]
    G[Monitoring Service] -->|Recoge métricas| B
    G -->|Recoge métricas| C
    G -->|Recoge métricas| D
    H[Clientes] -->|Predicciones| E


/proyecto-ml-distribuido/  
│
├── data/                  # Datasets etiquetados  
├── models/                # Modelos entrenados serializados  
├── training/              # Código de entrenamiento (con Ray)  
├── serving/               # API REST para inferencia  
├── monitor/               # Métricas y dashboards  
├── docker/                # Dockerfiles por componente  
├── scripts/               # Automatización y pruebas  
├── requirements.txt  
├── docker-compose.yml     # Orquestación local  
└── README.md

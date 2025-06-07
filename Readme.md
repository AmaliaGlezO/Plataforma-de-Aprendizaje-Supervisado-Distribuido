# 🧠 Plataforma de Aprendizaje Supervisado Distribuido

Este proyecto implementa una plataforma escalable para entrenamiento, despliegue y monitoreo de modelos de Machine Learning utilizando **Ray**, **Docker**, **Scikit-Learn** y **FastAPI**.

## 🚀 Objetivos

- Entrenar modelos supervisados en paralelo de forma distribuida.
- Desplegar modelos entrenados vía una API REST.
- Monitorear entrenamiento e inferencia en producción.
- Garantizar escalabilidad, portabilidad y tolerancia a fallos.

## ⚙️ Tecnologías

- **Ray**: Distribución de tareas y gestión de clúster.
- **Scikit-Learn**: Entrenamiento de modelos.
- **FastAPI**: Exposición de modelos vía API REST.
- **Docker**: Contenerización de los servicios.
- **Prometheus + Grafana** (o panel propio): Monitoreo y visualización.
- **Python 3.9+**

## 🗂️ Estructura del Proyecto

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

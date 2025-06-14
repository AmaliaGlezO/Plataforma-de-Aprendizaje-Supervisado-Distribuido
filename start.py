#!/usr/bin/env python3
"""
ARCHIVO PRINCIPAL DE INICIO DE LA PLATAFORMA
Este script inicia toda la infraestructura distribuida
Uso: python start.py
"""

import subprocess
import sys
import time
import yaml
import requests
import docker
import psutil
from pathlib import Path
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlatformStarter:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.docker_client = None
        self.services = {}
        
    def check_dependencies(self):
        """Verifica que Docker y Docker Compose estén instalados"""
        logger.info("🔍 Verificando dependencias del sistema...")
        
        # Verificar Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logger.info("✅ Docker instalado correctamente")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker no está instalado. Instala Docker para continuar.")
            sys.exit(1)
        
        # Verificar Docker Compose
        try:
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
            logger.info("✅ Docker Compose instalado correctamente")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker Compose no está instalado. Instala Docker Compose para continuar.")
            sys.exit(1)
        
        # Verificar que Docker esté ejecutándose
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("✅ Docker daemon está ejecutándose")
        except Exception as e:
            logger.error(f"❌ Docker daemon no está ejecutándose: {e}")
            sys.exit(1)
        
        # Verificar recursos del sistema
        self._check_system_resources()
    
    def _check_system_resources(self):
        """Verifica recursos mínimos del sistema"""
        # Verificar RAM disponible (mínimo 4GB)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 4:
            logger.warning(f"⚠️  RAM disponible: {ram_gb:.1f}GB. Se recomienda al menos 4GB")
        else:
            logger.info(f"✅ RAM disponible: {ram_gb:.1f}GB")
        
        # Verificar espacio en disco (mínimo 10GB libres)
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_free_gb < 10:
            logger.warning(f"⚠️  Espacio libre: {disk_free_gb:.1f}GB. Se recomienda al menos 10GB")
        else:
            logger.info(f"✅ Espacio libre: {disk_free_gb:.1f}GB")

    def load_configuration(self) -> Dict[str, Any]:
        """Carga configuraciones desde archivos YAML"""
        logger.info("📋 Cargando configuraciones...")
        
        config = {}
        config_files = {
            'ray': self.config_dir / 'ray_config.yaml',
            'api': self.config_dir / 'api_config.yaml',
            'logging': self.config_dir / 'logging_config.yaml'
        }
        
        for config_name, config_path in config_files.items():
            try:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config[config_name] = yaml.safe_load(f)
                    logger.info(f"✅ Configuración {config_name} cargada desde {config_path}")
                else:
                    logger.warning(f"⚠️  Archivo de configuración {config_path} no encontrado, usando valores por defecto")
                    config[config_name] = self._get_default_config(config_name)
            except Exception as e:
                logger.error(f"❌ Error cargando configuración {config_name}: {e}")
                config[config_name] = self._get_default_config(config_name)
        
        return config
    
    def _get_default_config(self, config_type: str) -> Dict[str, Any]:
        """Retorna configuraciones por defecto"""
        defaults = {
            'ray': {
                'head_port': 10001,
                'dashboard_port': 8265,
                'num_workers': 2,
                'worker_memory': '2g',
                'worker_cpus': 2
            },
            'api': {
                'port': 8000,
                'host': '0.0.0.0',
                'reload': False,
                'workers': 1
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        return defaults.get(config_type, {})

    def start_infrastructure(self):
        """Inicia todos los contenedores Docker con docker-compose"""
        logger.info("🐳 Iniciando infraestructura Docker...")
        
        # Verificar que existe docker-compose.yml
        compose_file = self.project_root / 'docker-compose.yml'
        if not compose_file.exists():
            logger.error(f"❌ Archivo docker-compose.yml no encontrado en {compose_file}")
            sys.exit(1)
        
        try:
            # Construir imágenes si es necesario
            logger.info("🔨 Construyendo imágenes Docker...")
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "build", "--parallel"
            ], check=True, cwd=self.project_root)
            
            # Iniciar servicios
            logger.info("🚀 Iniciando servicios...")
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "up", "-d"
            ], check=True, cwd=self.project_root)
            
            logger.info("✅ Infraestructura Docker iniciada correctamente")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error iniciando infraestructura Docker: {e}")
            sys.exit(1)

    def wait_for_services(self):
        """Espera a que todos los servicios estén listos"""
        logger.info("⏳ Esperando a que los servicios estén listos...")
        
        # Definir servicios a verificar
        services_to_check = [
            {'name': 'Ray Head', 'url': 'http://localhost:8265', 'timeout': 60},
            {'name': 'API REST', 'url': 'http://localhost:8000/health', 'timeout': 30},
            {'name': 'Dashboard', 'url': 'http://localhost:3000', 'timeout': 30}
        ]
        
        def check_service(service_info):
            """Verifica si un servicio está disponible"""
            name = service_info['name']
            url = service_info['url']
            timeout = service_info['timeout']
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code in [200, 404]:  # 404 puede ser válido para algunos endpoints
                        logger.info(f"✅ {name} está disponible en {url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
            
            logger.warning(f"⚠️  {name} no respondió en {timeout}s")
            return False
        
        # Verificar servicios en paralelo
        with ThreadPoolExecutor(max_workers=len(services_to_check)) as executor:
            future_to_service = {
                executor.submit(check_service, service): service['name']
                for service in services_to_check
            }
            
            ready_services = []
            for future in as_completed(future_to_service):
                service_name = future_to_service[future]
                try:
                    if future.result():
                        ready_services.append(service_name)
                except Exception as e:
                    logger.error(f"❌ Error verificando {service_name}: {e}")
        
        logger.info(f"📊 Servicios listos: {len(ready_services)}/{len(services_to_check)}")

    def verify_cluster_health(self):
        """Verifica que el cluster Ray esté funcionando correctamente"""
        logger.info("🔍 Verificando salud del cluster Ray...")
        
        try:
            # Verificar dashboard de Ray
            response = requests.get('http://localhost:8265/api/cluster_status', timeout=10)
            if response.status_code == 200:
                cluster_info = response.json()
                logger.info("✅ Ray cluster está funcionando correctamente")
                
                # Mostrar información del cluster
                if 'result' in cluster_info:
                    result = cluster_info['result']
                    logger.info(f"📊 Nodos activos: {len(result.get('nodes', []))}")
                    logger.info(f"📊 CPUs disponibles: {result.get('cluster_resources', {}).get('CPU', 0)}")
                    logger.info(f"📊 Memoria disponible: {result.get('cluster_resources', {}).get('memory', 0) / (1024**3):.1f} GB")
            else:
                logger.warning("⚠️  Ray dashboard responde pero con estado inválido")
                
        except Exception as e:
            logger.warning(f"⚠️  No se pudo verificar completamente el cluster Ray: {e}")
        
        # Verificar contenedores Docker
        try:
            containers = self.docker_client.containers.list()
            ray_containers = [c for c in containers if 'ray' in c.name.lower()]
            logger.info(f"📊 Contenedores Ray activos: {len(ray_containers)}")
            
            for container in ray_containers:
                status = container.status
                logger.info(f"   - {container.name}: {status}")
                
        except Exception as e:
            logger.error(f"❌ Error verificando contenedores: {e}")

    def display_startup_info(self):
        """Muestra información de endpoints y servicios disponibles"""
        logger.info("📋 Información de servicios disponibles:")
        
        services_info = [
            {
                'name': '🧠 Ray Dashboard',
                'url': 'http://localhost:8265',
                'description': 'Panel de control del cluster Ray'
            },
            {
                'name': '🌐 API REST',
                'url': 'http://localhost:8000',
                'description': 'API para entrenamiento y predicción'
            },
            {
                'name': '📊 Dashboard Monitoreo',
                'url': 'http://localhost:3000',
                'description': 'Dashboard de métricas y visualizaciones'
            },
            {
                'name': '📚 Documentación API',
                'url': 'http://localhost:8000/docs',
                'description': 'Documentación interactiva de la API (Swagger)'
            }
        ]
        
        print("\n" + "="*70)
        print("🚀 PLATAFORMA DE ML DISTRIBUIDO - SERVICIOS ACTIVOS")
        print("="*70)
        
        for service in services_info:
            print(f"\n{service['name']}")
            print(f"   URL: {service['url']}")
            print(f"   Descripción: {service['description']}")
        
        print("\n" + "="*70)
        print("💡 COMANDOS ÚTILES:")
        print("   - Ver logs: docker-compose logs -f")
        print("   - Detener servicios: docker-compose down")
        print("   - Reiniciar servicios: docker-compose restart")
        print("   - Ver estado: docker-compose ps")
        print("="*70 + "\n")

    def cleanup_on_error(self):
        """Limpia recursos en caso de error"""
        logger.info("🧹 Limpiando recursos...")
        try:
            subprocess.run([
                "docker-compose", "down"
            ], cwd=self.project_root, timeout=30)
        except Exception as e:
            logger.error(f"Error durante limpieza: {e}")

def main():
    """Función principal que orquesta el inicio de la plataforma"""
    starter = PlatformStarter()
    
    try:
        print("🚀 Iniciando Plataforma de ML Distribuido...")
        
        # 1. Verificar dependencias
        starter.check_dependencies()
        
        # 2. Cargar configuraciones
        config = starter.load_configuration()
        
        # 3. Iniciar infraestructura Docker
        starter.start_infrastructure()
        
        # 4. Esperar a que servicios estén listos
        starter.wait_for_services()
        
        # 5. Verificar salud del cluster
        starter.verify_cluster_health()
        
        # 6. Mostrar información de inicio
        starter.display_startup_info()
        
        print("✅ Plataforma iniciada correctamente!")
        print("📌 Presiona Ctrl+C para detener la plataforma")
        
        # Mantener el script ejecutándose para capturar Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Deteniendo plataforma...")
            starter.cleanup_on_error()
            print("✅ Plataforma detenida correctamente")
            
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        starter.cleanup_on_error()
        sys.exit(1)

if __name__ == "__main__":
    main()
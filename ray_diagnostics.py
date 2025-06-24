import requests
import socket
import subprocess
import os
from urllib.parse import urlparse

class RayDiagnostics:
    def __init__(self, ray_head_url="http://ray-head:8265"):
        self.base_url = ray_head_url
        self.parsed_url = urlparse(ray_head_url)
        self.host = self.parsed_url.hostname
        self.port = self.parsed_url.port
    
    def check_network_connectivity(self):
        """Verifica conectividad de red básica"""
        print(f"🔍 Verificando conectividad a {self.host}:{self.port}")
        
        try:
            # Test de ping básico
            result = subprocess.run(['ping', '-c', '1', self.host], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Ping exitoso a {self.host}")
            else:
                print(f"❌ Ping falló a {self.host}")
                return False
        except Exception as e:
            print(f"❌ Error en ping: {e}")
        
        # Test de puerto
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            
            if result == 0:
                print(f"✅ Puerto {self.port} está abierto")
                return True
            else:
                print(f"❌ Puerto {self.port} está cerrado o inaccesible")
                return False
        except Exception as e:
            print(f"❌ Error verificando puerto: {e}")
            return False
    
    def test_ray_endpoints(self):
        """Prueba diferentes endpoints de Ray"""
        endpoints = [
            "/",
            "/api/cluster_status",
            "/api/nodes",
            "/api/cluster_status.json"
        ]
        
        print(f"\n🔍 Probando endpoints de Ray en {self.base_url}")
        
        for endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=10)
                print(f"✅ {endpoint}: {response.status_code}")
                
                if endpoint == "/api/cluster_status" and response.status_code == 200:
                    print(f"   Respuesta: {response.json()}")
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ {endpoint}: Error de conexión")
            except requests.exceptions.Timeout:
                print(f"❌ {endpoint}: Timeout")
            except Exception as e:
                print(f"❌ {endpoint}: {e}")
    
    def check_ray_process(self):
        """Verifica si Ray está ejecutándose localmente"""
        print(f"\n🔍 Verificando procesos Ray locales")
        
        try:
            result = subprocess.run(['ray', 'status'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ray está ejecutándose localmente:")
                print(result.stdout)
            else:
                print("❌ Ray no está ejecutándose localmente")
                print(result.stderr)
        except FileNotFoundError:
            print("❌ Comando 'ray' no encontrado")
    
    def suggest_solutions(self):
        """Sugiere soluciones basadas en los diagnósticos"""
        print(f"\n💡 POSIBLES SOLUCIONES:")
        print("="*50)
        
        print("1. Verificar que Ray está ejecutándose:")
        print("   ray start --head --port=8265 --dashboard-host=0.0.0.0")
        
        print("\n2. Si usas Docker, verificar el mapeo de puertos:")
        print("   docker run -p 8265:8265 ...")
        
        print("\n3. Cambiar la URL de conexión:")
        print("   - Localhost: http://localhost:8265")
        print("   - IP específica: http://192.168.1.100:8265")
        
        print("\n4. Verificar firewall/seguridad:")
        print("   - Asegurar que el puerto 8265 esté abierto")
        print("   - Verificar reglas de firewall")
        
        print("\n5. Variables de entorno:")
        print("   export RAY_ADDRESS=ray://localhost:10001")
    
    def run_full_diagnosis(self):
        """Ejecuta diagnóstico completo"""
        print("🚀 DIAGNÓSTICO COMPLETO DE RAY")
        print("="*50)
        
        # Verificaciones básicas
        connectivity = self.check_network_connectivity()
        
        # Pruebas de endpoints
        self.test_ray_endpoints()
        
        # Verificar procesos locales
        self.check_ray_process()
        
        # Sugerencias
        self.suggest_solutions()
        
        return connectivity

# Función para usar en Streamlit
def diagnose_ray_connection(ray_url="http://ray-head:8265"):
    """Función para diagnosticar conexión a Ray desde Streamlit"""
    diagnostics = RayDiagnostics(ray_url)
    return diagnostics.run_full_diagnosis()

if __name__ == "__main__":
    # URLs comunes para probar
    urls_to_test = [
        "http://ray-head:8265",
        "http://localhost:8265",
        "http://127.0.0.1:8265"
    ]
    
    for url in urls_to_test:
        print(f"\n{'='*60}")
        print(f"PROBANDO: {url}")
        print(f"{'='*60}")
        
        diag = RayDiagnostics(url)
        diag.run_full_diagnosis()
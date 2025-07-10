param (
    [Parameter(Mandatory = $false)]
    [string]$WorkerName,
    
    [Parameter(Mandatory = $false)]
    [int]$CPUs = 2
)

function New-UniqueWorkerName {
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $randomPart = -join ((48..57) + (97..122) | Get-Random -Count 4 | ForEach-Object { [char]$_ })
    return "ray-worker${timestamp}${randomPart}"
}

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host " RAY CLUSTER - AGREGAR WORKER EXTERNO" -ForegroundColor Cyan
Write-Host "===========================================`n" -ForegroundColor Cyan

if ([string]::IsNullOrEmpty($WorkerName)) {
    $WorkerName = New-UniqueWorkerName
    Write-Host "No se proporcionó nombre para el worker. Se generará automáticamente:" -ForegroundColor Yellow
    Write-Host "  Nombre generado: $WorkerName" -ForegroundColor Green
} else {
    Write-Host "Se utilizará el nombre proporcionado para el worker:" -ForegroundColor Yellow
    Write-Host "  Nombre personalizado: $WorkerName" -ForegroundColor Green
}

Write-Host "`nConfiguración del worker:" -ForegroundColor Cyan
Write-Host "  - CPUs: $CPUs" -ForegroundColor White
Write-Host "  - Memoria compartida: 2GB" -ForegroundColor White
Write-Host "  - Red: plataforma-de-aprendizaje-supervisado-distribuido" -ForegroundColor White
Write-Host "  - Prioridad de failover: 3" -ForegroundColor White


$confirmation = Read-Host "`n¿Continuar con la creación del worker? (S/N)"
if ($confirmation -ne "S" -and $confirmation -ne "s") {
    Write-Host "`nOperación cancelada por el usuario." -ForegroundColor Yellow
    exit
}

Write-Host "`nCreando worker externo..." -ForegroundColor Cyan

Write-Host "`nEjecutando comando Docker..." -ForegroundColor DarkYellow

try {
    $bashCommand = "echo 'Worker externo iniciando...' && " +
                    "echo 'Esperando al cluster principal...' && " +
                    "sleep 10 && " +
                    "echo 'Conectando al cluster existente...' && " +
                    "ray start --address=ray-head:6379 --num-cpus=$CPUs && " +
                    "echo 'Worker externo conectado exitosamente!' && " +
                    "tail -f /dev/null"
    
    $dockerCmd = "docker run -d --name ray_worker_$WorkerName " +
                 "--hostname ray_worker_$WorkerName " +
                 "--network plataforma-de-aprendizaje-supervisado-distribuido_ray-network " +
                 "-e RAY_HEAD_SERVICE_HOST=ray-head " +
                 "-e NODE_ROLE=worker " +
                 "-e LEADER_NODE=false " +
                 "-e FAILOVER_PRIORITY=3 " +
                 "-e ENABLE_AUTO_FAILOVER=false " +
                 "--shm-size=2gb " +
                 "plataforma-de-aprendizaje-supervisado-distribuido-ray-head " +
                 "bash -c `"$bashCommand`""
    
    Write-Host $dockerCmd -ForegroundColor DarkGray
    $result = Invoke-Expression $dockerCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n¡Worker externo creado exitosamente!" -ForegroundColor Green
        Write-Host "  ID del contenedor: $result" -ForegroundColor White
        Write-Host "  Nombre del worker: ray_worker_$WorkerName" -ForegroundColor White
        Write-Host "`nEl worker se está conectando al cluster Ray. Este proceso puede tardar unos segundos..." -ForegroundColor Yellow
        Write-Host "Para verificar el estado, ejecute: 'docker logs ray_worker_$WorkerName'" -ForegroundColor DarkYellow
    }
    else {
        Write-Host "`n❌ Error al crear el worker externo." -ForegroundColor Red
        Write-Host "Código de salida: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host "`n❌ Error al ejecutar el comando Docker:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    $dockerRunning = $false
    $networkExists = $false
    $imageExists = $false
    $rayHeadRunning = $false
    
    try {
        $dockerStatus = docker info 2>&1
        if ($LASTEXITCODE -eq 0) { 
            $dockerRunning = $true 
            
            $networks = docker network ls --format "{{.Name}}" 2>&1
            if ($networks -contains "plataforma-de-aprendizaje-supervisado-distribuido_ray-network") {
                $networkExists = $true
            }
            
            $images = docker images --format "{{.Repository}}" 2>&1
            if ($images -contains "plataforma-de-aprendizaje-supervisado-distribuido-ray-head") {
                $imageExists = $true
            }
    
            $containers = docker ps --format "{{.Names}}" 2>&1
            if ($containers -contains "ray-head") {
                $rayHeadRunning = $true
            }
        }
    } catch {
    }
    
    Write-Host "`nDiagnóstico:" -ForegroundColor Yellow
    Write-Host "  - Docker instalado y en ejecución: $(if($dockerRunning){"✅ Sí"}else{"❌ No"})" -ForegroundColor $(if($dockerRunning){"Green"}else{"Red"})
    Write-Host "  - Red 'plataforma-de-aprendizaje-supervisado-distribuido_ray-network' existe: $(if($networkExists){"✅ Sí"}else{"❌ No"})" -ForegroundColor $(if($networkExists){"Green"}else{"Red"})
    Write-Host "  - Imagen 'plataforma-de-aprendizaje-supervisado-distribuido-ray-head' existe: $(if($imageExists){"✅ Sí"}else{"❌ No"})" -ForegroundColor $(if($imageExists){"Green"}else{"Red"})
    Write-Host "  - Nodo ray-head en ejecución: $(if($rayHeadRunning){"✅ Sí"}else{"❌ No"})" -ForegroundColor $(if($rayHeadRunning){"Green"}else{"Red"})
    
    Write-Host "`nPasos a seguir:" -ForegroundColor Yellow
    if (!$dockerRunning) {
        Write-Host "  1. Verifique que Docker esté instalado y en ejecución." -ForegroundColor Red
    }
    if (!$rayHeadRunning) {
        Write-Host "  2. Inicie el cluster Ray con la tarea 'Start Ray Cluster (Docker Compose)'." -ForegroundColor Red
    }
    if (!$networkExists) {
        Write-Host "  3. La red 'scr_pasd_2025_ray-network' no existe. Esto se crea al iniciar el cluster." -ForegroundColor Red
    }
    if (!$imageExists) {
        Write-Host "  4. La imagen 'scr_pasd_2025-ray-head' no existe. Ejecute 'docker-compose build'." -ForegroundColor Red
    }
}

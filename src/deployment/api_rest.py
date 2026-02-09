"""
API REST para detección de defectos MVTec AD usando FastAPI.

Endpoints:
- GET /health → Verifica que API está funcionando
- GET /classes → Lista clases disponibles
- POST /detect → Detecta defectos en una imagen (con clase automática o especificada)
- POST /batch → Detecta defectos en múltiples imágenes

Uso:
    uvicorn api_rest:app --host 0.0.0.0 --port 8000 --reload

Ejemplo con curl:
    # Health check
    curl http://localhost:8000/health
    
    # Detectar anomalía (clase automática)
    curl -X POST -F "file=@image.png" http://localhost:8000/detect
    
    # Detectar con clase específica
    curl -X POST -F "file=@image.png" -F "class_name=bottle" http://localhost:8000/detect
    
    # Batch de imágenes
    curl -X POST -F "files=@img1.png" -F "files=@img2.png" http://localhost:8000/batch
"""

import os
import io
import base64
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Importar detector local (relativo para soporte Docker)
from .class_retrieval import AutoAnomalyDetector
from .inference_onnx import MVTecONNXDetector, list_available_classes


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Directorio con modelos exportados (modificar según deployment)
EXPORTED_DIR = os.environ.get(
    "EXPORTED_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "exported")
)

# Configuración de detección
DEFAULT_SMOOTH_SIGMA = 0.8
DEFAULT_THRESHOLD = 0.6
DEFAULT_RETRIEVAL_K = 5
DEFAULT_ANOMALY_K = 1


# =============================================================================
# MODELOS DE RESPUESTA
# =============================================================================

class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    available_classes: List[str] = Field(..., description="Clases disponibles")
    gpu_available: bool = Field(..., description="Si hay GPU disponible")


class DetectionResult(BaseModel):
    """Resultado de detección para una imagen."""
    class_name: str = Field(..., description="Clase identificada")
    class_confidence: float = Field(..., description="Confianza en la clasificación [0, 1]")
    anomaly_score: Optional[float] = Field(None, description="Score de anomalía [0, 1]")
    is_anomaly: Optional[bool] = Field(None, description="Si se detectó anomalía")
    threshold: float = Field(..., description="Umbral usado para detección")
    anomaly_map_base64: Optional[str] = Field(
        None, 
        description="Mapa de anomalía codificado en base64 (PNG)"
    )
    all_class_scores: Optional[dict] = Field(None, description="Scores para todas las clases")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en ms")
    warning: Optional[str] = Field(None, description="Advertencia si aplica")


class BatchDetectionResult(BaseModel):
    """Resultado de detección para múltiples imágenes."""
    results: List[DetectionResult] = Field(..., description="Lista de resultados")
    total_images: int = Field(..., description="Total de imágenes procesadas")
    total_time_ms: float = Field(..., description="Tiempo total de procesamiento")
    avg_time_ms: float = Field(..., description="Tiempo promedio por imagen")


class ClassesResponse(BaseModel):
    """Respuesta con clases disponibles."""
    classes: List[str] = Field(..., description="Lista de clases disponibles")
    total: int = Field(..., description="Número total de clases")


# =============================================================================
# ESTADO GLOBAL Y INICIALIZACIÓN
# =============================================================================

# Detector global (cargado al iniciar)
detector: Optional[AutoAnomalyDetector] = None
single_class_detectors: dict = {}  # Cache para detectores de clase única


def load_detector():
    """Carga el detector de anomalías."""
    global detector
    
    if not os.path.exists(EXPORTED_DIR):
        raise RuntimeError(f"Directorio de modelos no encontrado: {EXPORTED_DIR}")
    
    try:
        detector = AutoAnomalyDetector(
            exported_dir=EXPORTED_DIR,
            retrieval_k=DEFAULT_RETRIEVAL_K,
            anomaly_k=DEFAULT_ANOMALY_K
        )
        print(f"✅ Detector cargado: {EXPORTED_DIR}")
        return True
    except Exception as e:
        print(f"⚠️ Error cargando detector automático: {e}")
        print("   Intentando modo single-class...")
        return False


def get_single_class_detector(class_name: str) -> MVTecONNXDetector:
    """Obtiene detector para una clase específica (con cache)."""
    global single_class_detectors
    
    if class_name not in single_class_detectors:
        single_class_detectors[class_name] = MVTecONNXDetector(
            exported_dir=EXPORTED_DIR,
            class_name=class_name,
            k=DEFAULT_ANOMALY_K
        )
    
    return single_class_detectors[class_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager para cargar modelo al iniciar."""
    print("🚀 Iniciando API de detección de anomalías...")
    load_detector()
    yield
    print("👋 Cerrando API...")


# =============================================================================
# APLICACIÓN FASTAPI
# =============================================================================

app = FastAPI(
    title="MVTec AD Anomaly Detection API",
    description=(
        "API REST para detección de defectos industriales usando DINOv2 + k-NN.\n\n"
        "Soporta detección automática de clase (15 categorías MVTec AD) o "
        "especificación manual de la clase a evaluar."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def encode_anomaly_map_to_base64(anomaly_map: np.ndarray) -> str:
    """
    Convierte mapa de anomalía a imagen PNG codificada en base64.
    """
    # Normalizar a [0, 255]
    amap_min, amap_max = anomaly_map.min(), anomaly_map.max()
    if amap_max > amap_min:
        amap_norm = (anomaly_map - amap_min) / (amap_max - amap_min)
    else:
        amap_norm = np.zeros_like(anomaly_map)
    
    amap_uint8 = (amap_norm * 255).astype(np.uint8)
    
    # Crear imagen PIL y codificar
    img = Image.fromarray(amap_uint8, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Carga una imagen desde un archivo subido.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error cargando imagen: {str(e)}"
        )


def to_python_float(value):
    """Convierte numpy float a Python float nativo."""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def detect_single_image(
    image: Image.Image,
    class_name: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    return_map: bool = True
) -> DetectionResult:
    """
    Detecta anomalías en una imagen.
    """
    global detector
    
    start_time = time.perf_counter()
    
    if detector is not None:
        # Usar AutoAnomalyDetector (con clasificación automática)
        result = detector.predict(
            image=image,
            smooth_sigma=DEFAULT_SMOOTH_SIGMA,
            force_class=class_name
        )
        
        detected_class = result['class_name']
        class_confidence = result['class_confidence']
        anomaly_score = result.get('anomaly_score')
        anomaly_map = result.get('anomaly_map')
        all_class_scores = result.get('all_class_scores')
        warning = result.get('warning')
        
    elif class_name:
        # Usar MVTecONNXDetector (clase específica)
        try:
            single_detector = get_single_class_detector(class_name)
            anomaly_map, anomaly_score = single_detector.predict(
                image=image,
                smooth_sigma=DEFAULT_SMOOTH_SIGMA
            )
            detected_class = class_name
            class_confidence = 1.0
            all_class_scores = {class_name: 1.0}
            warning = None
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Clase no encontrada: {class_name}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Detector automático no disponible. Especifica 'class_name'."
        )
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    # Determinar si es anomalía
    is_anomaly = None
    if anomaly_score is not None:
        is_anomaly = bool(anomaly_score > threshold)
    
    # Codificar mapa de anomalía
    anomaly_map_base64 = None
    if return_map and anomaly_map is not None:
        anomaly_map_base64 = encode_anomaly_map_to_base64(anomaly_map)
    
    # Convertir numpy types a Python nativos para serialización JSON
    class_confidence_native = to_python_float(class_confidence)
    anomaly_score_native = to_python_float(anomaly_score)
    
    # Convertir all_class_scores
    all_class_scores_native = None
    if all_class_scores:
        all_class_scores_native = {
            k: to_python_float(v) for k, v in all_class_scores.items()
        }
    
    return DetectionResult(
        class_name=detected_class,
        class_confidence=class_confidence_native,
        anomaly_score=anomaly_score_native,
        is_anomaly=is_anomaly,
        threshold=threshold,
        anomaly_map_base64=anomaly_map_base64,
        all_class_scores=all_class_scores_native,
        processing_time_ms=processing_time,
        warning=warning
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """
    Verifica el estado del servicio.
    
    Retorna información sobre:
    - Estado del servicio
    - Si el modelo está cargado
    - Clases disponibles
    - Disponibilidad de GPU
    """
    global detector
    
    try:
        import onnxruntime as ort
        gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    except:
        gpu_available = False
    
    try:
        classes = list_available_classes(EXPORTED_DIR)
    except:
        classes = []
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=detector is not None or len(single_class_detectors) > 0,
        available_classes=classes,
        gpu_available=gpu_available
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Sistema"])
async def get_classes():
    """
    Lista las clases disponibles para detección.
    """
    try:
        classes = list_available_classes(EXPORTED_DIR)
        return ClassesResponse(classes=classes, total=len(classes))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listando clases: {str(e)}"
        )


@app.post("/detect", response_model=DetectionResult, tags=["Detección"])
async def detect_anomaly(
    file: UploadFile = File(..., description="Imagen a analizar (PNG, JPG)"),
    class_name: Optional[str] = Form(
        None,
        description="Clase a evaluar. Si no se especifica, se detecta automáticamente."
    ),
    threshold: float = Form(
        DEFAULT_THRESHOLD,
        ge=0.0, le=1.0,
        description="Umbral de anomalía [0, 1]"
    ),
    return_map: bool = Form(
        True,
        description="Si retornar el mapa de anomalía en base64"
    )
):
    """
    Detecta defectos en una imagen.
    
    **Flujo:**
    1. Si `class_name` se especifica, usa el memory bank de esa clase
    2. Si no, identifica automáticamente la clase usando CLS tokens
    3. Calcula el mapa de anomalía usando k-NN sobre patch embeddings
    4. Retorna score, clasificación y opcionalmente el mapa de anomalía
    
    **Respuesta:**
    - `class_name`: Clase identificada/usada
    - `class_confidence`: Confianza en la clasificación
    - `anomaly_score`: Score de anomalía [0, 1], mayor = más anómalo
    - `is_anomaly`: Si supera el umbral
    - `anomaly_map_base64`: Mapa PNG codificado (37x37 pixels)
    """
    # Validar archivo
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (PNG, JPG)"
        )
    
    # Cargar imagen
    image = await load_image_from_upload(file)
    
    # Detectar
    result = detect_single_image(
        image=image,
        class_name=class_name,
        threshold=threshold,
        return_map=return_map
    )
    
    return result


@app.post("/batch", response_model=BatchDetectionResult, tags=["Detección"])
async def detect_batch(
    files: List[UploadFile] = File(..., description="Imágenes a analizar"),
    class_name: Optional[str] = Form(
        None,
        description="Clase a evaluar para todas las imágenes"
    ),
    threshold: float = Form(
        DEFAULT_THRESHOLD,
        ge=0.0, le=1.0,
        description="Umbral de anomalía [0, 1]"
    ),
    return_maps: bool = Form(
        False,
        description="Si retornar mapas de anomalía (puede ser lento)"
    )
):
    """
    Detecta defectos en múltiples imágenes.
    
    **Notas:**
    - Procesa las imágenes secuencialmente
    - Si `class_name` se especifica, aplica a todas las imágenes
    - Por defecto no retorna mapas para mejor rendimiento
    
    **Límites:**
    - Máximo 50 imágenes por request (recomendado < 20)
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Máximo 50 imágenes por request"
        )
    
    start_time = time.perf_counter()
    results = []
    
    for file in files:
        try:
            image = await load_image_from_upload(file)
            result = detect_single_image(
                image=image,
                class_name=class_name,
                threshold=threshold,
                return_map=return_maps
            )
            results.append(result)
        except HTTPException as e:
            # Agregar resultado con error
            results.append(DetectionResult(
                class_name="error",
                class_confidence=0.0,
                anomaly_score=None,
                is_anomaly=None,
                threshold=threshold,
                anomaly_map_base64=None,
                all_class_scores=None,
                processing_time_ms=0,
                warning=str(e.detail)
            ))
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    return BatchDetectionResult(
        results=results,
        total_images=len(results),
        total_time_ms=total_time,
        avg_time_ms=total_time / len(results) if results else 0
    )


# =============================================================================
# MANEJO DE ERRORES
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_rest:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

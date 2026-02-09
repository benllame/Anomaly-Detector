# 🔍 MVTec Anomaly Detection

[![CI](https://github.com/benllame/Anomaly-Detector/actions/workflows/ci.yml/badge.svg)](https://github.com/benllame/Anomaly-Detector/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Sistema de **detección de anomalías industriales** basado en **DINOv2** para el dataset [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Utiliza embeddings de patches con k-NN para localización pixel-wise de defectos.

![Example Detection](docs/images/000_comparison.png)

## ✨ Características

- 🧠 **DINOv2 ViT-B/14** como extractor de características (sin fine-tuning)
- 📦 **Exportación ONNX** para inferencia ligera sin PyTorch
- 🌐 **API REST** con FastAPI para integración en producción
- 🐳 **Docker** para deployment

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Image   │────▶│   DINOv2 ViT-B   │────▶│ Patch Embeddings│
│                 │     │     (ONNX)       │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │   Memory Bank    │◀─────────────┘
                        │  (Normal Patches)│     k-NN Search
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐     ┌─────────────────┐
                        │  Anomaly Scores  │────▶│   Anomaly Map   │
                        │   per Patch      │     │   (16x16 → HxW) │
                        └──────────────────┘     └─────────────────┘
```

## 📁 Estructura del Proyecto

```
mvtec_anomaly_detection/
├── src/
│   ├── deployment/
│   │   ├── api_rest.py          # API FastAPI
│   │   ├── inference_onnx.py    # Inferencia ONNX
│   │   ├── export_onnx.py       # Exportación de modelos
│   │   └── class_retrieval.py   # Clasificación automática
│   ├── evaluation/
│   │   ├── eval.py              # Script de evaluación
│   │   └── visualize_anomalies.py
│   └── config.py
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml       # Stack completo con monitoring
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml       # Configuración Prometheus
│   └── grafana/
│       ├── datasources.yml      # Datasource Prometheus
│       ├── dashboards.yml       # Provisioning dashboards
│       └── dashboards/          # JSON dashboards
├── notebooks/                    # Análisis exploratorio
├── tests/                        # Unit & integration tests
├── docs/
│   ├── EDA.md                   # Análisis exploratorio
│   └── EVALUATION.md            # Métricas de evaluación
└── requirements.txt
```

## 🚀 Instalación

### Requisitos
- Python 3.10+
- CUDA 11.8+ (opcional, para GPU)

### Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/benllame/Anomaly-Detector.git
cd Anomaly-Detector

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Con Docker (incluye Monitoring)

```bash
cd docker
docker-compose up --build
```

**Servicios disponibles:**

| Servicio | URL | Descripción |
|----------|-----|-------------|
| API | http://localhost:8000 | API REST de detección |
| Prometheus | http://localhost:9090 | Métricas y almacenamiento |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

## 📖 Uso

### API REST

```bash
# Iniciar servidor
cd src/deployment
uvicorn api_rest:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado del servicio |
| `GET` | `/classes` | Listar clases disponibles |
| `POST` | `/detect` | Detectar anomalías en imagen |
| `POST` | `/detect/batch` | Detección en múltiples imágenes |

**Ejemplo con cURL:**

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@imagen_test.png" \
  -F "threshold=0.5" \
  -F "return_map=true"
```

**Respuesta:**

```json
{
  "class_name": "bottle",
  "class_confidence": 0.95,
  "is_anomalous": true,
  "anomaly_score": 0.78,
  "anomaly_percentage": 12.5,
  "anomaly_map_base64": "iVBORw0KGgo...",
  "processing_time_ms": 48.2
}
```

### Python API

```python
from src.deployment.inference_onnx import MVTecONNXDetector
from PIL import Image

# Cargar detector para una clase
detector = MVTecONNXDetector(
    exported_dir="src/exported",
    class_name="bottle",
    k=1
)

# Procesar imagen
image = Image.open("test_image.png")
anomaly_map = detector.predict(image)

# Obtener métricas
print(f"Max anomaly score: {anomaly_map.max():.3f}")
```

## 📊 Métricas de Evaluación

Resultados en el dataset MVTec AD completo:

| Métrica | Valor | Desviación |
|---------|-------|------------|
| **IoU** | 0.277 | ± 0.196 |
| **Dice** | 0.398 | ± 0.232 |
| **Precision** | 0.322 | ± 0.248 |
| **Recall** | 0.795 | ± 0.250 |
| **AU-PRO** | 0.831 | ± 0.172 |

**Mejores categorías:**
- 🥇 Bottle: IoU 0.56, AU-PRO 0.89
- 🥈 Tile: IoU 0.46, AU-PRO 0.85
- 🥉 Metal Nut: IoU 0.38, AU-PRO 0.80

Ver [docs/EVALUATION.md](docs/EVALUATION.md) para detalles completos.

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Solo tests unitarios
pytest tests/unit -v -m unit

# Con coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Monitoring

El stack incluye **Prometheus + Grafana** para monitoreo en producción.

### Métricas Disponibles

La API expone métricas en `/metrics`:
- `http_requests_total` - Total de requests por método, endpoint y status
- `http_request_duration_seconds` - Latencia (histograma con percentiles)
- `http_requests_in_progress` - Requests concurrentes

### Dashboard Grafana

Dashboard predefinido con paneles para:
- 📊 Request rate (req/s)
- ⏱️ Latencia (p50, p95, p99)
- ✅ Success rate
- 📉 Requests por status code y endpoint

**Acceso:** http://localhost:3000 (admin/admin)

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [DINOv2 (Meta AI)](https://github.com/facebookresearch/dinov2)
- [ONNX Runtime](https://onnxruntime.ai/)

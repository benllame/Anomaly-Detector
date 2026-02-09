"""
Tests unitarios para la API REST.

Prueba los endpoints de la API usando TestClient de FastAPI:
- Health check
- Listado de clases
- Detección de anomalías
- Procesamiento batch
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Fixtures específicas para API
# =============================================================================


@pytest.fixture
def mock_detector():
    """Mock del AutoAnomalyDetector."""
    detector = MagicMock()
    detector.predict.return_value = {
        "class_name": "bottle",
        "class_confidence": 0.95,
        "anomaly_score": 0.3,
        "anomaly_map": np.random.rand(37, 37).astype(np.float32),
        "all_class_scores": {"bottle": 0.95, "cable": 0.03},
    }
    return detector


@pytest.fixture
def mock_single_detector():
    """Mock del MVTecONNXDetector."""
    detector = MagicMock()
    detector.predict.return_value = (
        np.random.rand(37, 37).astype(np.float32),  # anomaly_map
        0.25,  # anomaly_score
    )
    return detector


@pytest.fixture
def test_image_bytes(sample_rgb_image):
    """Imagen de prueba como bytes."""
    buffer = io.BytesIO()
    sample_rgb_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def api_test_client(mock_detector):
    """
    TestClient de FastAPI con manejo seguro del estado.

    Guarda y restaura lifespan_context para evitar efectos secundarios
    entre tests. Usa mock_detector automáticamente.
    """
    with patch("deployment.api_rest.detector", mock_detector):
        with patch(
            "deployment.api_rest.list_available_classes",
            return_value=["bottle", "cable"],
        ):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            # Guardar estado original del lifespan
            original_lifespan = getattr(app.router, "lifespan_context", None)

            # Desactivar lifespan para tests
            app.router.lifespan_context = None

            try:
                client = TestClient(app, raise_server_exceptions=False)
                yield client
            finally:
                # Restaurar estado original
                app.router.lifespan_context = original_lifespan


# =============================================================================
# Tests de modelos de respuesta
# =============================================================================


class TestResponseModels:
    """Tests para modelos Pydantic de respuesta."""

    def test_health_response_model(self):
        """Verifica modelo HealthResponse."""
        from deployment.api_rest import HealthResponse

        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=True,
            available_classes=["bottle", "cable"],
            gpu_available=False,
        )

        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_detection_result_model(self):
        """Verifica modelo DetectionResult."""
        from deployment.api_rest import DetectionResult

        result = DetectionResult(
            class_name="bottle",
            class_confidence=0.95,
            anomaly_score=0.3,
            is_anomaly=False,
            threshold=0.6,
            anomaly_map_base64=None,
            all_class_scores={"bottle": 0.95},
            processing_time_ms=150.5,
            warning=None,
        )

        assert result.class_name == "bottle"
        assert result.is_anomaly is False

    def test_classes_response_model(self):
        """Verifica modelo ClassesResponse."""
        from deployment.api_rest import ClassesResponse

        response = ClassesResponse(classes=["bottle", "cable", "capsule"], total=3)

        assert len(response.classes) == 3
        assert response.total == 3


# =============================================================================
# Tests de funciones auxiliares
# =============================================================================


class TestHelperFunctions:
    """Tests para funciones auxiliares de la API."""

    def test_encode_anomaly_map_to_base64(self, sample_anomaly_map):
        """Verifica codificación de mapa a base64."""
        from deployment.api_rest import encode_anomaly_map_to_base64

        encoded = encode_anomaly_map_to_base64(sample_anomaly_map)

        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Verificar que es base64 válido
        import base64

        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    def test_encode_anomaly_map_constant_values(self):
        """Verifica codificación con valores constantes."""
        from deployment.api_rest import encode_anomaly_map_to_base64

        constant_map = np.ones((37, 37), dtype=np.float32) * 0.5
        encoded = encode_anomaly_map_to_base64(constant_map)

        assert isinstance(encoded, str)

    def test_to_python_float_numpy(self):
        """Verifica conversión de numpy a Python float."""
        from deployment.api_rest import to_python_float

        np_float = np.float32(0.5)
        result = to_python_float(np_float)

        assert isinstance(result, float)
        assert result == 0.5

    def test_to_python_float_native(self):
        """Verifica que Python float se mantiene igual."""
        from deployment.api_rest import to_python_float

        py_float = 0.5
        result = to_python_float(py_float)

        assert result == 0.5

    def test_to_python_float_none(self):
        """Verifica manejo de None."""
        from deployment.api_rest import to_python_float

        result = to_python_float(None)
        assert result is None


# =============================================================================
# Tests de endpoints con TestClient
# =============================================================================


class TestHealthEndpoint:
    """Tests para endpoint /health."""

    def test_health_returns_200(self, mock_detector):
        """Verifica que health retorna 200."""
        with patch("deployment.api_rest.detector", mock_detector):
            with patch(
                "deployment.api_rest.list_available_classes", return_value=["bottle"]
            ):
                from fastapi.testclient import TestClient

                from deployment.api_rest import app

                # Desactivar lifespan para tests
                app.router.lifespan_context = None
                client = TestClient(app, raise_server_exceptions=False)

                response = client.get("/health")

                assert response.status_code == 200

    def test_health_response_structure(self, mock_detector):
        """Verifica estructura de respuesta health."""
        with patch("deployment.api_rest.detector", mock_detector):
            with patch(
                "deployment.api_rest.list_available_classes",
                return_value=["bottle", "cable"],
            ):
                from fastapi.testclient import TestClient

                from deployment.api_rest import app

                app.router.lifespan_context = None
                client = TestClient(app, raise_server_exceptions=False)

                response = client.get("/health")
                data = response.json()

                assert "status" in data
                assert "version" in data
                assert "model_loaded" in data
                assert "available_classes" in data


class TestClassesEndpoint:
    """Tests para endpoint /classes."""

    def test_classes_returns_list(self, mock_detector):
        """Verifica que classes retorna lista."""
        with patch("deployment.api_rest.detector", mock_detector):
            with patch(
                "deployment.api_rest.list_available_classes",
                return_value=["bottle", "cable"],
            ):
                from fastapi.testclient import TestClient

                from deployment.api_rest import app

                app.router.lifespan_context = None
                client = TestClient(app, raise_server_exceptions=False)

                response = client.get("/classes")

                assert response.status_code == 200
                data = response.json()
                assert "classes" in data
                assert isinstance(data["classes"], list)


class TestDetectEndpoint:
    """Tests para endpoint /detect."""

    def test_detect_requires_image(self, mock_detector):
        """Verifica que detect requiere imagen."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post("/detect")

            assert response.status_code == 422  # Validation error

    def test_detect_with_valid_image(self, mock_detector, test_image_bytes):
        """Verifica detección con imagen válida."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/detect", files={"file": ("test.png", test_image_bytes, "image/png")}
            )

            assert response.status_code == 200
            data = response.json()
            assert "class_name" in data
            assert "anomaly_score" in data

    def test_detect_with_threshold(self, mock_detector, test_image_bytes):
        """Verifica que threshold se aplica correctamente."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/detect",
                files={"file": ("test.png", test_image_bytes, "image/png")},
                data={"threshold": 0.3},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["threshold"] == 0.3

    def test_detect_with_class_name(self, mock_detector, test_image_bytes):
        """Verifica detección con clase específica."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/detect",
                files={"file": ("test.png", test_image_bytes, "image/png")},
                data={"class_name": "bottle"},
            )

            assert response.status_code == 200

    def test_detect_invalid_file_type(self, mock_detector):
        """Verifica rechazo de archivos no-imagen."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/detect", files={"file": ("test.txt", b"not an image", "text/plain")}
            )

            assert response.status_code == 400


class TestBatchEndpoint:
    """Tests para endpoint /batch."""

    def test_batch_multiple_images(self, mock_detector, test_image_bytes):
        """Verifica procesamiento de múltiples imágenes."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/batch",
                files=[
                    ("files", ("img1.png", test_image_bytes, "image/png")),
                    ("files", ("img2.png", test_image_bytes, "image/png")),
                ],
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["total_images"] == 2

    def test_batch_max_limit(self, mock_detector, test_image_bytes):
        """Verifica límite máximo de imágenes."""
        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            # Intentar subir más de 50 imágenes
            files = [
                ("files", (f"img{i}.png", test_image_bytes, "image/png"))
                for i in range(51)
            ]

            response = client.post("/batch", files=files)

            assert response.status_code == 400


# =============================================================================
# Tests de serialización
# =============================================================================


class TestSerialization:
    """Tests para serialización de respuestas."""

    def test_numpy_types_serializable(self, mock_detector, test_image_bytes):
        """Verifica que tipos numpy se serializan correctamente."""
        # Modificar mock para retornar tipos numpy
        mock_detector.predict.return_value = {
            "class_name": "bottle",
            "class_confidence": np.float32(0.95),  # numpy type
            "anomaly_score": np.float64(0.3),  # numpy type
            "anomaly_map": np.random.rand(37, 37).astype(np.float32),
            "all_class_scores": {"bottle": np.float32(0.95)},
        }

        with patch("deployment.api_rest.detector", mock_detector):
            from fastapi.testclient import TestClient

            from deployment.api_rest import app

            app.router.lifespan_context = None
            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/detect", files={"file": ("test.png", test_image_bytes, "image/png")}
            )

            # No debería haber error de serialización
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data["class_confidence"], float)

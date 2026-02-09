"""
Tests de integración para el pipeline completo de detección.

Estos tests requieren los modelos exportados y verifican el flujo
end-to-end del sistema.
"""
import sys
from pathlib import Path
import os

import pytest
import numpy as np
from PIL import Image

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Configuración
# =============================================================================

EXPORTED_DIR = Path(__file__).parent.parent.parent / "src" / "exported"
MODELS_AVAILABLE = EXPORTED_DIR.exists() and any(EXPORTED_DIR.iterdir()) if EXPORTED_DIR.exists() else False


# =============================================================================
# Tests de integración reales (requieren modelos)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Modelos exportados no disponibles")
class TestRealModels:
    """Tests con modelos reales (requieren src/exported/)."""
    
    def test_list_available_classes_real(self):
        """Verifica que encuentra clases reales."""
        from deployment.inference_onnx import list_available_classes
        
        classes = list_available_classes(str(EXPORTED_DIR))
        
        assert len(classes) > 0
        # Verificar algunas clases conocidas de MVTec
        mvtec_classes = {'bottle', 'cable', 'capsule', 'carpet', 'grid',
                         'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                         'tile', 'toothbrush', 'transistor', 'wood', 'zipper'}
        assert any(c in mvtec_classes for c in classes)
    
    def test_onnx_detector_initialization(self):
        """Verifica inicialización del detector ONNX."""
        from deployment.inference_onnx import MVTecONNXDetector, list_available_classes
        
        classes = list_available_classes(str(EXPORTED_DIR))
        if not classes:
            pytest.skip("No hay clases disponibles")
        
        first_class = classes[0]
        detector = MVTecONNXDetector(
            exported_dir=str(EXPORTED_DIR),
            class_name=first_class
        )
        
        assert detector.class_name == first_class
        assert detector.memory_bank is not None
    
    def test_auto_detector_initialization(self):
        """Verifica inicialización del AutoAnomalyDetector."""
        from deployment.class_retrieval import AutoAnomalyDetector
        
        detector = AutoAnomalyDetector(
            exported_dir=str(EXPORTED_DIR),
            retrieval_k=5,
            anomaly_k=1
        )
        
        assert detector.retriever is not None
        assert len(detector.available_classes) > 0
    
    @pytest.mark.slow
    def test_inference_produces_valid_output(self, sample_rgb_image):
        """Verifica que la inferencia produce salida válida."""
        from deployment.class_retrieval import AutoAnomalyDetector
        
        detector = AutoAnomalyDetector(
            exported_dir=str(EXPORTED_DIR),
            retrieval_k=5,
            anomaly_k=1
        )
        
        result = detector.predict(sample_rgb_image)
        
        # Verificar estructura del resultado
        assert 'class_name' in result
        assert 'class_confidence' in result
        assert 'anomaly_map' in result
        assert 'anomaly_score' in result
        
        # Verificar valores
        assert isinstance(result['class_name'], str)
        assert 0 <= result['class_confidence'] <= 1
        assert result['anomaly_map'].shape == (37, 37)
        assert result['anomaly_score'] >= 0
    
    @pytest.mark.slow
    def test_force_class_works(self, sample_rgb_image):
        """Verifica que force_class omite clasificación."""
        from deployment.class_retrieval import AutoAnomalyDetector
        from deployment.inference_onnx import list_available_classes
        
        classes = list_available_classes(str(EXPORTED_DIR))
        if len(classes) < 2:
            pytest.skip("Se necesitan al menos 2 clases")
        
        detector = AutoAnomalyDetector(
            exported_dir=str(EXPORTED_DIR),
            retrieval_k=5,
            anomaly_k=1
        )
        
        # Forzar una clase específica
        forced_class = classes[0]
        result = detector.predict(sample_rgb_image, force_class=forced_class)
        
        assert result['class_name'] == forced_class
        assert result['class_confidence'] == 1.0


# =============================================================================
# Tests de API en modo integración
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Modelos exportados no disponibles")
class TestAPIIntegration:
    """Tests de integración para la API."""
    
    def test_full_detection_pipeline(self, sample_rgb_image):
        """Test del pipeline completo a través de la API."""
        import io
        from unittest.mock import patch
        
        # Preparar imagen
        buffer = io.BytesIO()
        sample_rgb_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        
        # Importar app sin lifespan
        from fastapi.testclient import TestClient
        from deployment.api_rest import app, load_detector
        
        # Cargar detector real
        with patch.dict(os.environ, {'EXPORTED_DIR': str(EXPORTED_DIR)}):
            from deployment import api_rest
            api_rest.EXPORTED_DIR = str(EXPORTED_DIR)
            
            # Mock del lifespan
            app.router.lifespan_context = None
            
            # Cargar detector manualmente
            load_detector()
            
            client = TestClient(app, raise_server_exceptions=False)
            
            # Probar health
            health_response = client.get("/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data['model_loaded'] is True
            
            # Probar detect
            detect_response = client.post(
                "/detect",
                files={"file": ("test.png", image_bytes, "image/png")}
            )
            
            assert detect_response.status_code == 200
            detect_data = detect_response.json()
            assert 'class_name' in detect_data
            assert 'anomaly_score' in detect_data


# =============================================================================
# Tests de rendimiento
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Modelos exportados no disponibles")
class TestPerformance:
    """Tests de rendimiento."""
    
    def test_inference_time(self, sample_rgb_image):
        """Verifica que inferencia es razonablemente rápida."""
        import time
        from deployment.class_retrieval import AutoAnomalyDetector
        
        detector = AutoAnomalyDetector(
            exported_dir=str(EXPORTED_DIR),
            retrieval_k=5,
            anomaly_k=1
        )
        
        # Warmup
        _ = detector.predict(sample_rgb_image)
        
        # Medir tiempo
        start = time.perf_counter()
        n_runs = 5
        for _ in range(n_runs):
            _ = detector.predict(sample_rgb_image)
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / n_runs * 1000  # ms
        
        # Debería ser menor a 3 segundos por imagen (sin GPU)
        assert avg_time < 3000, f"Inferencia muy lenta: {avg_time:.2f} ms"
    
    def test_batch_prediction(self, sample_rgb_image):
        """Verifica predicción en batch con validación de valores."""
        from deployment.class_retrieval import AutoAnomalyDetector
        
        detector = AutoAnomalyDetector(
            exported_dir=str(EXPORTED_DIR),
            retrieval_k=5,
            anomaly_k=1
        )
        
        images = [sample_rgb_image] * 3
        results = detector.predict_batch(images)
        
        assert len(results) == 3, "Debería retornar exactamente 3 resultados"
        
        for i, result in enumerate(results):
            # Verificar existencia de claves
            assert 'class_name' in result, f"Resultado {i}: falta class_name"
            assert 'anomaly_score' in result, f"Resultado {i}: falta anomaly_score"
            assert 'class_confidence' in result, f"Resultado {i}: falta class_confidence"
            assert 'anomaly_map' in result, f"Resultado {i}: falta anomaly_map"
            
            # Verificar tipos de datos
            assert isinstance(result['class_name'], str), f"Resultado {i}: class_name no es string"
            
            # Verificar rangos válidos (si no hay warning, los valores deben ser válidos)
            if result.get('warning') is None:
                assert result['anomaly_score'] is not None, f"Resultado {i}: anomaly_score es None sin warning"
                assert result['anomaly_score'] >= 0, f"Resultado {i}: anomaly_score negativo"
                assert 0 <= result['class_confidence'] <= 1, f"Resultado {i}: class_confidence fuera de [0,1]"
                assert result['anomaly_map'].shape == (37, 37), f"Resultado {i}: shape incorrecto de anomaly_map"

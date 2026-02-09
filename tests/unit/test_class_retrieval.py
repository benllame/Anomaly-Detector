"""
Tests unitarios para el módulo class_retrieval.

Prueba las funcionalidades de ClassRetriever y AutoAnomalyDetector incluyendo:
- Identificación de clases
- Extracción de CLS tokens
- Pipeline de detección automática
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Tests de constantes y configuración
# =============================================================================


class TestConfiguration:
    """Tests para constantes de configuración."""

    def test_imagenet_constants(self):
        """Verifica constantes de normalización."""
        from deployment.class_retrieval import IMAGENET_MEAN, IMAGENET_STD

        assert IMAGENET_MEAN.shape == (3,)
        assert IMAGENET_STD.shape == (3,)
        assert all(0 < m < 1 for m in IMAGENET_MEAN)
        assert all(0 < s < 1 for s in IMAGENET_STD)


# =============================================================================
# Tests de ClassRetriever
# =============================================================================


class TestClassRetriever:
    """Tests para la clase ClassRetriever."""

    def test_preprocess_output_shape(self, sample_rgb_image):
        """Verifica shape de preprocesamiento."""
        with patch("onnxruntime.InferenceSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [
                MagicMock(name="pixel_values", shape=[1, 3, 518, 518])
            ]
            mock_session.get_outputs.return_value = [
                MagicMock(name="embeddings", shape=[1, 1369, 1024])
            ]
            mock_session_class.return_value = mock_session

            with patch(
                "deployment.class_retrieval.ClassRetriever._load_class_cls_tokens"
            ):
                from deployment.class_retrieval import ClassRetriever

                # Mock del path del modelo
                with patch("os.path.exists", return_value=True):
                    retriever = ClassRetriever.__new__(ClassRetriever)
                    retriever.session = mock_session
                    retriever.input_name = "pixel_values"
                    retriever.input_height = 518
                    retriever.input_width = 518
                    retriever.classes = ["bottle"]
                    retriever.class_tokens = {
                        "bottle": np.random.randn(100, 1024).astype(np.float32)
                    }
                    retriever.all_tokens = np.random.randn(100, 1024).astype(np.float32)
                    retriever.token_labels = ["bottle"] * 100
                    retriever.k = 5

                    result = retriever.preprocess(sample_rgb_image)

                    assert len(result.shape) == 4
                    assert result.shape[1] == 3

    def test_identify_class_returns_tuple(self, sample_rgb_image):
        """Verifica que identify_class retorna (clase, confianza, scores)."""
        with patch("onnxruntime.InferenceSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [
                MagicMock(name="pixel_values", shape=[1, 3, 518, 518])
            ]
            mock_session.get_outputs.return_value = [
                MagicMock(name="embeddings", shape=[1, 1369, 1024])
            ]
            mock_session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            mock_session_class.return_value = mock_session

            with patch(
                "deployment.class_retrieval.ClassRetriever._load_class_cls_tokens"
            ):
                from deployment.class_retrieval import ClassRetriever

                with patch("os.path.exists", return_value=True):
                    retriever = ClassRetriever.__new__(ClassRetriever)
                    retriever.session = mock_session
                    retriever.input_name = "pixel_values"
                    retriever.output_name = "embeddings"
                    retriever.input_height = 518
                    retriever.input_width = 518
                    retriever.classes = ["bottle", "cable"]
                    retriever.class_tokens = {
                        "bottle": np.random.randn(50, 1024).astype(np.float32),
                        "cable": np.random.randn(50, 1024).astype(np.float32),
                    }
                    retriever.all_tokens = np.vstack(
                        [
                            retriever.class_tokens["bottle"],
                            retriever.class_tokens["cable"],
                        ]
                    )
                    retriever.all_cls_tokens = (
                        retriever.all_tokens
                    )  # identify_class uses this name
                    retriever.token_labels = ["bottle"] * 50 + ["cable"] * 50
                    retriever.all_labels = (
                        retriever.token_labels
                    )  # identify_class uses this name
                    retriever.k = 5
                    retriever.use_patch_model = True

                    result = retriever.identify_class(
                        sample_rgb_image, return_all_scores=True
                    )

                    assert len(result) == 3
                    class_name, confidence, scores = result
                    assert isinstance(class_name, str)
                    assert isinstance(confidence, (int, float, np.floating))
                    assert isinstance(scores, dict)

    def test_identify_class_confidence_range(self, sample_rgb_image):
        """Verifica que la confianza está en [0, 1]."""
        with patch("onnxruntime.InferenceSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = [
                MagicMock(name="pixel_values", shape=[1, 3, 518, 518])
            ]
            mock_session.get_outputs.return_value = [
                MagicMock(name="embeddings", shape=[1, 1369, 1024])
            ]
            mock_session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            mock_session_class.return_value = mock_session

            with patch(
                "deployment.class_retrieval.ClassRetriever._load_class_cls_tokens"
            ):
                from deployment.class_retrieval import ClassRetriever

                with patch("os.path.exists", return_value=True):
                    retriever = ClassRetriever.__new__(ClassRetriever)
                    retriever.session = mock_session
                    retriever.input_name = "pixel_values"
                    retriever.output_name = "embeddings"
                    retriever.input_height = 518
                    retriever.input_width = 518
                    retriever.classes = ["bottle"]
                    retriever.class_tokens = {
                        "bottle": np.random.randn(100, 1024).astype(np.float32)
                    }
                    retriever.all_tokens = retriever.class_tokens["bottle"]
                    retriever.all_cls_tokens = (
                        retriever.all_tokens
                    )  # identify_class uses this name
                    retriever.token_labels = ["bottle"] * 100
                    retriever.all_labels = (
                        retriever.token_labels
                    )  # identify_class uses this name
                    retriever.k = 5
                    retriever.use_patch_model = True

                    class_name, confidence, scores = retriever.identify_class(
                        sample_rgb_image, return_all_scores=True
                    )

                    assert 0 <= confidence <= 1


# =============================================================================
# Tests de AutoAnomalyDetector
# =============================================================================


class TestAutoAnomalyDetector:
    """Tests para AutoAnomalyDetector."""

    def test_compute_anomaly_map_shape(
        self, sample_patch_embeddings, sample_memory_bank
    ):
        """Verifica shape del mapa de anomalía."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)
            detector.anomaly_k = 1
            detector.n_patches_h = 37
            detector.n_patches_w = 37

            # Crear patches con shape correcto
            patches = sample_patch_embeddings[:1369]  # 37x37 patches

            anomaly_map, anomaly_score = detector.compute_anomaly_map(
                patches, sample_memory_bank, smooth_sigma=0.8
            )

            assert anomaly_map.shape == (37, 37)

    def test_compute_anomaly_map_valid_output(
        self, sample_patch_embeddings, sample_memory_bank
    ):
        """Verifica que el mapa de anomalía tiene valores finitos y shape correcto."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)
            detector.anomaly_k = 1
            detector.n_patches_h = 37
            detector.n_patches_w = 37

            patches = sample_patch_embeddings[:1369]

            anomaly_map, anomaly_score = detector.compute_anomaly_map(
                patches, sample_memory_bank, smooth_sigma=0.8
            )

            # Con datos aleatorios, los valores pueden ser negativos
            # Lo importante es que sean finitos y tengan el shape correcto
            assert np.isfinite(anomaly_map).all()
            assert anomaly_map.shape == (37, 37)

    def test_upsample_anomaly_map(self, sample_anomaly_map):
        """Verifica upsampling del mapa."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)

            target_size = (512, 512)
            upsampled = detector.upsample_anomaly_map(sample_anomaly_map, target_size)

            assert upsampled.shape == target_size

    def test_predict_returns_dict(self, sample_rgb_image):
        """Verifica que predict retorna un diccionario con campos esperados."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)

            # Mock de todos los componentes necesarios
            detector.retriever = MagicMock()
            detector.retriever.identify_class.return_value = (
                "bottle",
                0.95,
                {"bottle": 0.95, "cable": 0.05},
            )

            detector.session = MagicMock()
            detector.session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            detector.input_name = "pixel_values"
            detector.output_name = "embeddings"
            detector.input_height = 518
            detector.input_width = 518
            detector.n_patches_h = 37
            detector.n_patches_w = 37
            detector.anomaly_k = 1
            detector.min_confidence = 0.3
            detector._memory_banks = {}

            # Mock de memory bank
            detector._get_memory_bank = MagicMock(
                return_value=np.random.randn(3000, 1024).astype(np.float32)
            )

            # Mock preprocess
            detector.preprocess = MagicMock(
                return_value=np.random.randn(1, 3, 518, 518).astype(np.float32)
            )

            result = detector.predict(sample_rgb_image)

            assert isinstance(result, dict)
            assert "class_name" in result
            assert "class_confidence" in result

    def test_predict_with_force_class(self, sample_rgb_image):
        """Verifica que force_class omite la clasificación."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)

            detector.retriever = MagicMock()
            detector.session = MagicMock()
            detector.session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            detector.input_name = "pixel_values"
            detector.output_name = "embeddings"
            detector.input_height = 518
            detector.input_width = 518
            detector.n_patches_h = 37
            detector.n_patches_w = 37
            detector.anomaly_k = 1
            detector.min_confidence = 0.3
            detector._memory_banks = {}
            detector._get_memory_bank = MagicMock(
                return_value=np.random.randn(3000, 1024).astype(np.float32)
            )
            detector.preprocess = MagicMock(
                return_value=np.random.randn(1, 3, 518, 518).astype(np.float32)
            )

            result = detector.predict(sample_rgb_image, force_class="cable")

            assert result["class_name"] == "cable"
            assert result["class_confidence"] == 1.0
            # No debería llamar a identify_class
            detector.retriever.identify_class.assert_not_called()


# =============================================================================
# Tests de integración de componentes mocked
# =============================================================================


class TestComponentIntegration:
    """Tests de integración entre componentes (con mocks)."""

    def test_end_to_end_prediction_flow(self, sample_rgb_image):
        """Simula el flujo completo de predicción."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)

            # Configurar todos los mocks necesarios
            detector.retriever = MagicMock()
            detector.retriever.identify_class.return_value = (
                "bottle",
                0.9,
                {"bottle": 0.9, "cable": 0.1},
            )
            detector.retriever.classes = ["bottle", "cable"]

            detector.session = MagicMock()
            detector.session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            detector.input_name = "pixel_values"
            detector.output_name = "embeddings"
            detector.input_height = 518
            detector.input_width = 518
            detector.n_patches_h = 37
            detector.n_patches_w = 37
            detector.anomaly_k = 1
            detector.min_confidence = 0.3
            detector._memory_banks = {}

            detector._get_memory_bank = MagicMock(
                return_value=np.random.randn(3000, 1024).astype(np.float32)
            )
            detector.preprocess = MagicMock(
                return_value=np.random.randn(1, 3, 518, 518).astype(np.float32)
            )

            # Ejecutar predicción
            result = detector.predict(sample_rgb_image)

            # Verificar resultado completo
            assert result["class_name"] == "bottle"
            assert result["class_confidence"] >= 0.3
            assert "anomaly_map" in result
            assert "anomaly_score" in result
            assert result["anomaly_map"].shape == (37, 37)


# =============================================================================
# Tests de manejo de errores
# =============================================================================


class TestErrorHandling:
    """Tests para manejo de errores."""

    def test_low_confidence_warning(self, sample_rgb_image):
        """Verifica advertencia cuando confianza es baja."""
        from deployment.class_retrieval import AutoAnomalyDetector

        with patch.object(AutoAnomalyDetector, "__init__", lambda self, **kwargs: None):
            detector = AutoAnomalyDetector.__new__(AutoAnomalyDetector)

            # Configurar confianza baja (0.2 < min_confidence de 0.3)
            low_confidence = 0.2
            detector.retriever = MagicMock()
            detector.retriever.identify_class.return_value = (
                "bottle",
                low_confidence,
                {"bottle": 0.2, "cable": 0.15},
            )

            detector.session = MagicMock()
            detector.session.run.return_value = [
                np.random.randn(1, 1369, 1024).astype(np.float32)
            ]
            detector.input_name = "pixel_values"
            detector.output_name = "embeddings"
            detector.input_height = 518
            detector.input_width = 518
            detector.n_patches_h = 37
            detector.n_patches_w = 37
            detector.anomaly_k = 1
            detector.min_confidence = 0.3  # Umbral mayor que la confianza
            detector._memory_banks = {}
            detector._get_memory_bank = MagicMock(
                return_value=np.random.randn(3000, 1024).astype(np.float32)
            )
            detector.preprocess = MagicMock(
                return_value=np.random.randn(1, 3, 518, 518).astype(np.float32)
            )

            result = detector.predict(sample_rgb_image)

            # Verificar que el warning existe y contiene información correcta
            assert (
                "warning" in result
            ), "Debería generar warning cuando confianza < min_confidence"
            assert (
                "Confianza baja" in result["warning"]
            ), f"Warning incorrecto: {result.get('warning')}"
            assert (
                f"{low_confidence:.2f}" in result["warning"]
            ), "Warning debe incluir el valor de confianza"

            # Verificar que no se calculó anomaly cuando la confianza es baja
            assert (
                result["anomaly_score"] is None
            ), "No debería calcular anomaly_score con baja confianza"
            assert (
                result["anomaly_map"] is None
            ), "No debería calcular anomaly_map con baja confianza"

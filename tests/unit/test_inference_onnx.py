"""
Tests unitarios para el módulo inference_onnx.

Prueba las funcionalidades del detector MVTecONNXDetector incluyendo:
- Preprocesamiento de imágenes
- Cálculo de anomalías
- Evaluación de métricas
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
# Tests de preprocesamiento
# =============================================================================


class TestPreprocessing:
    """Tests para funciones de preprocesamiento."""

    def test_image_normalization_constants(self):
        """Verifica que las constantes de normalización ImageNet están definidas."""
        from deployment.inference_onnx import IMAGENET_MEAN, IMAGENET_STD

        # Verificar valores estándar de ImageNet
        expected_mean = np.array([0.485, 0.456, 0.406])
        expected_std = np.array([0.229, 0.224, 0.225])

        np.testing.assert_array_almost_equal(IMAGENET_MEAN, expected_mean)
        np.testing.assert_array_almost_equal(IMAGENET_STD, expected_std)

    def test_preprocess_output_shape(
        self, sample_rgb_image, mock_onnx_session, configured_detector_dir
    ):
        """Verifica que preprocess retorna el shape correcto."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                result = detector.preprocess(sample_rgb_image)

                # Verificar shape [1, 3, H, W]
                assert len(result.shape) == 4
                assert result.shape[0] == 1
                assert result.shape[1] == 3

    def test_preprocess_dtype(
        self, sample_rgb_image, mock_onnx_session, configured_detector_dir
    ):
        """Verifica que preprocess retorna float32."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                result = detector.preprocess(sample_rgb_image)
                assert result.dtype == np.float32


# =============================================================================
# Tests de utilidades
# =============================================================================


class TestUtilities:
    """Tests para funciones de utilidad."""

    def test_list_available_classes(self, temp_export_dir):
        """Verifica que list_available_classes encuentra las clases."""
        from deployment.inference_onnx import list_available_classes

        classes = list_available_classes(str(temp_export_dir))

        assert isinstance(classes, list)
        assert "bottle" in classes

    def test_list_available_classes_empty_dir(self, tmp_path):
        """Verifica que retorna lista vacía para directorio sin clases."""
        from deployment.inference_onnx import list_available_classes

        empty_dir = tmp_path / "empty_export"
        empty_dir.mkdir()

        classes = list_available_classes(str(empty_dir))
        assert classes == []

    def test_list_available_classes_nonexistent_dir(self, tmp_path):
        """Verifica que lanza error para directorios inexistentes."""
        from deployment.inference_onnx import list_available_classes

        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            list_available_classes(str(nonexistent))


# =============================================================================
# Tests de normalización de mapas de anomalía
# =============================================================================


class TestAnomalyMapNormalization:
    """Tests para normalización de mapas de anomalía."""

    def test_normalize_anomaly_map_minmax(
        self, sample_anomaly_map, mock_onnx_session, configured_detector_dir
    ):
        """Verifica normalización min-max."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                normalized = detector.normalize_anomaly_map(
                    sample_anomaly_map, method="minmax"
                )

                assert normalized.min() >= 0.0
                assert normalized.max() <= 1.0

    def test_normalize_constant_map(self, mock_onnx_session, configured_detector_dir):
        """Verifica que mapas constantes se manejan correctamente."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                constant_map = np.ones((37, 37), dtype=np.float32) * 0.5
                normalized = detector.normalize_anomaly_map(
                    constant_map, method="minmax"
                )

                # Debería retornar zeros o el mismo valor, sin errores
                assert not np.isnan(normalized).any()


# =============================================================================
# Tests de upsampling
# =============================================================================


class TestUpsampling:
    """Tests para upsampling de mapas de anomalía."""

    def test_upsample_anomaly_map(
        self, sample_anomaly_map, mock_onnx_session, configured_detector_dir
    ):
        """Verifica que upsample produce el tamaño correcto."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                target_size = (256, 256)
                upsampled = detector.upsample_anomaly_map(
                    sample_anomaly_map, target_size
                )

                assert upsampled.shape == target_size


# =============================================================================
# Tests de evaluación
# =============================================================================


class TestEvaluation:
    """Tests para métricas de evaluación."""

    def test_evaluate_perfect_prediction(
        self, mock_onnx_session, configured_detector_dir
    ):
        """Verifica métricas para predicción perfecta."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                # Ground truth y predicción idénticas
                gt = np.zeros((100, 100), dtype=np.float32)
                gt[30:70, 30:70] = 1.0

                pred = gt.copy()

                metrics = detector.evaluate(pred, gt, threshold=0.5)

                assert metrics["IoU"] == 1.0
                assert metrics["Dice"] == 1.0
                assert metrics["Precision"] == 1.0
                assert metrics["Recall"] == 1.0

    def test_evaluate_no_overlap(self, mock_onnx_session, configured_detector_dir):
        """Verifica métricas cuando no hay overlap."""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session):
            with patch("numpy.load") as mock_np_load:
                mock_np_load.return_value = np.random.randn(3000, 1024).astype(
                    np.float32
                )

                from deployment.inference_onnx import MVTecONNXDetector

                detector = MVTecONNXDetector(
                    exported_dir=str(configured_detector_dir), class_name="bottle"
                )

                gt = np.zeros((100, 100), dtype=np.float32)
                gt[0:20, 0:20] = 1.0

                pred = np.zeros((100, 100), dtype=np.float32)
                pred[80:100, 80:100] = 1.0

                metrics = detector.evaluate(pred, gt, threshold=0.5)

                assert metrics["IoU"] == 0.0


# =============================================================================
# Tests de carga de ground truth
# =============================================================================


class TestGroundTruthLoading:
    """Tests para carga de máscaras ground truth."""

    def test_load_ground_truth(self, tmp_path, sample_gt_mask):
        """Verifica carga correcta de GT."""
        from deployment.inference_onnx import MVTecONNXDetector

        gt_path = tmp_path / "gt_mask.png"
        Image.fromarray(sample_gt_mask).save(gt_path)

        loaded = MVTecONNXDetector.load_ground_truth(str(gt_path))

        assert loaded.dtype == np.float32
        assert loaded.max() <= 1.0
        assert loaded.min() >= 0.0

    def test_load_ground_truth_binary(self, tmp_path):
        """Verifica que GT se carga como binario."""
        from deployment.inference_onnx import MVTecONNXDetector

        # Crear máscara con valores intermedios
        gt = np.array([[0, 128, 255]], dtype=np.uint8)
        gt_path = tmp_path / "gt.png"
        Image.fromarray(gt).save(gt_path)

        loaded = MVTecONNXDetector.load_ground_truth(str(gt_path))

        # Valores deben ser 0 o 1
        unique_values = np.unique(loaded)
        assert all(v in [0.0, 1.0] for v in unique_values)

"""
Fixtures compartidos para tests del proyecto MVTec Anomaly Detection.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Agregar src al path para imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Configuración de paths
# =============================================================================


@pytest.fixture
def project_root():
    """Path raíz del proyecto."""
    return PROJECT_ROOT


@pytest.fixture
def exported_dir(project_root):
    """Path al directorio de modelos exportados."""
    return project_root / "src" / "exported"


@pytest.fixture
def sample_data_dir(project_root):
    """Path al directorio de datos de ejemplo."""
    return project_root / "data"


# =============================================================================
# Fixtures de imágenes
# =============================================================================


@pytest.fixture
def sample_rgb_image():
    """Imagen RGB de prueba (518x518 como espera DINOv2)."""
    # Crear imagen con patrón de gradiente para tener variación
    arr = np.zeros((518, 518, 3), dtype=np.uint8)
    for i in range(518):
        for j in range(518):
            arr[i, j, 0] = i % 256  # R
            arr[i, j, 1] = j % 256  # G
            arr[i, j, 2] = (i + j) % 256  # B
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_small_image():
    """Imagen RGB pequeña de prueba (256x256)."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_grayscale_image():
    """Imagen en escala de grises."""
    arr = np.random.randint(0, 255, (518, 518), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_anomaly_map():
    """Mapa de anomalía de prueba (37x37 como DINOv2 patches)."""
    # Crear patrón con centro más alto
    arr = np.zeros((37, 37), dtype=np.float32)
    center = 18
    for i in range(37):
        for j in range(37):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            arr[i, j] = max(0, 1 - dist / center)
    return arr


@pytest.fixture
def sample_gt_mask():
    """Máscara ground truth de prueba."""
    arr = np.zeros((256, 256), dtype=np.uint8)
    # Región anómala en el centro
    arr[100:150, 100:150] = 255
    return arr


@pytest.fixture
def sample_binary_gt():
    """Máscara ground truth binaria normalizada."""
    arr = np.zeros((256, 256), dtype=np.float32)
    arr[100:150, 100:150] = 1.0
    return arr


# =============================================================================
# Fixtures de arrays numpy
# =============================================================================


@pytest.fixture
def sample_patch_embeddings():
    """Embeddings de patches de prueba (1369 patches x 1024 dims para DINOv2-large)."""
    return np.random.randn(1369, 1024).astype(np.float32)


@pytest.fixture
def sample_memory_bank():
    """Memory bank de prueba (3000 embeddings x 1024 dims)."""
    return np.random.randn(3000, 1024).astype(np.float32)


@pytest.fixture
def sample_cls_tokens():
    """CLS tokens de prueba para varias imágenes."""
    return np.random.randn(100, 1024).astype(np.float32)


# =============================================================================
# Fixtures de configuración
# =============================================================================


@pytest.fixture
def mvtec_classes():
    """Lista de clases MVTec AD."""
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]


@pytest.fixture
def default_threshold():
    """Umbral por defecto para detección de anomalías."""
    return 0.6


@pytest.fixture
def default_k():
    """K por defecto para k-NN."""
    return 1


# =============================================================================
# Fixtures de mocks
# =============================================================================


@pytest.fixture
def mock_onnx_session():
    """Mock de sesión ONNX Runtime."""
    session = MagicMock()
    session.get_inputs.return_value = [
        MagicMock(name="pixel_values", shape=[1, 3, 518, 518])
    ]
    session.get_outputs.return_value = [
        MagicMock(name="patch_embeddings", shape=[1, 1369, 1024])
    ]

    # Mock de run que retorna embeddings
    def mock_run(output_names, input_feed):
        batch_size = input_feed["pixel_values"].shape[0]
        return [np.random.randn(batch_size, 1369, 1024).astype(np.float32)]

    session.run.side_effect = mock_run
    return session


@pytest.fixture
def mock_class_retriever():
    """Mock de ClassRetriever."""
    retriever = MagicMock()
    retriever.classes = ["bottle", "cable", "capsule"]
    retriever.identify_class.return_value = (
        "bottle",
        0.95,
        {"bottle": 0.95, "cable": 0.03, "capsule": 0.02},
    )
    return retriever


# =============================================================================
# Fixtures de archivos temporales
# =============================================================================


@pytest.fixture
def temp_image_file(tmp_path, sample_rgb_image):
    """Archivo de imagen temporal."""
    path = tmp_path / "test_image.png"
    sample_rgb_image.save(path)
    return path


@pytest.fixture
def temp_export_dir(tmp_path, sample_memory_bank, sample_cls_tokens):
    """Directorio temporal con estructura de exportación simulada."""
    export_dir = tmp_path / "exported"

    # Crear estructura para clase 'bottle'
    bottle_dir = export_dir / "bottle"
    bottle_dir.mkdir(parents=True)

    # Guardar memory bank y metadata
    np.save(bottle_dir / "memory_bank.npy", sample_memory_bank)
    np.save(bottle_dir / "cls_tokens.npy", sample_cls_tokens)

    # Metadata
    import json

    metadata = {
        "class_name": "bottle",
        "num_train_images": 100,
        "memory_bank_size": len(sample_memory_bank),
        "k": 1,
    }
    with open(bottle_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return export_dir


@pytest.fixture
def configured_detector_dir(tmp_path, sample_memory_bank):
    """
    Directorio temporal completamente configurado para MVTecONNXDetector.

    Incluye:
    - Archivo ONNX (touch)
    - detector_config.json
    - dinov2_feature_extractor_metadata.json
    - memory_bank.npy

    Uso: Elimina la repetición de setup en test_inference_onnx.py
    """
    import json

    export_dir = tmp_path / "exported"
    export_dir.mkdir(parents=True)

    # Crear modelo ONNX principal (placeholder)
    model_path = export_dir / "dinov2_feature_extractor.onnx"
    model_path.touch()

    # Metadata global del modelo
    metadata = {
        "input_height": 518,
        "input_width": 518,
        "patch_size": 14,
        "n_patches_h": 37,
        "n_patches_w": 37,
    }
    with open(export_dir / "dinov2_feature_extractor_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Configurar clase 'bottle'
    bottle_dir = export_dir / "bottle"
    bottle_dir.mkdir(parents=True)

    # Guardar memory bank
    np.save(bottle_dir / "memory_bank.npy", sample_memory_bank)

    # Config del detector
    config = {"k": 1, "defect_types": []}
    with open(bottle_dir / "detector_config.json", "w") as f:
        json.dump(config, f)

    return export_dir


# =============================================================================
# Markers y configuración
# =============================================================================


def pytest_configure(config):
    """Configuración adicional de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "requires_models: marks tests that require exported models"
    )

"""
Exportación del modelo DINOv2 a formato ONNX para deployment.

Este módulo permite exportar el extractor de features DINOv2 a ONNX,
junto con los memory banks precomputados por clase para detección de anomalías
en el dataset MVTec AD.

Estructura esperada del dataset MVTec AD:
    data/raw/
    ├── bottle/
    │   ├── train/good/       # Imágenes normales para memory bank
    │   ├── test/             # Imágenes de test
    │   └── ground_truth/     # Máscaras GT
    ├── cable/
    │   ├── train/good/
    │   ├── test/
    │   └── ground_truth/
    └── ...

Uso típico:
    # Exportar modelo + memory banks para todas las clases
    python export_onnx.py \\
        --model_path /path/to/dinov2 \\
        --data_root /path/to/mvtec/data/raw \\
        --output_dir ./exported \\
        --all_classes

    # Exportar solo una clase
    python export_onnx.py \\
        --model_path /path/to/dinov2 \\
        --data_root /path/to/mvtec/data/raw \\
        --output_dir ./exported \\
        --classes bottle cable
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Agregar path del proyecto para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# =============================================================================
# CONFIGURACIÓN DEL DATASET MVTEC AD
# =============================================================================

# Clases disponibles en MVTec AD
MVTEC_CLASSES = [
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

# Ruta por defecto del dataset
DEFAULT_DATA_ROOT = "/home/bllancao/Portafolio/mvtec_anomaly_detection/data/raw"


def get_train_images_path(data_root: str, class_name: str) -> str:
    """Retorna la ruta a las imágenes de entrenamiento (normales) de una clase."""
    return os.path.join(data_root, class_name, "train", "good")


def get_test_images_path(
    data_root: str, class_name: str, defect_type: str = None
) -> str:
    """Retorna la ruta a las imágenes de test de una clase."""
    if defect_type:
        return os.path.join(data_root, class_name, "test", defect_type)
    return os.path.join(data_root, class_name, "test")


def get_ground_truth_path(
    data_root: str, class_name: str, defect_type: str = None
) -> str:
    """Retorna la ruta al ground truth de una clase."""
    if defect_type:
        return os.path.join(data_root, class_name, "ground_truth", defect_type)
    return os.path.join(data_root, class_name, "ground_truth")


def list_defect_types(data_root: str, class_name: str) -> List[str]:
    """Lista los tipos de defectos disponibles para una clase."""
    test_path = get_test_images_path(data_root, class_name)
    if os.path.exists(test_path):
        return [
            d
            for d in os.listdir(test_path)
            if os.path.isdir(os.path.join(test_path, d)) and d != "good"
        ]
    return []


# =============================================================================
# WRAPPER PARA EXPORTACIÓN ONNX
# =============================================================================


class DINOv2ForONNX(nn.Module):
    """
    Wrapper del modelo DINOv2 optimizado para exportación ONNX.

    Extrae patch embeddings normalizados (sin token CLS) de una imagen.
    Compatible con el flujo de detección de anomalías de eval.py.

    Args:
        model_path: Ruta al modelo DINOv2 (local o HuggingFace hub)
        layer_idx: Índice de la capa para extracción de features (-1 = última)
        normalize: Si True, normaliza embeddings L2
    """

    def __init__(self, model_path: str, layer_idx: int = -1, normalize: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.layer_idx = layer_idx
        self.normalize = normalize

        # Configuración del modelo para referencia
        self.patch_size = 14  # DINOv2 usa patches de 14x14
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para extracción de features.

        Args:
            pixel_values: Tensor de imagen preprocesada [B, 3, H, W]

        Returns:
            patch_embeddings: Tensor [B, num_patches, hidden_dim]
        """
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        if self.layer_idx == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.layer_idx]

        # Remover token CLS (primer token)
        patch_embeddings = hidden_states[:, 1:, :]

        if self.normalize:
            patch_embeddings = torch.nn.functional.normalize(
                patch_embeddings, p=2, dim=-1
            )

        return patch_embeddings


# =============================================================================
# WRAPPER PARA CLS TOKEN (IMAGE RETRIEVAL)
# =============================================================================


class DINOv2CLSForONNX(nn.Module):
    """
    Wrapper para extracción del CLS token de DINOv2.

    El CLS token captura la representación global de la imagen,
    ideal para clasificación e image retrieval.

    Args:
        model_path: Ruta al modelo DINOv2
        layer_idx: Índice de la capa (-1 = última)
        normalize: Si True, normaliza L2
    """

    def __init__(self, model_path: str, layer_idx: int = -1, normalize: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.layer_idx = layer_idx
        self.normalize = normalize
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para extracción de CLS token.

        Args:
            pixel_values: Tensor [B, 3, H, W]

        Returns:
            cls_token: Tensor [B, hidden_dim]
        """
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        if self.layer_idx == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[self.layer_idx]

        # CLS token es el primer token
        cls_token = hidden_states[:, 0, :]  # [B, hidden_dim]

        if self.normalize:
            cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=-1)

        return cls_token


def export_cls_model_to_onnx(
    model_path: str,
    output_path: str,
    layer_idx: int = -1,
    input_height: int = 518,
    input_width: int = 518,
    opset_version: int = 17,
    verbose: bool = True,
) -> str:
    """
    Exporta modelo para extracción de CLS token a ONNX.

    Este modelo se usa para image retrieval / clasificación de clase.
    """
    if verbose:
        print(f"📦 Exportando DINOv2 CLS Extractor a ONNX...")

    model = DINOv2CLSForONNX(model_path, layer_idx, normalize=True)
    model.eval()

    dummy_input = torch.randn(1, 3, input_height, input_width)

    with torch.no_grad():
        test_output = model(dummy_input)
        hidden_dim = test_output.shape[-1]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["cls_token"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "cls_token": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ CLS Extractor exportado: {output_path} ({file_size_mb:.1f} MB)")

    # Guardar metadatos
    metadata = {
        "model_path": model_path,
        "layer_idx": layer_idx,
        "input_height": input_height,
        "input_width": input_width,
        "hidden_dim": hidden_dim,
        "type": "cls_extractor",
    }

    metadata_path = output_path.replace(".onnx", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def extract_cls_tokens_for_class(
    extractor, data_root: str, class_name: str, verbose: bool = True
) -> torch.Tensor:
    """
    Extrae CLS tokens de todas las imágenes de entrenamiento de una clase.

    Estos CLS tokens se usan para image retrieval (identificar clase).

    Args:
        extractor: DINOv2FeatureExtractor con acceso al modelo
        data_root: Ruta raíz del dataset
        class_name: Nombre de la clase
        verbose: Si True, muestra progreso

    Returns:
        cls_tokens: Tensor [N_images, hidden_dim]
    """
    train_path = get_train_images_path(data_root, class_name)
    train_images = load_images_from_folder(train_path)

    if verbose:
        print(f"   Extrayendo CLS tokens de {len(train_images)} imágenes...")

    all_cls_tokens = []

    for i, img in enumerate(train_images):
        # Usar el modelo interno del extractor para obtener CLS
        inputs = extractor.processor(images=img, return_tensors="pt", do_rescale=True)
        inputs = {k: v.to(extractor.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = extractor.model(**inputs, output_hidden_states=True)

            if extractor.layer_idx == -1:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[extractor.layer_idx]

            # CLS token (primer token) normalizado
            cls_token = hidden_states[:, 0, :]
            cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=-1)

            all_cls_tokens.append(cls_token.cpu())

    cls_tokens = torch.cat(all_cls_tokens, dim=0)  # [N, hidden_dim]

    if verbose:
        print(f"   CLS tokens: {cls_tokens.shape}")

    return cls_tokens


def export_cls_tokens(
    cls_tokens: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    """
    Guarda CLS tokens para una clase.
    """
    if isinstance(cls_tokens, torch.Tensor):
        cls_tokens_np = cls_tokens.cpu().numpy()
    else:
        cls_tokens_np = cls_tokens

    np.save(output_path, cls_tokens_np)

    if verbose:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"   💾 CLS tokens: {output_path} ({size_kb:.1f} KB)")

    return output_path


# =============================================================================
# FUNCIONES DE CARGA DE IMÁGENES
# =============================================================================


def load_images_from_folder(
    folder_path: str,
    n_images: int = None,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp"),
) -> List[Image.Image]:
    """
    Carga N imágenes de una carpeta.

    Args:
        folder_path: Ruta a la carpeta con imágenes
        n_images: Número de imágenes a cargar (None = todas)
        extensions: Extensiones de archivo válidas

    Returns:
        Lista de imágenes PIL
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ Carpeta no existe: {folder_path}")
        return []

    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    )

    if n_images is not None:
        image_files = image_files[:n_images]

    images = []
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    return images


# =============================================================================
# FUNCIONES DE EXPORTACIÓN
# =============================================================================


def export_dinov2_to_onnx(
    model_path: str,
    output_path: str,
    layer_idx: int = -1,
    normalize: bool = True,
    input_height: int = 518,
    input_width: int = 518,
    opset_version: int = 17,
    verbose: bool = True,
) -> str:
    """
    Exporta el modelo DINOv2 a formato ONNX.

    Args:
        model_path: Ruta al modelo DINOv2 (local o desde HuggingFace)
        output_path: Ruta donde guardar el archivo .onnx
        layer_idx: Índice de la capa para extracción (-1 = última)
        normalize: Si True, incluye normalización L2 en el modelo
        input_height: Altura de imagen de entrada (debe ser divisible por 14)
        input_width: Ancho de imagen de entrada (debe ser divisible por 14)
        opset_version: Versión del opset ONNX
        verbose: Si True, muestra progreso

    Returns:
        output_path: Ruta al archivo ONNX guardado
    """
    if verbose:
        print(f"📦 Exportando DINOv2 a ONNX...")
        print(f"   Modelo fuente: {model_path}")
        print(f"   Capa: {layer_idx}, Normalización: {normalize}")
        print(f"   Tamaño entrada: {input_height}x{input_width}")

    patch_size = 14
    assert input_height % patch_size == 0, f"Altura debe ser divisible por {patch_size}"
    assert input_width % patch_size == 0, f"Ancho debe ser divisible por {patch_size}"

    n_patches_h = input_height // patch_size
    n_patches_w = input_width // patch_size
    n_patches_total = n_patches_h * n_patches_w

    model = DINOv2ForONNX(
        model_path=model_path, layer_idx=layer_idx, normalize=normalize
    )
    model.eval()

    dummy_input = torch.randn(1, 3, input_height, input_width)

    with torch.no_grad():
        test_output = model(dummy_input)
        hidden_dim = test_output.shape[-1]

    if verbose:
        print(f"   Patches: {n_patches_h}x{n_patches_w} = {n_patches_total}")
        print(f"   Hidden dim: {hidden_dim}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["patch_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "patch_embeddings": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Modelo exportado: {output_path} ({file_size_mb:.1f} MB)")

    metadata = {
        "model_path": model_path,
        "layer_idx": layer_idx,
        "normalize": normalize,
        "input_height": input_height,
        "input_width": input_width,
        "patch_size": patch_size,
        "n_patches_h": n_patches_h,
        "n_patches_w": n_patches_w,
        "hidden_dim": hidden_dim,
        "opset_version": opset_version,
    }

    metadata_path = output_path.replace(".onnx", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"📋 Metadatos guardados: {metadata_path}")

    return output_path


def build_memory_bank_for_class(
    extractor,
    data_root: str,
    class_name: str,
    coreset_ratio: float = 1.0,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Construye el memory bank para una clase específica usando sus imágenes de train.

    Args:
        extractor: DINOv2FeatureExtractor o modelo similar
        data_root: Ruta raíz del dataset MVTec AD
        class_name: Nombre de la clase (e.g., 'bottle')
        coreset_ratio: Ratio de subsampling del memory bank
        verbose: Si True, muestra progreso

    Returns:
        memory_bank: Tensor [N, hidden_dim]
    """
    train_path = get_train_images_path(data_root, class_name)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontró: {train_path}")

    if verbose:
        print(f"   Cargando imágenes de: {train_path}")

    # Cargar imágenes de entrenamiento
    train_images = load_images_from_folder(train_path)

    if len(train_images) == 0:
        raise ValueError(f"No se encontraron imágenes en {train_path}")

    if verbose:
        print(f"   Procesando {len(train_images)} imágenes...")

    # Extraer patches
    all_patches = []
    for i, img in enumerate(train_images):
        patches = extractor.extract_patches(img)  # [N, D]
        all_patches.append(patches)

        if verbose and (i + 1) % 20 == 0:
            print(f"   Procesadas {i + 1}/{len(train_images)} imágenes")

    memory_bank = torch.cat(all_patches, dim=0)

    # Coreset subsampling
    if coreset_ratio < 1.0:
        n_samples = int(len(memory_bank) * coreset_ratio)
        indices = torch.randperm(len(memory_bank))[:n_samples]
        memory_bank = memory_bank[indices]

    if verbose:
        print(f"   Memory bank: {memory_bank.shape[0]} patches")

    return memory_bank


def export_memory_bank(
    memory_bank: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    """
    Guarda el memory bank precomputado para usar en inferencia.

    Args:
        memory_bank: Tensor de embeddings [N, hidden_dim]
        output_path: Ruta donde guardar (sin extensión)
        metadata: Diccionario con información adicional
        verbose: Si True, muestra progreso

    Returns:
        output_path: Ruta base de los archivos guardados
    """
    if isinstance(memory_bank, torch.Tensor):
        memory_bank_np = memory_bank.cpu().numpy()
    else:
        memory_bank_np = memory_bank

    npy_path = output_path + ".npy"
    np.save(npy_path, memory_bank_np)

    if verbose:
        size_mb = os.path.getsize(npy_path) / (1024 * 1024)
        print(f"   💾 Guardado: {npy_path} ({size_mb:.1f} MB)")

    mb_metadata = {
        "shape": list(memory_bank_np.shape),
        "dtype": str(memory_bank_np.dtype),
        "n_patches": memory_bank_np.shape[0],
        "hidden_dim": (
            memory_bank_np.shape[1] if len(memory_bank_np.shape) > 1 else None
        ),
        **(metadata or {}),
    }

    json_path = output_path + "_metadata.json"
    with open(json_path, "w") as f:
        json.dump(mb_metadata, f, indent=2)

    return output_path


def verify_onnx_export(
    onnx_path: str,
    model_path: str,
    test_image: Optional[Image.Image] = None,
    tolerance: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """
    Verifica que el modelo ONNX produce los mismos resultados que PyTorch.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("⚠️ Instala onnx y onnxruntime para verificar")
        return False

    if verbose:
        print("🔍 Verificando exportación ONNX...")

    metadata_path = onnx_path.replace(".onnx", "_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    input_h = metadata["input_height"]
    input_w = metadata["input_width"]
    layer_idx = metadata["layer_idx"]
    normalize = metadata["normalize"]

    processor = AutoImageProcessor.from_pretrained(model_path)
    pytorch_model = DINOv2ForONNX(model_path, layer_idx, normalize)
    pytorch_model.eval()

    if test_image is not None:
        inputs = processor(images=test_image, return_tensors="pt", do_rescale=True)
        input_tensor = inputs["pixel_values"]
    else:
        input_tensor = torch.randn(1, 3, input_h, input_w)

    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor).numpy()

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    onnx_output = ort_session.run(None, {"pixel_values": input_tensor.numpy()})[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    is_valid = max_diff < tolerance

    if verbose:
        status = "✅ PASÓ" if is_valid else "❌ FALLÓ"
        print(f"   {status} | Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    return is_valid


# =============================================================================
# PIPELINE COMPLETO DE EXPORTACIÓN POR CLASE
# =============================================================================


def export_for_class(
    model_path: str,
    data_root: str,
    class_name: str,
    output_dir: str,
    layer_idx: int = -1,
    k: int = 1,
    coreset_ratio: float = 0.1,
    input_size: Tuple[int, int] = (518, 518),
    extractor=None,
    export_cls: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Exporta el memory bank y CLS tokens para una clase específica del dataset MVTec AD.

    Args:
        model_path: Ruta al modelo DINOv2
        data_root: Ruta raíz del dataset (e.g., data/raw)
        class_name: Nombre de la clase (e.g., 'bottle')
        output_dir: Directorio de salida
        layer_idx: Capa para extracción de features
        k: K para k-NN scoring
        coreset_ratio: Ratio de subsampling del memory bank
        input_size: (height, width) de entrada
        extractor: Extractor ya cargado (opcional, para reusar)
        export_cls: Si True, exporta CLS tokens para image retrieval
        verbose: Si True, muestra progreso

    Returns:
        paths: Diccionario con rutas a los archivos exportados
    """
    # Verificar que la clase existe
    class_path = os.path.join(data_root, class_name)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Clase no encontrada: {class_path}")

    # Crear directorio de salida para esta clase
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    if verbose:
        print(f"\n📦 Exportando clase: {class_name}")
        print("=" * 50)

    # Crear extractor si no se proporciona
    if extractor is None:
        from evaluation.eval import DINOv2FeatureExtractor

        extractor = DINOv2FeatureExtractor(model_path=model_path, layer_idx=layer_idx)

    # Construir memory bank desde imágenes de train
    memory_bank = build_memory_bank_for_class(
        extractor=extractor,
        data_root=data_root,
        class_name=class_name,
        coreset_ratio=coreset_ratio,
        verbose=verbose,
    )

    # Contar imágenes de train para metadata
    train_path = get_train_images_path(data_root, class_name)
    n_train_images = len(
        [
            f
            for f in os.listdir(train_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    # Exportar memory bank
    memory_bank_path = os.path.join(class_output_dir, "memory_bank")
    export_memory_bank(
        memory_bank,
        memory_bank_path,
        metadata={
            "class_name": class_name,
            "k": k,
            "coreset_ratio": coreset_ratio,
            "n_training_images": n_train_images,
            "model_path": model_path,
            "layer_idx": layer_idx,
            "defect_types": list_defect_types(data_root, class_name),
        },
        verbose=verbose,
    )

    result_paths = {
        "memory_bank": memory_bank_path + ".npy",
    }

    # Exportar CLS tokens para image retrieval
    if export_cls:
        cls_tokens = extract_cls_tokens_for_class(
            extractor=extractor,
            data_root=data_root,
            class_name=class_name,
            verbose=verbose,
        )

        cls_tokens_path = os.path.join(class_output_dir, "cls_tokens.npy")
        export_cls_tokens(cls_tokens, cls_tokens_path, verbose=verbose)
        result_paths["cls_tokens"] = cls_tokens_path

    # Guardar configuración del detector para esta clase
    config = {
        "class_name": class_name,
        "model_onnx": "../dinov2_feature_extractor.onnx",
        "memory_bank": "memory_bank.npy",
        "cls_tokens": "cls_tokens.npy" if export_cls else None,
        "k": k,
        "layer_idx": layer_idx,
        "input_size": list(input_size),
        "patch_size": 14,
        "coreset_ratio": coreset_ratio,
        "defect_types": list_defect_types(data_root, class_name),
    }

    config_path = os.path.join(class_output_dir, "detector_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"   ⚙️ Config: {config_path}")

    result_paths["config"] = config_path
    return result_paths


def export_all_classes(
    model_path: str,
    data_root: str = DEFAULT_DATA_ROOT,
    output_dir: str = "./exported",
    classes: Optional[List[str]] = None,
    layer_idx: int = -1,
    k: int = 1,
    coreset_ratio: float = 0.1,
    input_size: Tuple[int, int] = (518, 518),
    verify: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, str]]:
    """
    Exporta el modelo ONNX y memory banks para todas (o algunas) clases de MVTec AD.

    El modelo ONNX se comparte entre todas las clases, pero cada clase tiene
    su propio memory bank construido con sus imágenes de train.

    Args:
        model_path: Ruta al modelo DINOv2
        data_root: Ruta raíz del dataset MVTec AD
        output_dir: Directorio de salida
        classes: Lista de clases a exportar (None = todas)
        layer_idx: Capa para extracción de features
        k: K para k-NN scoring
        coreset_ratio: Ratio de subsampling del memory bank
        input_size: (height, width) de entrada
        verify: Si True, verifica la exportación ONNX
        verbose: Si True, muestra progreso

    Returns:
        all_paths: Diccionario con rutas por clase
    """
    from evaluation.eval import DINOv2FeatureExtractor

    os.makedirs(output_dir, exist_ok=True)

    # Determinar clases a exportar
    if classes is None:
        # Detectar clases disponibles
        classes = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d in MVTEC_CLASSES
        ]
        classes = sorted(classes)

    if verbose:
        print("=" * 60)
        print("🚀 EXPORTACIÓN MVTEC AD - DETECTOR DE ANOMALÍAS")
        print("=" * 60)
        print(f"Dataset: {data_root}")
        print(f"Clases: {', '.join(classes)}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # 1. Exportar modelo ONNX (compartido para todas las clases)
    onnx_path = os.path.join(output_dir, "dinov2_feature_extractor.onnx")
    if not os.path.exists(onnx_path):
        export_dinov2_to_onnx(
            model_path=model_path,
            output_path=onnx_path,
            layer_idx=layer_idx,
            input_height=input_size[0],
            input_width=input_size[1],
            verbose=verbose,
        )

        if verify:
            verify_onnx_export(onnx_path, model_path, verbose=verbose)
    else:
        if verbose:
            print(f"✅ Modelo ONNX ya existe: {onnx_path}")

    # 2. Crear extractor para reusar
    if verbose:
        print(f"\n📚 Cargando extractor DINOv2...")

    extractor = DINOv2FeatureExtractor(model_path=model_path, layer_idx=layer_idx)

    # 3. Exportar memory bank para cada clase
    all_paths = {"model_onnx": onnx_path}

    for class_name in classes:
        try:
            paths = export_for_class(
                model_path=model_path,
                data_root=data_root,
                class_name=class_name,
                output_dir=output_dir,
                layer_idx=layer_idx,
                k=k,
                coreset_ratio=coreset_ratio,
                input_size=input_size,
                extractor=extractor,
                verbose=verbose,
            )
            all_paths[class_name] = paths
        except Exception as e:
            print(f"❌ Error en clase {class_name}: {e}")
            continue

    # 4. Guardar configuración global
    global_config = {
        "model_onnx": "dinov2_feature_extractor.onnx",
        "classes": list(all_paths.keys()),
        "layer_idx": layer_idx,
        "k": k,
        "input_size": list(input_size),
        "patch_size": 14,
        "coreset_ratio": coreset_ratio,
        "data_root": data_root,
    }

    global_config_path = os.path.join(output_dir, "global_config.json")
    with open(global_config_path, "w") as f:
        json.dump(global_config, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("✅ EXPORTACIÓN COMPLETADA")
        print("=" * 60)
        print(f"Clases exportadas: {len([k for k in all_paths if k != 'model_onnx'])}")
        print(f"Archivos en: {output_dir}")
        print("\nEstructura:")
        print(f"  {output_dir}/")
        print(f"  ├── dinov2_feature_extractor.onnx  (modelo compartido)")
        print(f"  ├── global_config.json")
        for class_name in classes:
            if class_name in all_paths:
                print(f"  ├── {class_name}/")
                print(f"  │   ├── memory_bank.npy        (patch embeddings)")
                print(f"  │   ├── cls_tokens.npy         (CLS tokens retrieval)")
                print(f"  │   └── detector_config.json")

    return all_paths


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Exportar modelo DINOv2 a ONNX para detección de anomalías MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Exportar todas las clases
  python export_onnx.py \\
      --model_path facebook/dinov2-base \\
      --data_root /path/to/mvtec/data/raw \\
      --output_dir ./exported \\
      --all_classes

  # Exportar clases específicas
  python export_onnx.py \\
      --model_path /path/to/dinov2 \\
      --data_root /path/to/mvtec/data/raw \\
      --output_dir ./exported \\
      --classes bottle cable capsule

  # Solo exportar el modelo ONNX (sin memory banks)
  python export_onnx.py \\
      --model_path facebook/dinov2-base \\
      --output_path model.onnx
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo DINOv2 (local o HuggingFace hub)",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"Ruta raíz del dataset MVTec AD (default: {DEFAULT_DATA_ROOT})",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="dinov2.onnx",
        help="Ruta de salida para modelo ONNX (modo simple)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exported",
        help="Directorio de salida para exportación completa",
    )

    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Exportar memory banks para todas las clases de MVTec AD",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        choices=MVTEC_CLASSES,
        help="Clases específicas a exportar",
    )

    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Índice de capa para extracción de features (-1 = última)",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[518, 518],
        help="Tamaño de entrada (height width)",
    )

    parser.add_argument("--k", type=int, default=1, help="K para k-NN scoring")

    parser.add_argument(
        "--coreset_ratio",
        type=float,
        default=0.1,
        help="Ratio de subsampling para memory bank (0.1 = 10%%)",
    )

    parser.add_argument(
        "--verify", action="store_true", help="Verificar exportación ONNX"
    )

    parser.add_argument("--opset", type=int, default=17, help="Versión del opset ONNX")

    args = parser.parse_args()

    if args.all_classes or args.classes:
        # Modo completo: exportar modelo + memory banks por clase
        export_all_classes(
            model_path=args.model_path,
            data_root=args.data_root,
            output_dir=args.output_dir,
            classes=args.classes,
            layer_idx=args.layer_idx,
            k=args.k,
            coreset_ratio=args.coreset_ratio,
            input_size=tuple(args.input_size),
            verify=args.verify,
        )
    else:
        # Modo simple: solo exportar modelo ONNX
        export_dinov2_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            layer_idx=args.layer_idx,
            input_height=args.input_size[0],
            input_width=args.input_size[1],
            opset_version=args.opset,
        )

        if args.verify:
            verify_onnx_export(args.output_path, args.model_path)


if __name__ == "__main__":
    main()

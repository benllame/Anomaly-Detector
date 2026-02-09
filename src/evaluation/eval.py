import warnings
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

# =============================================================================
# FUNCIONES DE NORMALIZACIÓN Y RESIZE
# =============================================================================


def normalize_anomaly_map(
    anomaly_map: np.ndarray,
    method: str = "minmax",
    clip_percentile: Optional[Tuple[float, float]] = None,
    robust_percentile: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Normaliza un mapa de anomalías al rango [0, 1].

    Args:
        anomaly_map: Mapa de anomalías [H, W] con valores continuos
        method: Método de normalización:
            - 'minmax': Normalización min-max estándar
            - 'robust': Normalización robusta usando percentiles (casos de outliers)
        clip_percentile: Tupla (min_percentile, max_percentile) para recortar
                        valores extremos antes de normalizar. Ej: (1, 99)
        robust_percentile: Percentiles a usar para normalización robusta.
                          Por defecto (2, 98)

    Returns:
        normalized_map: Mapa normalizado en [0, 1]

    Example:
        >>> amap = normalize_anomaly_map(raw_scores, method='robust')
        >>> amap = normalize_anomaly_map(raw_scores, clip_percentile=(1, 99))
    """
    amap = anomaly_map.astype(np.float32).copy()

    # Paso 1: Recortar valores extremos si se especifica
    if clip_percentile is not None:
        p_min, p_max = clip_percentile
        v_min = np.percentile(amap, p_min)
        v_max = np.percentile(amap, p_max)
        amap = np.clip(amap, v_min, v_max)

    # Paso 2: Normalizar según el método
    if method == "minmax":
        # Normalización min-max estándar
        amap_min = amap.min()
        amap_max = amap.max()

        if amap_max > amap_min:
            normalized = (amap - amap_min) / (amap_max - amap_min)
        else:
            # Mapa uniforme, retornar ceros
            normalized = np.zeros_like(amap)
            warnings.warn("Mapa de anomalía tiene valores uniformes (min == max)")

    elif method == "robust":
        # Normalización robusta usando percentiles
        p_low, p_high = robust_percentile
        v_low = np.percentile(amap, p_low)
        v_high = np.percentile(amap, p_high)

        if v_high > v_low:
            normalized = (amap - v_low) / (v_high - v_low)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(amap)
            warnings.warn("Percentiles robustos iguales, usando normalización min-max")
            return normalize_anomaly_map(amap, method="minmax")
    else:
        raise ValueError(
            f"Método de normalización no válido: {method}. Usar 'minmax' o 'robust'"
        )

    return normalized


def resize_anomaly_map(
    anomaly_map: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Redimensiona un mapa de anomalía a un tamaño objetivo.

    Args:
        anomaly_map: Mapa de anomalía [H, W]
        target_size: Tamaño objetivo (height, width), el cual corresponde a la imagen original
        interpolation: Método de interpolación:
            - 'bilinear': Interpolación bilinear
            - 'nearest': Vecino más cercano
            - 'cubic': Interpolación bicúbica

    Returns:
        resized_map: Mapa redimensionado [target_H, target_W]
    """
    interp_methods = {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }

    if interpolation not in interp_methods:
        raise ValueError(
            f"Interpolación no válida: {interpolation}. Usar: {list(interp_methods.keys())}"
        )

    # cv2.resize espera (width, height)
    target_wh = (target_size[1], target_size[0])

    resized = cv2.resize(
        anomaly_map.astype(np.float32),
        target_wh,
        interpolation=interp_methods[interpolation],
    )

    return resized


# =============================================================================
# CONFIGURACIÓN DEL MODELO
# =============================================================================


class DINOv2FeatureExtractor:
    """
    Extractor de features con DINOv2 y capa configurable.

    Args:
        model_path: Ruta al modelo DINOv2
        layer_idx: Índice de la capa a usar (-1 = última, 0 = embedding, etc.)
        device: Dispositivo para inferencia ('cuda' o 'cpu')
    """

    def __init__(self, model_path: str, layer_idx: int = -1, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.layer_idx = layer_idx

        # DINOv2-base usa patches de 14x14
        self.patch_size = 14

    def extract_patches(
        self, image: Image.Image, normalize: bool = True
    ) -> torch.Tensor:
        """
        Extrae embeddings de patches de una imagen.

        Args:
            image: Imagen PIL
            normalize: Si True, normaliza los embeddings (L2)

        Returns:
            patches: Tensor [num_patches, hidden_dim]
        """
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # Seleccionar capa específica
            if self.layer_idx == -1:
                # Última capa (last_hidden_state)
                hidden_states = outputs.last_hidden_state
            else:
                # Capa específica de hidden_states
                hidden_states = outputs.hidden_states[self.layer_idx]

            # Remover token CLS (primer token)
            patches = hidden_states[:, 1:, :].squeeze(0)  # [num_patches, hidden_dim]

            if normalize:
                patches = F.normalize(patches, p=2, dim=-1)

        return patches

    def extract_patches_batch(
        self, images: List[Image.Image], normalize: bool = True
    ) -> torch.Tensor:
        """
        Extrae patches de múltiples imágenes en batch.

        Returns:
            patches: Tensor [batch, num_patches, hidden_dim]
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=True).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            if self.layer_idx == -1:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[self.layer_idx]

            patches = hidden_states[:, 1:, :]  # [batch, num_patches, hidden_dim]

            if normalize:
                patches = F.normalize(patches, p=2, dim=-1)

        return patches

    def get_grid_size(self, image: Image.Image) -> Tuple[int, int]:
        """Retorna el tamaño del grid de patches (h, w)."""
        inputs = self.processor(images=image, return_tensors="pt")
        h = inputs["pixel_values"].shape[-2] // self.patch_size
        w = inputs["pixel_values"].shape[-1] // self.patch_size
        return h, w


# =============================================================================
# MÉTODO 1: DENSE MATCHING (POSICIONAL)
# =============================================================================


class DenseMatchingDetector:
    """
    Detector de anomalías por Dense Matching (correspondencia posicional 1:1).

    En el dataset MVTec AD, las imágenes están bien alineadas, por tanto
    se puede usar la correspondencia posicional 1:1 para detectar anomalías.

    Compara cada patch de la imagen test con el patch en la misma posición
    de la imagen de referencia.

    Args:
        extractor: Instancia de DINOv2FeatureExtractor
    """

    def __init__(self, extractor: DINOv2FeatureExtractor):
        self.extractor = extractor

    def compute_anomaly_map(
        self,
        test_image: Image.Image,
        reference_image: Image.Image,
        smooth_sigma: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula mapa de anomalía por comparación posicional 1:1.

        Args:
            test_image: Imagen a evaluar
            reference_image: Imagen de referencia (sin defectos)
            smooth_sigma: Sigma para suavizado Gaussiano

        Returns:
            anomaly_map: Mapa de anomalía [H, W] sin suavizar
            anomaly_map_smooth: Mapa de anomalía [H, W] suavizado
        """
        # Extraer patches de ambas imágenes
        test_patches = self.extractor.extract_patches(test_image)  # [N, D]
        ref_patches = self.extractor.extract_patches(reference_image)  # [N, D]

        # Similitud coseno por patch (correspondencia posicional)
        cosine_sim = (test_patches * ref_patches).sum(dim=-1)  # [N]

        # Anomalía = 1 - similitud
        anomaly_scores = (1 - cosine_sim).cpu().numpy()

        # Reshape a grid 2D
        h, w = self.extractor.get_grid_size(test_image)
        anomaly_map = anomaly_scores.reshape(h, w)

        # Suavizado
        anomaly_map_smooth = ndimage.gaussian_filter(anomaly_map, sigma=smooth_sigma)

        return anomaly_map, anomaly_map_smooth

    def compute_anomaly_map_multi_ref(
        self,
        test_image: Image.Image,
        reference_images: List[Image.Image],
        aggregation: str = "min",
        smooth_sigma: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula mapa de anomalía contra múltiples referencias.

        Args:
            test_image: Imagen a evaluar
            reference_images: Lista de imágenes de referencia
            aggregation: 'min' (menor distancia) o 'mean' (promedio)
            smooth_sigma: Sigma para suavizado

        Returns:
            anomaly_map, anomaly_map_smooth
        """
        test_patches = self.extractor.extract_patches(test_image)  # [N, D]

        all_scores = []
        for ref_img in reference_images:
            ref_patches = self.extractor.extract_patches(ref_img)
            cosine_sim = (test_patches * ref_patches).sum(dim=-1)
            scores = 1 - cosine_sim
            all_scores.append(scores)

        all_scores = torch.stack(all_scores, dim=0)  # [num_refs, N]

        if aggregation == "min":
            anomaly_scores = all_scores.min(dim=0)[0]
        else:
            anomaly_scores = all_scores.mean(dim=0)

        h, w = self.extractor.get_grid_size(test_image)
        anomaly_map = anomaly_scores.cpu().numpy().reshape(h, w)
        anomaly_map_smooth = ndimage.gaussian_filter(anomaly_map, sigma=smooth_sigma)

        return anomaly_map, anomaly_map_smooth

    def visualize(
        self,
        test_image: Image.Image,
        reference_image: Image.Image,
        anomaly_map: np.ndarray,
        title: str = "Dense Matching - Detección de Anomalías",
    ):
        """Visualiza resultado de detección."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(reference_image)
        axes[0].set_title("Referencia (Sin defectos)")
        axes[0].axis("off")

        axes[1].imshow(test_image)
        axes[1].set_title("Imagen Test")
        axes[1].axis("off")

        axes[2].imshow(test_image)
        im = axes[2].imshow(
            anomaly_map,
            cmap="jet",
            alpha=0.5,
            extent=(0, test_image.width, test_image.height, 0),
        )
        axes[2].set_title("Mapa de Anomalía")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return fig


# =============================================================================
# MÉTODO 2: MEMORY BANK + k-NN (PATCHCORE-STYLE)
# =============================================================================


class MemoryBankDetector:
    """
    Detector de anomalías estilo PatchCore con Memory Bank + k-NN.

    Estado del arte para MVTec AD. Construye un banco de memoria con
    patches de imágenes normales y detecta anomalías buscando patches
    que no tienen vecinos cercanos en el banco.

    Args:
        extractor: Instancia de DINOv2FeatureExtractor
        k: Número de vecinos más cercanos para scoring
        coreset_ratio: Ratio de subsampling del memory bank (1.0 = sin subsampling)
    """

    def __init__(
        self, extractor: DINOv2FeatureExtractor, k: int = 1, coreset_ratio: float = 1.0
    ):
        self.extractor = extractor
        self.k = k
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None

    def build_memory_bank(self, good_images: List[Image.Image], verbose: bool = True):
        """
        Construye el banco de memoria con patches de imágenes sin defectos.

        Args:
            good_images: Lista de imágenes PIL sin defectos (training set)
            verbose: Si True, muestra progreso
        """
        all_patches = []

        for i, img in enumerate(good_images):
            patches = self.extractor.extract_patches(img)  # [N, D]
            all_patches.append(patches)

            if verbose and (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(good_images)} imágenes")

        # Concatenar todos los patches
        self.memory_bank = torch.cat(all_patches, dim=0)  # [total_patches, D]

        # Coreset subsampling (opcional, para reducir memoria)
        if self.coreset_ratio < 1.0:
            n_samples = int(len(self.memory_bank) * self.coreset_ratio)
            indices = torch.randperm(len(self.memory_bank))[:n_samples]
            self.memory_bank = self.memory_bank[indices]

        if verbose:
            print(
                f"Memory Bank construido: {self.memory_bank.shape[0]} patches, "
                f"dim={self.memory_bank.shape[1]}"
            )

    def compute_anomaly_map(
        self, test_image: Image.Image, smooth_sigma: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calcula mapa de anomalía comparando contra memory bank.

        Args:
            test_image: Imagen a evaluar
            smooth_sigma: Sigma para suavizado

        Returns:
            anomaly_map: Mapa [H, W] sin suavizar
            anomaly_map_smooth: Mapa [H, W] suavizado
            image_score: Score de anomalía a nivel de imagen
        """
        if self.memory_bank is None:
            raise RuntimeError("Primero ejecuta build_memory_bank()")

        # Extraer patches de imagen test
        test_patches = self.extractor.extract_patches(test_image)  # [N, D]

        # Calcular similitud con todo el memory bank
        # sim[i, j] = similitud entre test_patch[i] y memory_patch[j]
        sim_matrix = torch.mm(test_patches, self.memory_bank.t())  # [N, memory_size]

        # k-NN: obtener k vecinos más similares
        topk_sim, _ = sim_matrix.topk(self.k, dim=1)  # [N, k]

        # Anomalía = 1 - similitud promedio de k vecinos
        anomaly_scores = 1 - topk_sim.mean(dim=1)  # [N]

        # Reshape a grid 2D
        h, w = self.extractor.get_grid_size(test_image)
        anomaly_map = anomaly_scores.cpu().numpy().reshape(h, w)

        # Suavizado
        anomaly_map_smooth = ndimage.gaussian_filter(anomaly_map, sigma=smooth_sigma)

        # Score a nivel de imagen (máximo score de anomalía)
        image_score = anomaly_map_smooth.max()

        return anomaly_map, anomaly_map_smooth, image_score

    def predict_batch(
        self, test_images: List[Image.Image], smooth_sigma: float = 0.8
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predice anomalías para un batch de imágenes.

        Returns:
            anomaly_maps: Lista de mapas de anomalía
            image_scores: Lista de scores por imagen
        """
        anomaly_maps = []
        image_scores = []

        for img in test_images:
            _, amap_smooth, score = self.compute_anomaly_map(img, smooth_sigma)
            anomaly_maps.append(amap_smooth)
            image_scores.append(score)

        return anomaly_maps, image_scores

    def visualize(
        self,
        test_image: Image.Image,
        anomaly_map: np.ndarray,
        image_score: float,
        title: str = "Memory Bank + k-NN - Detección de Anomalías",
    ):
        """Visualiza resultado de detección."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(test_image)
        axes[0].set_title("Imagen Test")
        axes[0].axis("off")

        axes[1].imshow(test_image)
        im = axes[1].imshow(
            anomaly_map,
            cmap="jet",
            alpha=0.5,
            extent=(0, test_image.width, test_image.height, 0),
        )
        axes[1].set_title(f"Mapa de Anomalía (Score: {image_score:.4f})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return fig


# =============================================================================
# UTILIDADES DE VISUALIZACIÓN
# =============================================================================


def visualize_layer_comparison(
    extractor: DINOv2FeatureExtractor,
    image: Image.Image,
    layers_to_compare: List[int] = [0, 3, 6, 9, 12, -1],
):
    """
    Compara visualización PCA de diferentes capas del modelo.

    Útil para elegir la capa óptima para extracción de features.
    """
    original_layer = extractor.layer_idx

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, layer_idx in zip(axes, layers_to_compare):
        extractor.layer_idx = layer_idx
        patches = extractor.extract_patches(image, normalize=False).cpu().numpy()

        # PCA a 3 componentes para RGB
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(patches)

        # Normalizar a [0, 1]
        pca_min = pca_result.min(axis=0)
        pca_max = pca_result.max(axis=0)
        pca_norm = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)

        h, w = extractor.get_grid_size(image)
        pca_image = pca_norm.reshape(h, w, 3)

        layer_name = "Última" if layer_idx == -1 else str(layer_idx)
        ax.imshow(pca_image)
        ax.set_title(f"Capa {layer_name}")
        ax.axis("off")

    extractor.layer_idx = original_layer
    plt.suptitle("Comparación de Features por Capa (PCA→RGB)", fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


def upsample_anomaly_map(
    anomaly_map: np.ndarray, target_size: Tuple[int, int], mode: str = "bilinear"
) -> np.ndarray:
    """
    Escala el mapa de anomalía al tamaño de la imagen original.

    Args:
        anomaly_map: Mapa de anomalía [H, W]
        target_size: (height, width) objetivo
        mode: 'bilinear' o 'nearest'
    """
    amap_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0).float()

    # align_corners solo para modos que lo soporten
    if mode in ("nearest", "area"):
        upsampled = F.interpolate(amap_tensor, size=target_size, mode=mode)
    else:
        upsampled = F.interpolate(
            amap_tensor, size=target_size, mode=mode, align_corners=False
        )

    return upsampled.squeeze().numpy()


# =============================================================================
# MÉTRICAS DE EVALUACIÓN
# =============================================================================


class AnomalyEvaluator:
    """
    Evaluador de métricas para detección de anomalías.

    Calcula métricas pixel-level y region-level comparando
    mapas de anomalía predichos contra ground truth masks.

    Métricas implementadas:
    - Pixel-level: IoU, Dice, Precision, Recall, F1
    - Region-level: PRO (Per-Region Overlap)

    Características mejoradas:
    - auto_normalize: Normalización automática al rango [0, 1]
    - Resize automático al tamaño del ground truth
    - Estadísticas de diagnóstico en resultados
    """

    def __init__(self, threshold: float = 0.5, auto_normalize: bool = True):
        """
        Args:
            threshold: Umbral para binarizar el mapa de anomalía (0-1).
                      Se aplica DESPUÉS de normalizar al rango [0, 1].
            auto_normalize: Si True, normaliza automáticamente el mapa
                           al rango [0, 1] antes de aplicar el umbral.
        """
        self.threshold = threshold
        self.auto_normalize = auto_normalize

    @staticmethod
    def load_ground_truth(gt_path: str) -> np.ndarray:
        """
        Carga una máscara ground truth como array binario.

        Args:
            gt_path: Ruta a la imagen de ground truth

        Returns:
            mask: Array binario [H, W] con valores 0 o 1
        """
        gt_image = Image.open(gt_path).convert("L")
        gt_array = np.array(gt_image)
        # Binarizar (MVTec usa 255 para anomalía, 0 para normal)
        return (gt_array > 127).astype(np.float32)

    def binarize_anomaly_map(
        self, anomaly_map: np.ndarray, threshold: float = None
    ) -> np.ndarray:
        """
        Binariza un mapa de anomalía usando el umbral.

        Args:
            anomaly_map: Mapa de anomalía [H, W] con valores en [0, 1]
            threshold: Umbral (usa self.threshold si es None)

        Returns:
            binary_mask: Máscara binaria [H, W]
        """
        if threshold is None:
            threshold = self.threshold

        # Normalizar a [0, 1] si es necesario
        amap_min = anomaly_map.min()
        amap_max = anomaly_map.max()
        if amap_max > amap_min:
            normalized = (anomaly_map - amap_min) / (amap_max - amap_min)
        else:
            normalized = np.zeros_like(anomaly_map)

        return (normalized >= threshold).astype(np.float32)

    # -------------------------------------------------------------------------
    # PIXEL-LEVEL METRICS
    # -------------------------------------------------------------------------

    def compute_pixel_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
        """
        Calcula métricas a nivel de pixel.

        Args:
            pred_mask: Máscara predicha binaria [H, W]
            gt_mask: Ground truth binario [H, W]

        Returns:
            dict con: IoU, Dice, Precision, Recall, F1
        """
        # Asegurar mismo tamaño
        if pred_mask.shape != gt_mask.shape:
            pred_mask = upsample_anomaly_map(
                pred_mask, target_size=gt_mask.shape, mode="nearest"
            )

        pred = pred_mask.flatten().astype(bool)
        gt = gt_mask.flatten().astype(bool)

        # Componentes de la matriz de confusión
        tp = np.sum(pred & gt)  # True Positives
        fp = np.sum(pred & ~gt)  # False Positives
        fn = np.sum(~pred & gt)  # False Negatives
        tn = np.sum(~pred & ~gt)  # True Negatives

        # Métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0

        dice = 2 * intersection / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            "IoU": iou,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

    # -------------------------------------------------------------------------
    # REGION-LEVEL METRICS (PRO)
    # -------------------------------------------------------------------------

    def compute_pro_single(
        self, pred_mask: np.ndarray, gt_mask: np.ndarray
    ) -> Tuple[float, int, List[float]]:
        """
        Calcula PRO para una sola imagen con un umbral fijo.

        Fórmula del paper:
        PRO = (1/N) * Σ_i Σ_k (|P_i ∩ C_{i,k}| / |C_{i,k}|)

        Donde:
        - N = número total de regiones conectadas en el ground truth
        - C_{i,k} = píxeles del componente k en imagen i
        - P_i = píxeles predichos como anómalos

        Args:
            pred_mask: Máscara binaria de predicción [H, W]
            gt_mask: Ground truth binario [H, W]

        Returns:
            pro_score: Score PRO para esta imagen (promedio sobre regiones)
            num_regions: Número de regiones conectadas
            region_overlaps: Lista de overlaps por región
        """
        from scipy import ndimage as ndi

        # Asegurar mismo tamaño
        if pred_mask.shape != gt_mask.shape:
            pred_mask = upsample_anomaly_map(
                pred_mask, target_size=gt_mask.shape, mode="nearest"
            )

        # Encontrar regiones conectadas en ground truth
        labeled_gt, num_regions = ndi.label(gt_mask > 0)

        if num_regions == 0:
            return 1.0, 0, []

        # Calcular overlap para cada región: |P ∩ C_k| / |C_k|
        region_overlaps = []
        pred_binary = pred_mask > 0

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_gt == region_id
            region_size = np.sum(region_mask)  # |C_k|

            if region_size > 0:
                intersection = np.sum(pred_binary & region_mask)  # |P ∩ C_k|
                overlap = intersection / region_size
                region_overlaps.append(overlap)

        # PRO para esta imagen = promedio sobre regiones
        pro_score = np.mean(region_overlaps) if region_overlaps else 0.0

        return pro_score, num_regions, region_overlaps

    def compute_pro(
        self, anomaly_map: np.ndarray, gt_mask: np.ndarray, threshold: float = None
    ) -> Tuple[float, int, List[float]]:
        """
        Calcula PRO para un mapa de anomalía usando un umbral.

        Args:
            anomaly_map: Mapa de anomalía [H, W] (valores continuos)
            gt_mask: Ground truth binario [H, W]
            threshold: Umbral para binarizar (usa self.threshold si None)

        Returns:
            pro_score: Score PRO
            num_regions: Número de regiones
            region_overlaps: Overlaps por región
        """
        # Binarizar mapa de anomalía
        pred_mask = self.binarize_anomaly_map(anomaly_map, threshold)
        return self.compute_pro_single(pred_mask, gt_mask)

    def compute_au_pro(
        self,
        anomaly_map: np.ndarray,
        gt_mask: np.ndarray,
        num_thresholds: int = 100,
        fpr_limit: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Calcula AU-PRO (Area Under PRO curve).

        Calcula la curva PRO vs FPR para múltiples umbrales y el área
        bajo la curva hasta un límite de FPR (típicamente 0.3).

        Args:
            anomaly_map: Mapa de anomalía [H, W] (valores continuos)
            gt_mask: Ground truth binario [H, W]
            num_thresholds: Número de umbrales para la curva
            fpr_limit: Límite de FPR para integración (default 0.3)

        Returns:
            thresholds: Umbrales usados
            fpr_values: FPR para cada umbral
            pro_values: PRO para cada umbral
            au_pro: Área bajo la curva PRO normalizada
        """
        from scipy import ndimage as ndi

        # Asegurar mismo tamaño
        if anomaly_map.shape != gt_mask.shape:
            anomaly_map = upsample_anomaly_map(
                anomaly_map, target_size=gt_mask.shape, mode="bilinear"
            )

        # Normalizar mapa a [0, 1]
        amap_min = anomaly_map.min()
        amap_max = anomaly_map.max()
        if amap_max > amap_min:
            anomaly_map_norm = (anomaly_map - amap_min) / (amap_max - amap_min)
        else:
            anomaly_map_norm = np.zeros_like(anomaly_map)

        # Encontrar regiones conectadas en ground truth
        labeled_gt, num_regions = ndi.label(gt_mask > 0)

        if num_regions == 0:
            return np.array([0.0]), np.array([0.0]), np.array([1.0]), 1.0

        # Calcular PRO y FPR para múltiples umbrales
        thresholds = np.linspace(0, 1, num_thresholds)
        fpr_values = []
        pro_values = []

        total_normal_pixels = np.sum(gt_mask == 0)

        for thresh in thresholds:
            pred_mask = anomaly_map_norm >= thresh

            # FPR: False Positive Rate en píxeles normales
            if total_normal_pixels > 0:
                fp = np.sum(pred_mask & (gt_mask == 0))
                fpr = fp / total_normal_pixels
            else:
                fpr = 0.0

            # PRO: (1/N) * Σ_k (|P ∩ C_k| / |C_k|)
            region_overlaps = []
            for region_id in range(1, num_regions + 1):
                region_mask = labeled_gt == region_id
                region_size = np.sum(region_mask)

                if region_size > 0:
                    intersection = np.sum(pred_mask & region_mask)
                    overlap = intersection / region_size
                    region_overlaps.append(overlap)

            pro = np.mean(region_overlaps) if region_overlaps else 0.0

            fpr_values.append(fpr)
            pro_values.append(pro)

        thresholds = np.array(thresholds)
        fpr_values = np.array(fpr_values)
        pro_values = np.array(pro_values)

        # Calcular AU-PRO (área bajo curva hasta FPR limit)
        valid_idx = fpr_values <= fpr_limit

        if np.sum(valid_idx) > 1:
            # Ordenar por FPR e integrar
            sorted_idx = np.argsort(fpr_values[valid_idx])
            fpr_sorted = fpr_values[valid_idx][sorted_idx]
            pro_sorted = pro_values[valid_idx][sorted_idx]
            au_pro = np.trapz(pro_sorted, fpr_sorted) / fpr_limit
        else:
            au_pro = 0.0

        return thresholds, fpr_values, pro_values, au_pro

    def evaluate(
        self, anomaly_map: np.ndarray, gt_mask: np.ndarray, threshold: float = None
    ) -> dict:
        """
        Evalúa todas las métricas para un mapa de anomalía.

        El proceso mejorado:
        1. Guarda estadísticas originales para diagnóstico
        2. Resize automático al tamaño del GT si difieren
        3. Normaliza al rango [0, 1] (si auto_normalize=True)
        4. Aplica umbral sobre valores normalizados

        Args:
            anomaly_map: Mapa de anomalía [H, W] (valores crudos del modelo)
            gt_mask: Ground truth binario [H, W]
            threshold: Umbral para binarización (0-1). Usa self.threshold si es None.

        Returns:
            dict con métricas + estadísticas de diagnóstico:
            - Métricas estándar: IoU, Dice, Precision, Recall, F1, PRO, AU-PRO
            - Estadísticas: orig_min, orig_max, orig_mean, normalized
        """
        use_threshold = threshold if threshold is not None else self.threshold

        # =====================================================================
        # PASO 1: Guardar estadísticas originales para diagnóstico
        # =====================================================================
        orig_min = float(anomaly_map.min())
        orig_max = float(anomaly_map.max())
        orig_mean = float(anomaly_map.mean())
        orig_std = float(anomaly_map.std())

        # =====================================================================
        # PASO 2: Resize automático al tamaño del ground truth
        # =====================================================================
        if anomaly_map.shape != gt_mask.shape:
            anomaly_map_resized = resize_anomaly_map(
                anomaly_map, target_size=gt_mask.shape, interpolation="bilinear"
            )
        else:
            anomaly_map_resized = anomaly_map.copy()

        # =====================================================================
        # PASO 3: Normalizar al rango [0, 1] antes de aplicar umbral
        # =====================================================================
        if self.auto_normalize:
            anomaly_map_normalized = normalize_anomaly_map(
                anomaly_map_resized, method="minmax"
            )
        else:
            anomaly_map_normalized = anomaly_map_resized

        # =====================================================================
        # PASO 4: Binarizar usando el umbral sobre valores normalizados [0, 1]
        # =====================================================================
        pred_mask = (anomaly_map_normalized >= use_threshold).astype(np.float32)

        # =====================================================================
        # PASO 5: Calcular métricas
        # =====================================================================
        # Pixel-level metrics
        pixel_metrics = self.compute_pixel_metrics(pred_mask, gt_mask)

        # Region-level PRO (para el umbral seleccionado)
        pro_score, num_regions, region_overlaps = self.compute_pro_single(
            pred_mask, gt_mask
        )

        # AU-PRO (área bajo la curva PRO-FPR)
        _, _, _, au_pro = self.compute_au_pro(anomaly_map_normalized, gt_mask)

        return {
            **pixel_metrics,
            "PRO": pro_score,
            "AU-PRO": au_pro,
            "Num_Regions": num_regions,
            "Threshold": use_threshold,
            # Estadísticas de diagnóstico
            "orig_min": orig_min,
            "orig_max": orig_max,
            "orig_mean": orig_mean,
            "orig_std": orig_std,
            "normalized": self.auto_normalize,
            # Mapa normalizado para visualización
            "_anomaly_map_normalized": anomaly_map_normalized,
            "_pred_mask": pred_mask,
        }

    def visualize_comparison(
        self,
        test_image: Image.Image,
        anomaly_map: np.ndarray,
        gt_mask: np.ndarray,
        metrics: dict,
        title: str = "Comparación: Predicción vs Ground Truth",
        show_original_values: bool = True,
        save_path: str = None,
        dpi: int = 150,
        show: bool = True,
    ):
        """
        Visualiza predicción vs ground truth con métricas.

        Muestra 5 paneles:
        1. Imagen original
        2. Ground Truth
        3. Mapa original (valores sin normalizar)
        4. Mapa normalizado [0,1]
        5. Predicción binarizada
        """
        # Preparar datos
        # Resize al tamaño de la imagen para visualización
        if anomaly_map.shape != gt_mask.shape:
            amap_resized = resize_anomaly_map(anomaly_map, gt_mask.shape)
        else:
            amap_resized = anomaly_map.copy()

        # Guardar valores originales
        amap_original = amap_resized.copy()
        orig_min, orig_max = amap_original.min(), amap_original.max()

        # Normalizar
        amap_normalized = normalize_anomaly_map(amap_resized, method="minmax")

        # Crear figura
        n_cols = 5 if show_original_values else 4
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        ax_idx = 0

        # Panel 1: Imagen original
        axes[ax_idx].imshow(test_image)
        axes[ax_idx].set_title("Imagen Test")
        axes[ax_idx].axis("off")
        ax_idx += 1

        # Panel 2: Ground Truth
        axes[ax_idx].imshow(test_image)
        axes[ax_idx].imshow(
            resize_anomaly_map(
                gt_mask.astype(float), (test_image.height, test_image.width)
            ),
            cmap="Reds",
            alpha=0.5,
            extent=(0, test_image.width, test_image.height, 0),
        )
        axes[ax_idx].set_title(f"Ground Truth\n({gt_mask.sum()} píxeles)")
        axes[ax_idx].axis("off")
        ax_idx += 1

        if show_original_values:
            # Panel 3: Mapa original (sin normalizar)
            axes[ax_idx].imshow(test_image)
            im_orig = axes[ax_idx].imshow(
                amap_original,
                cmap="jet",
                alpha=0.5,
                extent=(0, test_image.width, test_image.height, 0),
            )
            axes[ax_idx].set_title(f"Mapa Original\n[{orig_min:.4f}, {orig_max:.4f}]")
            axes[ax_idx].axis("off")
            plt.colorbar(im_orig, ax=axes[ax_idx], fraction=0.046, pad=0.04)
            ax_idx += 1

        # Panel 4: Mapa normalizado
        axes[ax_idx].imshow(test_image)
        im_norm = axes[ax_idx].imshow(
            amap_normalized,
            cmap="jet",
            alpha=0.5,
            vmin=0,
            vmax=1,
            extent=(0, test_image.width, test_image.height, 0),
        )
        axes[ax_idx].set_title(f"Mapa Normalizado\n[0.0, 1.0]")
        axes[ax_idx].axis("off")
        plt.colorbar(im_norm, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        ax_idx += 1

        # Panel 5: Predicción binarizada
        pred_binary = amap_normalized >= self.threshold
        axes[ax_idx].imshow(test_image)
        axes[ax_idx].imshow(
            pred_binary,
            cmap="Blues",
            alpha=0.5,
            extent=(0, test_image.width, test_image.height, 0),
        )
        n_pred = pred_binary.sum()
        axes[ax_idx].set_title(
            f"Predicción Binaria\n(τ={self.threshold}, {n_pred} píxeles)"
        )
        axes[ax_idx].axis("off")

        # Métricas como texto
        metrics_text = (
            f"IoU: {metrics['IoU']:.3f} | "
            f"Dice: {metrics['Dice']:.3f} | "
            f"F1: {metrics['F1']:.3f} | "
            f"Precision: {metrics['Precision']:.3f} | "
            f"Recall: {metrics['Recall']:.3f} | "
            f"AU-PRO: {metrics['AU-PRO']:.3f}"
        )

        plt.figtext(
            0.5,
            0.02,
            metrics_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

        # Guardar figura si se especifica ruta
        if save_path:
            import os

            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def load_test_with_ground_truth(
    test_folder: str, gt_folder: str, n_images: int = None
) -> List[Tuple[Image.Image, np.ndarray, str]]:
    """
    Carga imágenes de test junto con sus ground truth masks.

    Args:
        test_folder: Carpeta con imágenes de test (ej: .../test/broken_large)
        gt_folder: Carpeta con ground truth (ej: .../ground_truth/broken_large)
        n_images: Número de imágenes a cargar (None = todas)

    Returns:
        Lista de tuplas (test_image, gt_mask, filename)
    """
    import os

    extensions = (".png", ".jpg", ".jpeg", ".bmp")

    test_files = sorted(
        [f for f in os.listdir(test_folder) if f.lower().endswith(extensions)]
    )

    if n_images is not None:
        test_files = test_files[:n_images]

    results = []
    for filename in test_files:
        # Cargar imagen test
        test_path = os.path.join(test_folder, filename)
        test_img = Image.open(test_path).convert("RGB")

        # Cargar ground truth (mismo nombre, extensión _mask.png en MVTec)
        gt_filename = filename.replace(".png", "_mask.png")
        gt_path = os.path.join(gt_folder, gt_filename)

        if os.path.exists(gt_path):
            gt_mask = AnomalyEvaluator.load_ground_truth(gt_path)
        else:
            # Intentar con el mismo nombre
            gt_path = os.path.join(gt_folder, filename)
            if os.path.exists(gt_path):
                gt_mask = AnomalyEvaluator.load_ground_truth(gt_path)
            else:
                print(f"⚠️ Ground truth no encontrado para {filename}")
                gt_mask = None

        if gt_mask is not None:
            results.append((test_img, gt_mask, filename))

    print(f"Cargados {len(results)} pares (test, ground_truth) de {test_folder}")
    return results


# =============================================================================
# EJEMPLO DE USO
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
    import os

    # Obtener lista de archivos de imagen
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    )

    # Limitar a N imágenes si se especifica
    if n_images is not None:
        image_files = image_files[:n_images]

    # Cargar imágenes
    images = []
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    print(f"Cargadas {len(images)} imágenes de {folder_path}")
    return images


# =============================================================================
# EVALUADOR COMPLETO MVTEC AD
# =============================================================================


class MVTecDatasetEvaluator:
    """
    Evaluador completo del dataset MVTec AD.

    Itera sobre todas las categorías y tipos de anomalía,
    calcula métricas globales, por clase y por anomalía.

    Args:
        dataset_path: Ruta base al dataset MVTec AD (contiene las 15 carpetas de categorías)
        model_path: Ruta al modelo DINOv2
        layer_idx: Índice de la capa a usar (-1 = última)
        n_good_images: Número de imágenes "good" para memory bank (None = todas)
        k: Número de vecinos para k-NN
        coreset_ratio: Ratio de subsampling del memory bank
        threshold: Umbral para binarización
        auto_normalize: Si True, normaliza mapas de anomalía automáticamente
    """

    # Lista de las 15 categorías de MVTec AD
    CATEGORIES = [
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

    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        layer_idx: int = -1,
        n_good_images: int = None,
        k: int = 1,
        coreset_ratio: float = 1.0,
        threshold: float = 0.5,
        auto_normalize: bool = True,
        device: str = None,
    ):
        import os

        self.dataset_path = dataset_path
        self.model_path = model_path
        self.layer_idx = layer_idx
        self.n_good_images = n_good_images
        self.k = k
        self.coreset_ratio = coreset_ratio
        self.threshold = threshold
        self.auto_normalize = auto_normalize
        self.device = device

        # Crear extractor de features
        self.extractor = DINOv2FeatureExtractor(
            model_path=model_path, layer_idx=layer_idx, device=device
        )

        # Evaluador de métricas
        self.evaluator = AnomalyEvaluator(
            threshold=threshold, auto_normalize=auto_normalize
        )

        # Descubrir categorías disponibles
        self.available_categories = self._discover_categories()
        print(
            f"🔍 Encontradas {len(self.available_categories)} categorías en {dataset_path}"
        )

    def _discover_categories(self) -> List[str]:
        """Descubre las categorías disponibles en el dataset."""
        import os

        categories = []
        for cat in self.CATEGORIES:
            cat_path = os.path.join(self.dataset_path, cat)
            if os.path.isdir(cat_path):
                categories.append(cat)
        return categories

    def _get_anomaly_types(self, category: str) -> List[str]:
        """
        Obtiene los tipos de anomalía para una categoría (excluyendo 'good').
        Solo incluye tipos que tienen ground truth disponible.
        """
        import os

        test_path = os.path.join(self.dataset_path, category, "test")
        gt_path = os.path.join(self.dataset_path, category, "ground_truth")

        anomaly_types = []
        if os.path.isdir(test_path):
            for folder in sorted(os.listdir(test_path)):
                folder_path = os.path.join(test_path, folder)
                gt_folder_path = os.path.join(gt_path, folder)
                # Solo incluir si no es 'good' y tiene ground truth
                if os.path.isdir(folder_path) and folder != "good":
                    if os.path.isdir(gt_folder_path):
                        anomaly_types.append(folder)
        return anomaly_types

    def _load_good_images(self, category: str) -> List[Image.Image]:
        """Carga imágenes 'good' del training set para una categoría."""
        import os

        good_path = os.path.join(self.dataset_path, category, "train", "good")
        return load_images_from_folder(good_path, n_images=self.n_good_images)

    def _build_memory_bank(
        self, category: str, verbose: bool = True
    ) -> MemoryBankDetector:
        """Construye el memory bank para una categoría."""
        if verbose:
            print(f"\n📦 Construyendo Memory Bank para '{category}'...")

        good_images = self._load_good_images(category)

        detector = MemoryBankDetector(
            extractor=self.extractor, k=self.k, coreset_ratio=self.coreset_ratio
        )
        detector.build_memory_bank(good_images, verbose=verbose)

        return detector

    def evaluate_category(
        self,
        category: str,
        detector: MemoryBankDetector = None,
        verbose: bool = True,
        save_visualizations: bool = False,
        visualization_dir: str = None,
        save_curves: bool = False,
    ) -> dict:
        """
        Evalúa todas las anomalías de una categoría.

        Args:
            category: Nombre de la categoría
            detector: Memory bank detector (si None, se construye uno nuevo)
            verbose: Si True, muestra progreso
            save_visualizations: Si True, guarda visualizaciones de cada imagen
            visualization_dir: Directorio donde guardar las visualizaciones
                              (default: ./visualizations/{category})
            save_curves: Si True, guarda curvas de evaluación (ROC, PR, PRO, IoU)

        Returns:
            dict con métricas por anomalía y resumen de la categoría
        """
        import os

        if detector is None:
            detector = self._build_memory_bank(category, verbose)

        # Configurar directorio de visualizaciones
        if save_visualizations and visualization_dir is None:
            visualization_dir = os.path.join(".", "visualizations", category)

        anomaly_types = self._get_anomaly_types(category)

        if verbose:
            print(
                f"\n🔍 Evaluando {len(anomaly_types)} tipos de anomalía en '{category}'"
            )

        # Listas para almacenar mapas para curvas
        all_anomaly_maps = []
        all_gt_masks = []

        category_results = {
            "category": category,
            "anomaly_results": {},
            "all_metrics": [],
            "summary": {},
        }

        for anomaly_type in anomaly_types:
            test_folder = os.path.join(
                self.dataset_path, category, "test", anomaly_type
            )
            gt_folder = os.path.join(
                self.dataset_path, category, "ground_truth", anomaly_type
            )

            if not os.path.isdir(gt_folder):
                if verbose:
                    print(f"  ⚠️ Sin ground truth para {anomaly_type}, saltando...")
                continue

            # Cargar pares test/ground_truth
            test_data = load_test_with_ground_truth(
                test_folder=test_folder,
                gt_folder=gt_folder,
                n_images=None,  # Todas las imágenes
            )

            if len(test_data) == 0:
                if verbose:
                    print(f"  ⚠️ No hay datos para {anomaly_type}")
                continue

            if verbose:
                print(f"\n  📍 Tipo: {anomaly_type} ({len(test_data)} imágenes)")

            anomaly_metrics = []

            for test_img, gt_mask, filename in test_data:
                # Calcular mapa de anomalía
                amap, amap_smooth, score = detector.compute_anomaly_map(test_img)

                # Evaluar métricas
                metrics = self.evaluator.evaluate(amap_smooth, gt_mask)
                metrics["filename"] = filename
                metrics["category"] = category
                metrics["anomaly_type"] = anomaly_type
                metrics["image_score"] = score

                # Guardar visualización si está habilitado
                if save_visualizations:
                    vis_subdir = os.path.join(visualization_dir, anomaly_type)
                    os.makedirs(vis_subdir, exist_ok=True)

                    # Nombre de archivo sin extensión
                    base_name = os.path.splitext(filename)[0]
                    save_path = os.path.join(vis_subdir, f"{base_name}_comparison.png")

                    self.evaluator.visualize_comparison(
                        test_image=test_img,
                        anomaly_map=amap_smooth,
                        gt_mask=gt_mask,
                        metrics=metrics,
                        title=f"{category}/{anomaly_type}: {filename}",
                        save_path=save_path,
                        show=False,
                    )

                # Acumular mapas para curvas
                if save_curves:
                    all_anomaly_maps.append(amap_smooth)
                    all_gt_masks.append(gt_mask)

                anomaly_metrics.append(metrics)

            # Calcular promedios para este tipo de anomalía
            anomaly_summary = self._compute_summary(anomaly_metrics)
            anomaly_summary["n_images"] = len(anomaly_metrics)

            category_results["anomaly_results"][anomaly_type] = {
                "metrics": anomaly_metrics,
                "summary": anomaly_summary,
            }
            category_results["all_metrics"].extend(anomaly_metrics)

            if verbose:
                print(
                    f"     IoU: {anomaly_summary['IoU']:.4f} | "
                    f"Dice: {anomaly_summary['Dice']:.4f} | "
                    f"F1: {anomaly_summary['F1']:.4f} | "
                    f"AU-PRO: {anomaly_summary['AU-PRO']:.4f}"
                )

        # Resumen de la categoría completa
        if category_results["all_metrics"]:
            category_results["summary"] = self._compute_summary(
                category_results["all_metrics"]
            )
            category_results["summary"]["n_images"] = len(
                category_results["all_metrics"]
            )
            category_results["summary"]["n_anomaly_types"] = len(anomaly_types)

        # Generar y guardar curvas si está habilitado
        if save_curves and len(all_anomaly_maps) > 0:
            if verbose:
                print(f"\n📈 Generando curvas para '{category}'...")

            curves_dir = visualization_dir or os.path.join(
                ".", "visualizations", category
            )
            os.makedirs(curves_dir, exist_ok=True)

            curve_visualizer = CurveVisualizer()

            # Computar todas las curvas
            curves_dict = curve_visualizer.compute_all_curves(
                anomaly_maps=all_anomaly_maps,
                gt_masks=all_gt_masks,
                normalize_maps=True,
            )

            # Guardar curvas combinadas
            curves_path = os.path.join(curves_dir, f"curves_{category}_all.png")
            curve_visualizer.plot_all_curves(
                curves_dict,
                title=f"Curvas de Evaluación - {category.upper()}",
                save_path=curves_path,
                show=False,
            )

            # Guardar datos de curvas en JSON
            curves_data_path = os.path.join(curves_dir, f"curves_{category}_data.json")
            curve_visualizer.export_curve_data(
                curves_dict, curves_data_path, format="json"
            )

            if verbose:
                print(f"   📊 Curvas guardadas en: {curves_path}")
                print(f"   💾 Datos guardados en: {curves_data_path}")

            # Agregar métricas de curvas al resumen
            category_results["curves"] = {
                "au_roc": curves_dict.get("roc", {}).get("au_roc"),
                "au_pr": curves_dict.get("pr", {}).get("au_pr"),
                "au_pro": curves_dict.get("pro", {}).get("au_pro"),
                "best_iou": curves_dict.get("iou", {}).get("best_iou"),
                "best_threshold": curves_dict.get("iou", {}).get("best_threshold"),
            }

        return category_results

    def _compute_summary(self, metrics_list: List[dict]) -> dict:
        """Calcula métricas promedio de una lista de métricas."""
        if not metrics_list:
            return {}

        summary = {}
        metric_keys = ["IoU", "Dice", "Precision", "Recall", "F1", "PRO", "AU-PRO"]

        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                summary[key] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)

        return summary

    def evaluate_all(
        self,
        categories: List[str] = None,
        verbose: bool = True,
        save_results: bool = True,
        output_path: str = None,
        save_visualizations: bool = False,
        visualization_dir: str = None,
        save_curves: bool = False,
    ) -> dict:
        """
        Evalúa todas las categorías (o las especificadas).

        Args:
            categories: Lista de categorías a evaluar (None = todas)
            verbose: Si True, muestra progreso detallado
            save_results: Si True, guarda resultados en JSON
            output_path: Ruta para guardar resultados (si save_results=True)
            save_visualizations: Si True, guarda visualizaciones de cada imagen
            visualization_dir: Directorio base para visualizaciones
                              (default: ./visualizations/)
            save_curves: Si True, guarda curvas de evaluación (ROC, PR, PRO, IoU)

        Returns:
            dict con resultados completos: por categoría, por anomalía y globales
        """
        import json
        import os
        from datetime import datetime

        if categories is None:
            categories = self.available_categories

        print("\n" + "=" * 80)
        print("🚀 EVALUACIÓN COMPLETA DEL DATASET MVTEC AD")
        print("=" * 80)
        print(f"   Modelo: {self.model_path}")
        print(f"   Capa: {self.layer_idx}")
        print(f"   k-NN: k={self.k}")
        print(f"   Umbral: {self.threshold}")
        print(f"   Categorías: {len(categories)}")
        print("=" * 80)

        all_results = {
            "config": {
                "model_path": self.model_path,
                "layer_idx": self.layer_idx,
                "k": self.k,
                "coreset_ratio": self.coreset_ratio,
                "threshold": self.threshold,
                "auto_normalize": self.auto_normalize,
                "n_good_images": self.n_good_images,
                "timestamp": datetime.now().isoformat(),
            },
            "category_results": {},
            "global_metrics": [],
            "summary_by_category": {},
            "summary_by_anomaly_type": {},
            "global_summary": {},
        }

        for i, category in enumerate(categories, 1):
            print(f"\n{'='*80}")
            print(f"📂 [{i}/{len(categories)}] Categoría: {category.upper()}")
            print("=" * 80)

            try:
                # Configurar directorio de visualizaciones por categoría
                cat_vis_dir = None
                if save_visualizations:
                    base_vis_dir = visualization_dir or "./visualizations"
                    cat_vis_dir = os.path.join(base_vis_dir, category)

                cat_results = self.evaluate_category(
                    category,
                    verbose=verbose,
                    save_visualizations=save_visualizations,
                    visualization_dir=cat_vis_dir,
                    save_curves=save_curves,
                )
                all_results["category_results"][category] = cat_results
                all_results["global_metrics"].extend(cat_results["all_metrics"])
                all_results["summary_by_category"][category] = cat_results["summary"]

                # Agregar métricas por tipo de anomalía al resumen global
                for anomaly_type, anom_data in cat_results["anomaly_results"].items():
                    key = f"{category}/{anomaly_type}"
                    all_results["summary_by_anomaly_type"][key] = anom_data["summary"]

            except Exception as e:
                print(f"  ❌ Error procesando {category}: {e}")
                import traceback

                traceback.print_exc()

        # Calcular métricas globales
        if all_results["global_metrics"]:
            all_results["global_summary"] = self._compute_summary(
                all_results["global_metrics"]
            )
            all_results["global_summary"]["n_total_images"] = len(
                all_results["global_metrics"]
            )
            all_results["global_summary"]["n_categories"] = len(categories)

        # Imprimir resumen final
        self._print_final_summary(all_results)

        # Guardar resultados
        if save_results:
            if output_path is None:
                output_path = os.path.join(
                    self.dataset_path,
                    "..",
                    f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                )
            self._save_results(all_results, output_path)

        return all_results

    def _print_final_summary(self, results: dict):
        """Imprime el resumen final de la evaluación."""
        print("\n" + "=" * 80)
        print("📊 RESUMEN FINAL DE EVALUACIÓN")
        print("=" * 80)

        # Resumen por categoría
        print("\n🏷️  MÉTRICAS POR CATEGORÍA:")
        print("-" * 80)
        print(
            f"{'Categoría':<15} {'IoU':<10} {'Dice':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'AU-PRO':<10}"
        )
        print("-" * 80)

        for category, summary in results["summary_by_category"].items():
            if summary:
                print(
                    f"{category:<15} "
                    f"{summary.get('IoU', 0):<10.4f} "
                    f"{summary.get('Dice', 0):<10.4f} "
                    f"{summary.get('F1', 0):<10.4f} "
                    f"{summary.get('Precision', 0):<10.4f} "
                    f"{summary.get('Recall', 0):<10.4f} "
                    f"{summary.get('AU-PRO', 0):<10.4f}"
                )

        # Resumen global
        print("\n" + "=" * 80)
        print("🌍 MÉTRICAS GLOBALES:")
        print("=" * 80)
        gs = results["global_summary"]
        if gs:
            print(f"   Total imágenes evaluadas: {gs.get('n_total_images', 0)}")
            print(f"   Total categorías: {gs.get('n_categories', 0)}")
            print()
            print(f"   📈 Métricas Pixel-Level:")
            print(
                f"      IoU:       {gs.get('IoU', 0):.4f} (± {gs.get('IoU_std', 0):.4f})"
            )
            print(
                f"      Dice:      {gs.get('Dice', 0):.4f} (± {gs.get('Dice_std', 0):.4f})"
            )
            print(
                f"      Precision: {gs.get('Precision', 0):.4f} (± {gs.get('Precision_std', 0):.4f})"
            )
            print(
                f"      Recall:    {gs.get('Recall', 0):.4f} (± {gs.get('Recall_std', 0):.4f})"
            )
            print(
                f"      F1:        {gs.get('F1', 0):.4f} (± {gs.get('F1_std', 0):.4f})"
            )
            print()
            print(f"   📈 Métricas Region-Level:")
            print(
                f"      PRO:       {gs.get('PRO', 0):.4f} (± {gs.get('PRO_std', 0):.4f})"
            )
            print(
                f"      AU-PRO:    {gs.get('AU-PRO', 0):.4f} (± {gs.get('AU-PRO_std', 0):.4f})"
            )

    def _save_results(self, results: dict, output_path: str):
        """Guarda los resultados en formato JSON."""
        import json
        import os

        # Eliminar datos internos que no son serializables
        results_clean = self._clean_results_for_json(results)

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultados guardados en: {output_path}")

    def _clean_results_for_json(self, results: dict) -> dict:
        """Limpia los resultados para serialización JSON."""
        import copy

        def clean_value(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {
                    k: clean_value(val) for k, val in v.items() if not k.startswith("_")
                }
            elif isinstance(v, list):
                return [clean_value(item) for item in v]
            else:
                return v

        return clean_value(results)

    def generate_report(self, results: dict, output_path: str = None) -> str:
        """
        Genera un reporte detallado en formato Markdown.

        Args:
            results: Resultados de evaluate_all()
            output_path: Ruta para guardar el reporte (opcional)

        Returns:
            Contenido del reporte en Markdown
        """
        import os
        from datetime import datetime

        report = []
        report.append("# 📊 Reporte de Evaluación MVTec AD")
        report.append(f"\n**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Configuración
        config = results.get("config", {})
        report.append("## ⚙️ Configuración\n")
        report.append(f"- **Modelo:** `{config.get('model_path', 'N/A')}`")
        report.append(f"- **Capa DINOv2:** {config.get('layer_idx', 'N/A')}")
        report.append(f"- **k-NN (k):** {config.get('k', 'N/A')}")
        report.append(f"- **Umbral:** {config.get('threshold', 'N/A')}")
        report.append(
            f"- **Imágenes 'good' para Memory Bank:** {config.get('n_good_images', 'Todas')}"
        )
        report.append("")

        # Resumen global
        gs = results.get("global_summary", {})
        report.append("## 🌍 Métricas Globales\n")
        report.append(f"- **Total imágenes evaluadas:** {gs.get('n_total_images', 0)}")
        report.append(f"- **Total categorías:** {gs.get('n_categories', 0)}")
        report.append("")
        report.append("| Métrica | Valor | Desv. Est. |")
        report.append("|---------|-------|------------|")
        for metric in ["IoU", "Dice", "Precision", "Recall", "F1", "PRO", "AU-PRO"]:
            val = gs.get(metric, 0)
            std = gs.get(f"{metric}_std", 0)
            report.append(f"| {metric} | {val:.4f} | ± {std:.4f} |")
        report.append("")

        # Tabla por categoría
        report.append("## 🏷️ Métricas por Categoría\n")
        report.append("| Categoría | IoU | Dice | F1 | Precision | Recall | AU-PRO |")
        report.append("|-----------|-----|------|----|-----------|----- --|--------|")

        for category, summary in results.get("summary_by_category", {}).items():
            if summary:
                report.append(
                    f"| {category} | "
                    f"{summary.get('IoU', 0):.4f} | "
                    f"{summary.get('Dice', 0):.4f} | "
                    f"{summary.get('F1', 0):.4f} | "
                    f"{summary.get('Precision', 0):.4f} | "
                    f"{summary.get('Recall', 0):.4f} | "
                    f"{summary.get('AU-PRO', 0):.4f} |"
                )
        report.append("")

        # Detalle por tipo de anomalía
        report.append("## 🔬 Detalle por Tipo de Anomalía\n")

        for category, cat_data in results.get("category_results", {}).items():
            report.append(f"### {category.upper()}\n")
            report.append("| Tipo de Anomalía | Imágenes | IoU | Dice | F1 | AU-PRO |")
            report.append("|------------------|----------|-----|------|----|----- --|")

            for anomaly_type, anom_data in cat_data.get("anomaly_results", {}).items():
                summary = anom_data.get("summary", {})
                report.append(
                    f"| {anomaly_type} | "
                    f"{summary.get('n_images', 0)} | "
                    f"{summary.get('IoU', 0):.4f} | "
                    f"{summary.get('Dice', 0):.4f} | "
                    f"{summary.get('F1', 0):.4f} | "
                    f"{summary.get('AU-PRO', 0):.4f} |"
                )
            report.append("")

        report_content = "\n".join(report)

        # Guardar si se especifica ruta
        if output_path:
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n📝 Reporte guardado en: {output_path}")

        return report_content


class CurveVisualizer:
    """
    Computes and visualizes evaluation curves for anomaly detection.

    Supported curves:
    - ROC (Receiver Operating Characteristic): FPR vs TPR
    - PR (Precision-Recall): Recall vs Precision
    - PRO (Per-Region Overlap): FPR vs PRO
    - IoU (Intersection over Union): Threshold vs IoU

    All curves are computed at pixel-level using aggregated predictions.

    Args:
        num_thresholds: Number of thresholds for curve computation
        fpr_limit: FPR limit for AU-PRO calculation (default 0.3)
    """

    def __init__(self, num_thresholds: int = 200, fpr_limit: float = 0.3):
        self.num_thresholds = num_thresholds
        self.fpr_limit = fpr_limit

    def compute_roc_curve(
        self,
        anomaly_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
        normalize_maps: bool = True,
    ) -> dict:
        """
        Computes pixel-level ROC curve.

        Args:
            anomaly_maps: List of anomaly maps [H, W]
            gt_masks: List of binary ground truth masks [H, W]
            normalize_maps: If True, normalize each map to [0, 1]

        Returns:
            dict with 'fpr', 'tpr', 'thresholds', 'au_roc'
        """
        from sklearn.metrics import auc, roc_curve

        # Flatten and concatenate all maps
        y_scores = []
        y_trues = []

        for amap, gt in zip(anomaly_maps, gt_masks):
            # Resize if needed
            if amap.shape != gt.shape:
                amap = resize_anomaly_map(amap, gt.shape)

            # Normalize
            if normalize_maps:
                amap = normalize_anomaly_map(amap, method="minmax")

            y_scores.append(amap.flatten())
            y_trues.append((gt > 0).astype(np.float32).flatten())

        y_score = np.concatenate(y_scores)
        y_true = np.concatenate(y_trues)

        # Compute ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        au_roc = auc(fpr, tpr)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "au_roc": au_roc,
            "curve_type": "roc",
        }

    def compute_pr_curve(
        self,
        anomaly_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
        normalize_maps: bool = True,
    ) -> dict:
        """
        Computes pixel-level Precision-Recall curve.

        Args:
            anomaly_maps: List of anomaly maps [H, W]
            gt_masks: List of binary ground truth masks [H, W]
            normalize_maps: If True, normalize each map to [0, 1]

        Returns:
            dict with 'precision', 'recall', 'thresholds', 'au_pr'
        """
        from sklearn.metrics import auc, precision_recall_curve

        # Flatten and concatenate
        y_scores = []
        y_trues = []

        for amap, gt in zip(anomaly_maps, gt_masks):
            if amap.shape != gt.shape:
                amap = resize_anomaly_map(amap, gt.shape)

            if normalize_maps:
                amap = normalize_anomaly_map(amap, method="minmax")

            y_scores.append(amap.flatten())
            y_trues.append((gt > 0).astype(np.float32).flatten())

        y_score = np.concatenate(y_scores)
        y_true = np.concatenate(y_trues)

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # AU-PR (area under precision-recall curve)
        au_pr = auc(recall, precision)

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "au_pr": au_pr,
            "curve_type": "pr",
        }

    def compute_iou_curve(
        self,
        anomaly_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
        normalize_maps: bool = True,
    ) -> dict:
        """
        Computes IoU vs Threshold curve.

        Args:
            anomaly_maps: List of anomaly maps [H, W]
            gt_masks: List of binary ground truth masks [H, W]
            normalize_maps: If True, normalize each map to [0, 1]

        Returns:
            dict with 'thresholds', 'iou_values', 'au_iou', 'best_threshold', 'best_iou'
        """
        thresholds = np.linspace(0, 1, self.num_thresholds)
        iou_values = []

        for thresh in thresholds:
            total_intersection = 0
            total_union = 0

            for amap, gt in zip(anomaly_maps, gt_masks):
                if amap.shape != gt.shape:
                    amap = resize_anomaly_map(amap, gt.shape)

                if normalize_maps:
                    amap = normalize_anomaly_map(amap, method="minmax")

                pred = (amap >= thresh).astype(np.float32)
                gt_binary = (gt > 0).astype(np.float32)

                intersection = np.sum(pred * gt_binary)
                union = np.sum(pred) + np.sum(gt_binary) - intersection

                total_intersection += intersection
                total_union += union

            iou = total_intersection / (total_union + 1e-8)
            iou_values.append(iou)

        iou_values = np.array(iou_values)

        # AU-IoU (area under IoU curve)
        au_iou = np.trapz(iou_values, thresholds)

        # Best threshold and IoU
        best_idx = np.argmax(iou_values)
        best_threshold = thresholds[best_idx]
        best_iou = iou_values[best_idx]

        return {
            "thresholds": thresholds,
            "iou_values": iou_values,
            "au_iou": au_iou,
            "best_threshold": best_threshold,
            "best_iou": best_iou,
            "curve_type": "iou",
        }

    def compute_pro_curve(
        self,
        anomaly_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
        normalize_maps: bool = True,
    ) -> dict:
        """
        Computes PRO (Per-Region Overlap) vs FPR curve.

        This is the standard metric for MVTec AD evaluation.

        Args:
            anomaly_maps: List of anomaly maps [H, W]
            gt_masks: List of binary ground truth masks [H, W]
            normalize_maps: If True, normalize each map to [0, 1]

        Returns:
            dict with 'fpr', 'pro', 'thresholds', 'au_pro'
        """
        from scipy import ndimage as ndi

        thresholds = np.linspace(0, 1, self.num_thresholds)
        pro_values = []
        fpr_values = []

        for thresh in thresholds:
            all_region_overlaps = []
            total_fp = 0
            total_normal_pixels = 0

            for amap, gt in zip(anomaly_maps, gt_masks):
                if amap.shape != gt.shape:
                    amap = resize_anomaly_map(amap, gt.shape)

                if normalize_maps:
                    amap = normalize_anomaly_map(amap, method="minmax")

                pred = amap >= thresh
                gt_binary = gt > 0

                # FPR calculation
                normal_pixels = np.sum(~gt_binary)
                fp = np.sum(pred & ~gt_binary)
                total_fp += fp
                total_normal_pixels += normal_pixels

                # PRO calculation - per region overlap
                labeled_gt, num_regions = ndi.label(gt_binary)

                for region_id in range(1, num_regions + 1):
                    region_mask = labeled_gt == region_id
                    region_size = np.sum(region_mask)

                    if region_size > 0:
                        intersection = np.sum(pred & region_mask)
                        overlap = intersection / region_size
                        all_region_overlaps.append(overlap)

            # Aggregate
            fpr = total_fp / (total_normal_pixels + 1e-8)
            pro = np.mean(all_region_overlaps) if all_region_overlaps else 1.0

            fpr_values.append(fpr)
            pro_values.append(pro)

        fpr_values = np.array(fpr_values)
        pro_values = np.array(pro_values)

        # Compute AU-PRO (area under PRO curve up to fpr_limit)
        valid_idx = fpr_values <= self.fpr_limit

        if np.sum(valid_idx) > 1:
            sorted_idx = np.argsort(fpr_values[valid_idx])
            fpr_sorted = fpr_values[valid_idx][sorted_idx]
            pro_sorted = pro_values[valid_idx][sorted_idx]
            au_pro = np.trapz(pro_sorted, fpr_sorted) / self.fpr_limit
        else:
            au_pro = 0.0

        return {
            "fpr": fpr_values,
            "pro": pro_values,
            "thresholds": thresholds,
            "au_pro": au_pro,
            "fpr_limit": self.fpr_limit,
            "curve_type": "pro",
        }

    def compute_all_curves(
        self,
        anomaly_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
        normalize_maps: bool = True,
    ) -> dict:
        """
        Computes all available curves (ROC, PR, PRO, IoU).

        Args:
            anomaly_maps: List of anomaly maps
            gt_masks: List of ground truth masks
            normalize_maps: If True, normalize maps

        Returns:
            dict with all curve data
        """
        return {
            "roc": self.compute_roc_curve(anomaly_maps, gt_masks, normalize_maps),
            "pr": self.compute_pr_curve(anomaly_maps, gt_masks, normalize_maps),
            "pro": self.compute_pro_curve(anomaly_maps, gt_masks, normalize_maps),
            "iou": self.compute_iou_curve(anomaly_maps, gt_masks, normalize_maps),
        }

    def plot_curve(
        self,
        curve_data: dict,
        ax=None,
        title: str = None,
        color: str = None,
        label: str = None,
        show_auc: bool = True,
    ):
        """
        Plots a single curve.

        Args:
            curve_data: Curve data from compute_*_curve methods
            ax: Matplotlib axes (creates new figure if None)
            title: Plot title
            color: Line color
            label: Legend label
            show_auc: If True, show AUC value in legend

        Returns:
            matplotlib figure and axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = ax.figure

        curve_type = curve_data.get("curve_type", "")

        if curve_type == "roc":
            x, y = curve_data["fpr"], curve_data["tpr"]
            auc_val = curve_data["au_roc"]
            xlabel, ylabel = "False Positive Rate", "True Positive Rate"
            default_title = "ROC Curve"
            # Add diagonal reference
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

        elif curve_type == "pr":
            x, y = curve_data["recall"], curve_data["precision"]
            auc_val = curve_data["au_pr"]
            xlabel, ylabel = "Recall", "Precision"
            default_title = "Precision-Recall Curve"

        elif curve_type == "pro":
            x, y = curve_data["fpr"], curve_data["pro"]
            auc_val = curve_data["au_pro"]
            xlabel, ylabel = "False Positive Rate", "Per-Region Overlap"
            default_title = f'PRO Curve (FPR ≤ {curve_data["fpr_limit"]})'
            # Add FPR limit line
            ax.axvline(
                x=curve_data["fpr_limit"], color="gray", linestyle="--", alpha=0.5
            )

        elif curve_type == "iou":
            x, y = curve_data["thresholds"], curve_data["iou_values"]
            auc_val = curve_data["au_iou"]
            xlabel, ylabel = "Threshold", "IoU"
            default_title = "IoU vs Threshold"
            # Mark best point
            best_t = curve_data["best_threshold"]
            best_iou = curve_data["best_iou"]
            ax.scatter(
                [best_t],
                [best_iou],
                color="red",
                s=100,
                zorder=5,
                label=f"Best: τ={best_t:.2f}, IoU={best_iou:.3f}",
            )
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")

        # Build label
        if label is None:
            label = curve_type.upper()
        if show_auc:
            label = f"{label} (AUC={auc_val:.4f})"

        # Plot
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title or default_title, fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        return fig, ax

    def plot_all_curves(
        self,
        curves_dict: dict,
        title: str = None,
        save_path: str = None,
        dpi: int = 150,
        show: bool = True,
    ):
        """
        Plots all curves in a 2x2 grid.

        Args:
            curves_dict: Dict with 'roc', 'pr', 'pro', 'iou' curve data
            title: Overall figure title
            save_path: Path to save figure (optional)
            dpi: DPI for saved image
            show: If True, display the plot

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot each curve
        curve_configs = [
            ("roc", axes[0, 0], "#1f77b4"),
            ("pr", axes[0, 1], "#2ca02c"),
            ("pro", axes[1, 0], "#d62728"),
            ("iou", axes[1, 1], "#9467bd"),
        ]

        for curve_name, ax, color in curve_configs:
            if curve_name in curves_dict:
                self.plot_curve(curves_dict[curve_name], ax=ax, color=color)

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            import os

            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"📊 Curves saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_curves_comparison(
        self,
        curves_list: List[dict],
        labels: List[str],
        curve_type: str = "roc",
        title: str = None,
        save_path: str = None,
        dpi: int = 150,
        show: bool = True,
    ):
        """
        Plots multiple curves of the same type for comparison.

        Args:
            curves_list: List of curve data dicts
            labels: Labels for each curve
            curve_type: Type of curve ('roc', 'pr', 'pro', 'iou')
            title: Plot title
            save_path: Path to save figure
            dpi: DPI for saved image
            show: If True, display plot

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(curves_list)))

        for i, (curves_dict, label) in enumerate(zip(curves_list, labels)):
            if curve_type in curves_dict:
                self.plot_curve(
                    curves_dict[curve_type], ax=ax, color=colors[i], label=label
                )

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        ax.legend(loc="best")
        plt.tight_layout()

        if save_path:
            import os

            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"📊 Comparison curves saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def export_curve_data(
        self, curves_dict: dict, output_path: str, format: str = "json"
    ):
        """
        Exports curve data to file.

        Args:
            curves_dict: Dict with curve data
            output_path: Path to save data
            format: 'json' or 'csv'
        """
        import os

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        if format == "json":
            import json

            # Convert numpy arrays to lists
            def to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_serializable(v) for v in obj]
                return obj

            data = to_serializable(curves_dict)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            # Export each curve type to separate CSV sections
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                for curve_name, curve_data in curves_dict.items():
                    writer.writerow([f"# {curve_name.upper()} Curve"])

                    if curve_name == "roc":
                        writer.writerow(["fpr", "tpr", "threshold"])
                        for fpr, tpr, t in zip(
                            curve_data["fpr"],
                            curve_data["tpr"],
                            np.append(curve_data["thresholds"], np.nan),
                        ):
                            writer.writerow([fpr, tpr, t])
                        writer.writerow([f'# AU-ROC: {curve_data["au_roc"]:.4f}'])

                    elif curve_name == "pr":
                        writer.writerow(["recall", "precision", "threshold"])
                        for r, p, t in zip(
                            curve_data["recall"],
                            curve_data["precision"],
                            np.append(curve_data["thresholds"], np.nan),
                        ):
                            writer.writerow([r, p, t])
                        writer.writerow([f'# AU-PR: {curve_data["au_pr"]:.4f}'])

                    elif curve_name == "pro":
                        writer.writerow(["fpr", "pro", "threshold"])
                        for fpr, pro, t in zip(
                            curve_data["fpr"],
                            curve_data["pro"],
                            curve_data["thresholds"],
                        ):
                            writer.writerow([fpr, pro, t])
                        writer.writerow([f'# AU-PRO: {curve_data["au_pro"]:.4f}'])

                    elif curve_name == "iou":
                        writer.writerow(["threshold", "iou"])
                        for t, iou in zip(
                            curve_data["thresholds"], curve_data["iou_values"]
                        ):
                            writer.writerow([t, iou])
                        writer.writerow([f'# AU-IoU: {curve_data["au_iou"]:.4f}'])
                        writer.writerow(
                            [
                                f'# Best: threshold={curve_data["best_threshold"]:.4f}, IoU={curve_data["best_iou"]:.4f}'
                            ]
                        )

                    writer.writerow([])  # Empty line between curves

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")

        print(f"💾 Curve data exported to: {output_path}")

    def generate_evaluation_report(
        self,
        curves_dict: dict,
        category: str = None,
        output_dir: str = None,
        prefix: str = "curves",
    ) -> dict:
        """
        Generates a complete evaluation report with curves.

        Saves:
        - Combined curves plot (PNG)
        - Individual curve plots (PNG)
        - Curve data (JSON)

        Args:
            curves_dict: Dict with curve data
            category: Category name (for filename)
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Dict with paths to generated files
        """
        import os

        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        name_suffix = f"_{category}" if category else ""
        generated_files = {}

        # 1. Combined curves plot
        combined_path = os.path.join(output_dir, f"{prefix}{name_suffix}_all.png")
        self.plot_all_curves(
            curves_dict,
            title=f"Evaluation Curves{' - ' + category.upper() if category else ''}",
            save_path=combined_path,
            show=False,
        )
        generated_files["combined_plot"] = combined_path

        # 2. Individual curve plots
        for curve_name in ["roc", "pr", "pro", "iou"]:
            if curve_name in curves_dict:
                curve_path = os.path.join(
                    output_dir, f"{prefix}{name_suffix}_{curve_name}.png"
                )
                fig, ax = self.plot_curve(curves_dict[curve_name])
                fig.savefig(curve_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                generated_files[f"{curve_name}_plot"] = curve_path

        # 3. Export curve data
        json_path = os.path.join(output_dir, f"{prefix}{name_suffix}_data.json")
        self.export_curve_data(curves_dict, json_path, format="json")
        generated_files["curve_data"] = json_path

        # 4. Summary metrics
        summary = {
            "category": category,
            "metrics": {
                "AU-ROC": curves_dict.get("roc", {}).get("au_roc", None),
                "AU-PR": curves_dict.get("pr", {}).get("au_pr", None),
                "AU-PRO": curves_dict.get("pro", {}).get("au_pro", None),
                "AU-IoU": curves_dict.get("iou", {}).get("au_iou", None),
                "Best IoU": curves_dict.get("iou", {}).get("best_iou", None),
                "Best Threshold": curves_dict.get("iou", {}).get(
                    "best_threshold", None
                ),
            },
        }

        print(f"\n📈 Evaluation Report Generated:")
        print(
            f"   AU-ROC: {summary['metrics']['AU-ROC']:.4f}"
            if summary["metrics"]["AU-ROC"]
            else ""
        )
        print(
            f"   AU-PR:  {summary['metrics']['AU-PR']:.4f}"
            if summary["metrics"]["AU-PR"]
            else ""
        )
        print(
            f"   AU-PRO: {summary['metrics']['AU-PRO']:.4f}"
            if summary["metrics"]["AU-PRO"]
            else ""
        )
        print(
            f"   AU-IoU: {summary['metrics']['AU-IoU']:.4f}"
            if summary["metrics"]["AU-IoU"]
            else ""
        )
        print(
            f"   Best IoU: {summary['metrics']['Best IoU']:.4f} @ τ={summary['metrics']['Best Threshold']:.2f}"
            if summary["metrics"]["Best IoU"]
            else ""
        )

        return {"files": generated_files, "summary": summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MVTec AD Anomaly Detection con DINOv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de ejecución:
  demo              - Ejecuta demo con una sola imagen
  single_category   - Evalúa todas las anomalías de una categoría
  full_evaluation   - Evalúa TODAS las categorías y anomalías (completo)

Ejemplos:
  python eval.py --mode demo
  python eval.py --mode single_category --category bottle
  python eval.py --mode full_evaluation --n-good-images 50
  python eval.py --mode full_evaluation --categories bottle cable carpet
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "single_category", "full_evaluation"],
        help="Modo de evaluación (default: demo)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/bllancao/Portafolio/mvtec_anomaly_detection/models/dinov2-base",
        help="Ruta al modelo DINOv2",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/bllancao/Portafolio/mvtec_anomaly_detection/data/raw",
        help="Ruta base al dataset MVTec AD",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="Índice de capa DINOv2 (-1=última, 11=penúltima, etc.)",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="Número de vecinos más cercanos para k-NN"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Umbral para binarización (0-1)"
    )
    parser.add_argument(
        "--n-good-images",
        type=int,
        default=None,
        help='Número de imágenes "good" para memory bank (None=todas)',
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Categoría a evaluar (para modo single_category)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Lista de categorías a evaluar (para modo full_evaluation)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Ruta para guardar resultados JSON",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Ruta para guardar reporte Markdown",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="No guardar resultados en archivo"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progreso detallado",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Guardar visualizaciones de cada imagen evaluada",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=None,
        help="Directorio donde guardar visualizaciones (default: ./visualizations/)",
    )
    parser.add_argument(
        "--save-curves",
        action="store_true",
        help="Guardar curvas de evaluación (ROC, PR, PRO, IoU) para cada categoría",
    )

    args = parser.parse_args()

    # =========================================================================
    # MODO: DEMO (una sola imagen)
    # =========================================================================
    if args.mode == "demo":
        MODEL_PATH = args.model_path
        LAYER_IDX = args.layer_idx

        # Carpeta con imágenes "good" para el memory bank
        GOOD_FOLDER = f"{args.dataset_path}/bottle/train/good"
        N_GOOD_IMAGES = 20  # Número de imágenes "good" a cargar

        # Imagen de test (con defecto)
        TEST_IMAGE_PATH = f"{args.dataset_path}/bottle/test/broken_large/000.png"

        # Cargar imágenes "good" automáticamente
        good_images = load_images_from_folder(GOOD_FOLDER, n_images=N_GOOD_IMAGES)

        # Usar la primera imagen "good" como referencia para Dense Matching
        image_good = good_images[0]

        # Cargar imagen de test
        image_broken = Image.open(TEST_IMAGE_PATH).convert("RGB")

        print("=" * 60)
        print("MVTec AD Anomaly Detection con DINOv2 - DEMO")
        print("=" * 60)

        # Crear extractor con capa configurable
        extractor = DINOv2FeatureExtractor(model_path=MODEL_PATH, layer_idx=LAYER_IDX)
        print(f"\nExtractor creado - Capa: {LAYER_IDX}, Device: {extractor.device}")

        # -------------------------------------------------------------------------
        # MÉTODO 1: Dense Matching
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("MÉTODO 1: Dense Matching (Posicional)")
        print("-" * 60)

        dense_detector = DenseMatchingDetector(extractor)

        # Calcular mapa de anomalía
        amap_dense, amap_dense_smooth = dense_detector.compute_anomaly_map(
            test_image=image_broken, reference_image=image_good
        )

        print(f"Mapa de anomalía shape: {amap_dense.shape}")
        print(f"Score máximo: {amap_dense_smooth.max():.4f}")

        # Visualizar
        dense_detector.visualize(
            test_image=image_broken,
            reference_image=image_good,
            anomaly_map=amap_dense_smooth,
            title=f"Dense Matching (Capa {LAYER_IDX})",
        )

        # -------------------------------------------------------------------------
        # MÉTODO 2: Memory Bank + k-NN
        # -------------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("MÉTODO 2: Memory Bank + k-NN (PatchCore-style)")
        print("-" * 60)

        memory_detector = MemoryBankDetector(
            extractor=extractor, k=1, coreset_ratio=1.0
        )

        print("\nConstruyendo Memory Bank...")
        memory_detector.build_memory_bank(good_images, verbose=True)

        # Calcular mapa de anomalía
        amap_memory, amap_memory_smooth, image_score = (
            memory_detector.compute_anomaly_map(test_image=image_broken)
        )

        print(f"Mapa de anomalía shape: {amap_memory.shape}")
        print(f"Score de imagen: {image_score:.4f}")

        # Visualizar
        memory_detector.visualize(
            test_image=image_broken,
            anomaly_map=amap_memory_smooth,
            image_score=image_score,
            title=f"Memory Bank + k-NN (Capa {LAYER_IDX})",
        )

        print("\n✅ Demo completada")

    # =========================================================================
    # MODO: SINGLE CATEGORY (una categoría, todas las anomalías)
    # =========================================================================
    elif args.mode == "single_category":
        print("=" * 80)
        print(f"🔬 EVALUACIÓN DE CATEGORÍA: {args.category.upper()}")
        print("=" * 80)

        # Crear evaluador
        evaluator_mvtec = MVTecDatasetEvaluator(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            layer_idx=args.layer_idx,
            n_good_images=args.n_good_images,
            k=args.k,
            threshold=args.threshold,
        )

        # Evaluar categoría
        results = evaluator_mvtec.evaluate_category(
            category=args.category, verbose=args.verbose
        )

        # Mostrar resumen
        print("\n" + "=" * 80)
        print(f"📊 RESUMEN DE {args.category.upper()}")
        print("=" * 80)

        summary = results["summary"]
        print(f"\n  Total imágenes: {summary.get('n_images', 0)}")
        print(f"  Tipos de anomalía: {summary.get('n_anomaly_types', 0)}")
        print(f"\n  Métricas promedio:")
        print(f"    IoU:       {summary.get('IoU', 0):.4f}")
        print(f"    Dice:      {summary.get('Dice', 0):.4f}")
        print(f"    F1:        {summary.get('F1', 0):.4f}")
        print(f"    Precision: {summary.get('Precision', 0):.4f}")
        print(f"    Recall:    {summary.get('Recall', 0):.4f}")
        print(f"    AU-PRO:    {summary.get('AU-PRO', 0):.4f}")

        # Detalle por tipo de anomalía
        print("\n  Detalle por tipo de anomalía:")
        print(f"  {'Tipo':<20} {'IoU':<10} {'F1':<10} {'AU-PRO':<10}")
        print("  " + "-" * 50)
        for anomaly_type, anom_data in results["anomaly_results"].items():
            s = anom_data["summary"]
            print(
                f"  {anomaly_type:<20} {s.get('IoU', 0):<10.4f} "
                f"{s.get('F1', 0):<10.4f} {s.get('AU-PRO', 0):<10.4f}"
            )

        print("\n✅ Evaluación de categoría completada")

    # =========================================================================
    # MODO: FULL EVALUATION (todas las categorías, todas las anomalías)
    # =========================================================================
    elif args.mode == "full_evaluation":
        print("=" * 80)
        print("🚀 EVALUACIÓN COMPLETA DEL DATASET MVTEC AD")
        print("=" * 80)

        # Crear evaluador
        evaluator_mvtec = MVTecDatasetEvaluator(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            layer_idx=args.layer_idx,
            n_good_images=args.n_good_images,
            k=args.k,
            threshold=args.threshold,
        )

        # Evaluar todas las categorías (o las especificadas)
        results = evaluator_mvtec.evaluate_all(
            categories=args.categories,
            verbose=args.verbose,
            save_results=not args.no_save,
            output_path=args.output_path,
            save_visualizations=args.save_visualizations,
            visualization_dir=args.visualization_dir,
            save_curves=args.save_curves,
        )

        # Generar reporte Markdown si se especifica
        if args.report_path:
            evaluator_mvtec.generate_report(results, output_path=args.report_path)

        print("\n✅ Evaluación completa finalizada")

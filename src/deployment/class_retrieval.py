"""
Image Retrieval para identificación automática de clase en MVTec AD.

En producción, no sabemos a qué clase pertenece una imagen.
Este módulo usa el CLS token de DINOv2 para identificar la clase
más probable antes de aplicar el detector de anomalías correcto.

Pipeline completo:
1. Extraer CLS token de la imagen de entrada
2. Comparar contra CLS tokens de entrenamiento de cada clase
3. Identificar la clase más similar (k-NN sobre CLS tokens)
4. Usar el memory bank de esa clase para detección de anomalías

Uso:
    from class_retrieval import ClassRetriever, AutoAnomalyDetector

    # Opción 1: Solo clasificación
    retriever = ClassRetriever('./exported')
    class_name, confidence = retriever.identify_class(image)

    # Opción 2: Pipeline completo (clasificación + detección)
    detector = AutoAnomalyDetector('./exported')
    result = detector.predict(image)
    # result = {'class': 'bottle', 'confidence': 0.95, 'anomaly_score': 0.3, ...}
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# CLASS RETRIEVER - Identificación de clase usando CLS tokens
# =============================================================================


class ClassRetriever:
    """
    Identificador de clase usando CLS tokens de DINOv2.

    Compara el CLS token de una imagen contra los CLS tokens de las
    imágenes de entrenamiento de cada clase para identificar la clase
    más probable.

    Args:
        exported_dir: Directorio con archivos exportados
        k: Número de vecinos para votación (default: 5)
        providers: Providers de ONNX Runtime
    """

    def __init__(
        self, exported_dir: str, k: int = 5, providers: Optional[List[str]] = None
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no instalado. pip install onnxruntime")

        self.exported_dir = exported_dir
        self.k = k

        # Cargar modelo ONNX para extracción de CLS
        onnx_path = os.path.join(exported_dir, "dinov2_cls_extractor.onnx")

        # Si no existe el modelo CLS específico, usar el de patches
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(exported_dir, "dinov2_feature_extractor.onnx")
            self.use_patch_model = True
        else:
            self.use_patch_model = False

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {onnx_path}")

        # Configurar ONNX Runtime
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Cargar metadatos
        metadata_path = onnx_path.replace(".onnx", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"input_height": 518, "input_width": 518}

        self.input_height = self.metadata["input_height"]
        self.input_width = self.metadata["input_width"]

        # Cargar CLS tokens de todas las clases
        self.class_cls_tokens = {}
        self.class_labels = []
        self.all_cls_tokens = []
        self.all_labels = []

        self._load_class_cls_tokens()

        print(f"✅ ClassRetriever inicializado")
        print(f"   Clases: {list(self.class_cls_tokens.keys())}")
        print(f"   Total CLS tokens: {len(self.all_cls_tokens)}")
        print(f"   K: {self.k}")

    def _load_class_cls_tokens(self):
        """Carga los CLS tokens de entrenamiento de cada clase."""
        for class_name in os.listdir(self.exported_dir):
            class_dir = os.path.join(self.exported_dir, class_name)
            cls_path = os.path.join(class_dir, "cls_tokens.npy")

            if os.path.exists(cls_path):
                cls_tokens = np.load(cls_path)
                self.class_cls_tokens[class_name] = cls_tokens

                # Agregar a lista global con etiquetas
                for token in cls_tokens:
                    self.all_cls_tokens.append(token)
                    self.all_labels.append(class_name)

        if self.all_cls_tokens:
            self.all_cls_tokens = np.array(self.all_cls_tokens)
        else:
            raise ValueError(
                "No se encontraron CLS tokens. Ejecuta export con --export_cls"
            )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocesa imagen para el modelo."""
        image_resized = image.resize(
            (self.input_width, self.input_height), Image.Resampling.BILINEAR
        )

        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = img_normalized.transpose(2, 0, 1)
        pixel_values = np.expand_dims(img_chw, axis=0).astype(np.float32)

        return pixel_values

    def extract_cls_token(self, image: Image.Image) -> np.ndarray:
        """
        Extrae el CLS token de una imagen.

        Si usamos el modelo de patches, simulamos CLS promediando patches.
        """
        pixel_values = self.preprocess(image)

        outputs = self.session.run([self.output_name], {self.input_name: pixel_values})

        if self.use_patch_model:
            # Promediar patches como aproximación del CLS
            # [1, num_patches, hidden_dim] -> [hidden_dim]
            patch_embeddings = outputs[0][0]
            cls_token = patch_embeddings.mean(axis=0)
        else:
            # [1, hidden_dim] -> [hidden_dim]
            cls_token = outputs[0][0]

        # Normalizar
        cls_token = cls_token / (np.linalg.norm(cls_token) + 1e-8)

        return cls_token

    def identify_class(
        self, image: Image.Image, return_all_scores: bool = False
    ) -> Tuple[str, float]:
        """
        Identifica la clase más probable para una imagen.

        Usa k-NN con votación ponderada por similitud.

        Args:
            image: Imagen PIL RGB
            return_all_scores: Si True, retorna scores para todas las clases

        Returns:
            class_name: Nombre de la clase predicha
            confidence: Confianza de la predicción [0, 1]
            (all_scores): Diccionario con scores por clase (si return_all_scores)
        """
        # Extraer CLS token
        query_cls = self.extract_cls_token(image)

        # Similitud coseno con todos los CLS tokens
        similarities = np.dot(self.all_cls_tokens, query_cls)

        # Obtener k vecinos más similares
        topk_idx = np.argsort(similarities)[-self.k :][::-1]
        topk_sim = similarities[topk_idx]
        topk_labels = [self.all_labels[i] for i in topk_idx]

        # Votación ponderada por similitud
        class_scores = {}
        for label, sim in zip(topk_labels, topk_sim):
            class_scores[label] = class_scores.get(label, 0) + sim

        # Normalizar scores
        total_score = sum(class_scores.values())
        for label in class_scores:
            class_scores[label] /= total_score

        # Clase ganadora
        predicted_class = max(class_scores, key=class_scores.get)
        confidence = class_scores[predicted_class]

        if return_all_scores:
            return predicted_class, confidence, class_scores

        return predicted_class, confidence

    def identify_class_batch(
        self, images: List[Image.Image]
    ) -> List[Tuple[str, float]]:
        """Identifica clases para un batch de imágenes."""
        return [self.identify_class(img) for img in images]


# =============================================================================
# AUTO ANOMALY DETECTOR - Pipeline completo
# =============================================================================


class AutoAnomalyDetector:
    """
    Detector de anomalías con identificación automática de clase.

    Pipeline:
    1. Identifica la clase usando ClassRetriever (CLS tokens)
    2. Carga el memory bank de esa clase
    3. Detecta anomalías usando k-NN sobre patch embeddings

    Args:
        exported_dir: Directorio con archivos exportados
        retrieval_k: K para clasificación de clase
        anomaly_k: K para detección de anomalías
        min_confidence: Confianza mínima para proceder con detección
        providers: Providers de ONNX Runtime
    """

    def __init__(
        self,
        exported_dir: str,
        retrieval_k: int = 5,
        anomaly_k: int = 1,
        min_confidence: float = 0.3,
        providers: Optional[List[str]] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no instalado")

        self.exported_dir = exported_dir
        self.anomaly_k = anomaly_k
        self.min_confidence = min_confidence

        # Inicializar retriever
        self.retriever = ClassRetriever(
            exported_dir=exported_dir, k=retrieval_k, providers=providers
        )

        # Cargar sesión ONNX para patches
        onnx_path = os.path.join(exported_dir, "dinov2_feature_extractor.onnx")

        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Cargar metadatos
        metadata_path = onnx_path.replace(".onnx", "_metadata.json")
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.input_height = self.metadata["input_height"]
        self.input_width = self.metadata["input_width"]
        self.n_patches_h = self.metadata["n_patches_h"]
        self.n_patches_w = self.metadata["n_patches_w"]

        # Cache para memory banks (lazy loading)
        self._memory_banks = {}

        # Lista de clases disponibles
        self.available_classes = list(self.retriever.class_cls_tokens.keys())

        print(f"✅ AutoAnomalyDetector inicializado")
        print(f"   Clases: {self.available_classes}")
        print(f"   Min confidence: {self.min_confidence}")

    def _get_memory_bank(self, class_name: str) -> np.ndarray:
        """Obtiene memory bank para una clase (con cache)."""
        if class_name not in self._memory_banks:
            mb_path = os.path.join(self.exported_dir, class_name, "memory_bank.npy")
            if not os.path.exists(mb_path):
                raise FileNotFoundError(f"Memory bank no encontrado: {mb_path}")
            self._memory_banks[class_name] = np.load(mb_path)
        return self._memory_banks[class_name]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocesa imagen."""
        image_resized = image.resize(
            (self.input_width, self.input_height), Image.Resampling.BILINEAR
        )

        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_chw = img_normalized.transpose(2, 0, 1)
        pixel_values = np.expand_dims(img_chw, axis=0).astype(np.float32)

        return pixel_values

    def extract_patches(self, image: Image.Image) -> np.ndarray:
        """Extrae patch embeddings."""
        pixel_values = self.preprocess(image)

        outputs = self.session.run([self.output_name], {self.input_name: pixel_values})

        return outputs[0][0]  # [num_patches, hidden_dim]

    def compute_anomaly_map(
        self,
        patch_embeddings: np.ndarray,
        memory_bank: np.ndarray,
        smooth_sigma: float = 0.8,
    ) -> Tuple[np.ndarray, float]:
        """Calcula mapa de anomalía usando k-NN."""
        sim_matrix = np.dot(patch_embeddings, memory_bank.T)

        topk_indices = np.argpartition(sim_matrix, -self.anomaly_k, axis=1)[
            :, -self.anomaly_k :
        ]
        topk_sim = np.take_along_axis(sim_matrix, topk_indices, axis=1)

        anomaly_scores = 1 - topk_sim.mean(axis=1)

        anomaly_map = anomaly_scores.reshape(self.n_patches_h, self.n_patches_w)
        anomaly_map_smooth = ndimage.gaussian_filter(anomaly_map, sigma=smooth_sigma)

        image_score = float(anomaly_map_smooth.max())

        return anomaly_map_smooth, image_score

    def predict(
        self,
        image: Image.Image,
        smooth_sigma: float = 0.8,
        force_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predice anomalías con identificación automática de clase.

        Args:
            image: Imagen PIL RGB
            smooth_sigma: Sigma para suavizado del mapa
            force_class: Forzar una clase específica (omite retrieval)

        Returns:
            result: Diccionario con:
                - class_name: Clase identificada
                - class_confidence: Confianza en la clasificación
                - anomaly_score: Score de anomalía [0, 1]
                - anomaly_map: Mapa de anomalía [H, W]
                - is_anomaly: Bool basado en umbral automático
                - all_class_scores: Scores para todas las clases
        """
        # Paso 1: Identificar clase
        if force_class:
            class_name = force_class
            class_confidence = 1.0
            all_class_scores = {force_class: 1.0}
        else:
            class_name, class_confidence, all_class_scores = (
                self.retriever.identify_class(image, return_all_scores=True)
            )

        result = {
            "class_name": class_name,
            "class_confidence": class_confidence,
            "all_class_scores": all_class_scores,
        }

        # Verificar confianza mínima
        if class_confidence < self.min_confidence:
            result["warning"] = (
                f"Confianza baja: {class_confidence:.2f} < {self.min_confidence}"
            )
            result["anomaly_score"] = None
            result["anomaly_map"] = None
            result["is_anomaly"] = None
            return result

        # Paso 2: Obtener memory bank de la clase
        memory_bank = self._get_memory_bank(class_name)

        # Paso 3: Extraer patches y calcular anomalía
        patch_embeddings = self.extract_patches(image)
        anomaly_map, anomaly_score = self.compute_anomaly_map(
            patch_embeddings, memory_bank, smooth_sigma
        )

        result["anomaly_score"] = anomaly_score
        result["anomaly_map"] = anomaly_map

        # Umbral adaptativo basado en estadísticas del memory bank
        # Por ahora usamos un umbral fijo
        result["is_anomaly"] = anomaly_score > 0.3

        return result

    def predict_batch(
        self, images: List[Image.Image], smooth_sigma: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Predice para un batch de imágenes."""
        return [self.predict(img, smooth_sigma) for img in images]

    def upsample_anomaly_map(
        self, anomaly_map: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Escala mapa de anomalía al tamaño original."""
        import cv2

        return cv2.resize(
            anomaly_map.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def visualize(
        self,
        image: Image.Image,
        result: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """Visualiza resultado de predicción."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Imagen original
        axes[0].imshow(image)
        axes[0].set_title("Imagen Original")
        axes[0].axis("off")

        # 2. Scores de clase
        if result.get("all_class_scores"):
            classes = list(result["all_class_scores"].keys())
            scores = [result["all_class_scores"][c] for c in classes]
            colors = ["green" if c == result["class_name"] else "gray" for c in classes]

            axes[1].barh(classes, scores, color=colors)
            axes[1].set_xlim(0, 1)
            axes[1].set_title(
                f"Clasificación\n{result['class_name']} ({result['class_confidence']:.2f})"
            )

        # 3. Mapa de anomalía
        if result.get("anomaly_map") is not None:
            amap = result["anomaly_map"]
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

            axes[2].imshow(amap_norm, cmap="jet")
            axes[2].set_title(f"Mapa de Anomalía\nScore: {result['anomaly_score']:.4f}")
            axes[2].axis("off")

            # 4. Overlay
            amap_up = self.upsample_anomaly_map(amap_norm, (image.height, image.width))
            axes[3].imshow(image)
            axes[3].imshow(amap_up, cmap="jet", alpha=0.5)
            status = "ANOMALÍA" if result.get("is_anomaly") else "NORMAL"
            axes[3].set_title(f"Overlay\n{status}")
            axes[3].axis("off")
        else:
            axes[2].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
            axes[2].axis("off")
            axes[3].text(
                0.5,
                0.5,
                result.get("warning", "Error"),
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 Guardado: {save_path}")

        plt.show()
        return fig


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detección de anomalías con clasificación automática"
    )

    parser.add_argument(
        "--exported_dir",
        type=str,
        required=True,
        help="Directorio con archivos exportados",
    )

    parser.add_argument(
        "--image_path", type=str, required=True, help="Ruta a la imagen"
    )

    parser.add_argument(
        "--retrieval_k", type=int, default=5, help="K para clasificación"
    )

    parser.add_argument(
        "--anomaly_k", type=int, default=1, help="K para detección de anomalías"
    )

    parser.add_argument("--output", type=str, help="Ruta para guardar visualización")

    parser.add_argument("--force_class", type=str, help="Forzar clase específica")

    args = parser.parse_args()

    # Cargar detector
    detector = AutoAnomalyDetector(
        exported_dir=args.exported_dir,
        retrieval_k=args.retrieval_k,
        anomaly_k=args.anomaly_k,
    )

    # Cargar imagen
    image = Image.open(args.image_path).convert("RGB")

    # Predecir
    result = detector.predict(image, force_class=args.force_class)

    print(f"\n📊 Resultados:")
    print(f"   Clase: {result['class_name']} (conf: {result['class_confidence']:.2f})")

    if result.get("anomaly_score") is not None:
        print(f"   Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"   Es anomalía: {'Sí' if result['is_anomaly'] else 'No'}")
    else:
        print(f"   ⚠️ {result.get('warning', 'Error')}")

    # Visualizar
    if args.output:
        detector.visualize(image, result, args.output)


if __name__ == "__main__":
    main()

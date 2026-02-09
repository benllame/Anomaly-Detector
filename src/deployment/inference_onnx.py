"""
Inferencia con modelo ONNX exportado para detección de anomalías MVTec AD.

Este módulo proporciona una clase de inferencia ligera que no requiere
PyTorch en producción, solo ONNX Runtime.

Soporta la estructura del dataset MVTec AD:
- Un modelo ONNX compartido
- Memory banks separados por clase (bottle, cable, etc.)
- Evaluación por defecto type

Uso típico:
    from inference_onnx import MVTecONNXDetector

    # Cargar detector para una clase específica
    detector = MVTecONNXDetector(
        exported_dir='./exported',
        class_name='bottle'
    )
    
    # Predecir anomalía
    anomaly_map, score = detector.predict(image)
    
    # Evaluar contra ground truth
    metrics = detector.evaluate(image, gt_mask)
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import scipy.ndimage as ndimage


# =============================================================================
# CONFIGURACIÓN DE PREPROCESSING
# =============================================================================

# ImageNet normalization (usado por DINOv2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# DETECTOR DE ANOMALÍAS MVTEC AD CON ONNX
# =============================================================================

class MVTecONNXDetector:
    """
    Detector de anomalías para MVTec AD usando modelo ONNX y memory bank k-NN.
    
    Diseñado para cargar fácilmente el modelo exportado para una clase específica
    del dataset MVTec AD.
    
    Args:
        exported_dir: Directorio con los archivos exportados (output de export_onnx.py)
        class_name: Nombre de la clase a evaluar (e.g., 'bottle')
        k: Número de vecinos para k-NN (opcional, usa config si no se especifica)
        providers: Providers de ONNX Runtime (default: auto-detect GPU/CPU)
    
    Ejemplo:
        detector = MVTecONNXDetector('./exported', 'bottle')
        anomaly_map, score = detector.predict(test_image)
    """
    
    def __init__(
        self,
        exported_dir: str,
        class_name: str,
        k: Optional[int] = None,
        providers: Optional[List[str]] = None
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime no está instalado. Instala con:\n"
                "  pip install onnxruntime       # CPU\n"
                "  pip install onnxruntime-gpu   # GPU (CUDA)"
            )
        
        self.exported_dir = exported_dir
        self.class_name = class_name
        
        # Rutas de archivos
        onnx_path = os.path.join(exported_dir, 'dinov2_feature_extractor.onnx')
        class_dir = os.path.join(exported_dir, class_name)
        memory_bank_path = os.path.join(class_dir, 'memory_bank.npy')
        config_path = os.path.join(class_dir, 'detector_config.json')
        
        # Verificar archivos
        for path, name in [(onnx_path, 'Modelo ONNX'), 
                           (memory_bank_path, 'Memory bank'),
                           (config_path, 'Config')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} no encontrado: {path}")
        
        # Cargar configuración
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.k = k if k is not None else self.config.get('k', 1)
        
        # Configurar providers de ONNX Runtime
        if providers is None:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        self.providers = providers
        
        # Cargar sesión ONNX
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        # Obtener info del modelo
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Cargar metadatos del modelo
        metadata_path = onnx_path.replace('.onnx', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.model_metadata = json.load(f)
        else:
            self.model_metadata = {
                'input_height': 518,
                'input_width': 518,
                'patch_size': 14,
                'n_patches_h': 37,
                'n_patches_w': 37
            }
        
        self.input_height = self.model_metadata['input_height']
        self.input_width = self.model_metadata['input_width']
        self.patch_size = self.model_metadata['patch_size']
        self.n_patches_h = self.model_metadata['n_patches_h']
        self.n_patches_w = self.model_metadata['n_patches_w']
        
        # Cargar memory bank de esta clase
        self.memory_bank = np.load(memory_bank_path)
        
        # Tipos de defectos disponibles
        self.defect_types = self.config.get('defect_types', [])
        
        print(f"✅ MVTecONNXDetector inicializado")
        print(f"   Clase: {class_name}")
        print(f"   Providers: {providers}")
        print(f"   Memory bank: {self.memory_bank.shape}")
        print(f"   K: {self.k}")
        if self.defect_types:
            print(f"   Defects: {', '.join(self.defect_types)}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesa una imagen PIL para el modelo.
        
        Args:
            image: Imagen PIL RGB
            
        Returns:
            pixel_values: Array [1, 3, H, W] normalizado
        """
        # Resize
        image_resized = image.resize(
            (self.input_width, self.input_height),
            Image.Resampling.BILINEAR
        )
        
        # Convertir a numpy y normalizar a [0, 1]
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # ImageNet normalization
        img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        
        # Transponer a CHW y agregar batch dimension
        img_chw = img_normalized.transpose(2, 0, 1)
        pixel_values = np.expand_dims(img_chw, axis=0).astype(np.float32)
        
        return pixel_values
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extrae embeddings de patches de una imagen.
        
        Args:
            image: Imagen PIL RGB
            
        Returns:
            patch_embeddings: Array [num_patches, hidden_dim]
        """
        pixel_values = self.preprocess(image)
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: pixel_values}
        )
        
        patch_embeddings = outputs[0][0]
        return patch_embeddings
    
    def compute_anomaly_scores(self, patch_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula scores de anomalía usando k-NN contra memory bank.
        
        Args:
            patch_embeddings: Embeddings [num_patches, hidden_dim]
            
        Returns:
            anomaly_scores: Array [num_patches]
        """
        sim_matrix = np.dot(patch_embeddings, self.memory_bank.T)
        
        topk_indices = np.argpartition(sim_matrix, -self.k, axis=1)[:, -self.k:]
        topk_sim = np.take_along_axis(sim_matrix, topk_indices, axis=1)
        
        anomaly_scores = 1 - topk_sim.mean(axis=1)
        return anomaly_scores
    
    def predict(
        self,
        image: Image.Image,
        smooth_sigma: float = 0.8,
        return_raw: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Predice mapa de anomalía para una imagen.
        
        Args:
            image: Imagen PIL RGB
            smooth_sigma: Sigma para suavizado Gaussiano
            return_raw: Si True, retorna también el mapa sin suavizar
            
        Returns:
            anomaly_map: Mapa de anomalía [H, W] suavizado
            image_score: Score de anomalía a nivel de imagen
            (anomaly_map_raw): Mapa sin suavizar (si return_raw=True)
        """
        patch_embeddings = self.extract_features(image)
        anomaly_scores = self.compute_anomaly_scores(patch_embeddings)
        
        anomaly_map = anomaly_scores.reshape(self.n_patches_h, self.n_patches_w)
        anomaly_map_smooth = ndimage.gaussian_filter(anomaly_map, sigma=smooth_sigma)
        
        image_score = float(anomaly_map_smooth.max())
        
        if return_raw:
            return anomaly_map_smooth, image_score, anomaly_map
        
        return anomaly_map_smooth, image_score
    
    def predict_batch(
        self,
        images: List[Image.Image],
        smooth_sigma: float = 0.8
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predice anomalías para un batch de imágenes.
        """
        anomaly_maps = []
        image_scores = []
        
        for image in images:
            amap, score = self.predict(image, smooth_sigma)
            anomaly_maps.append(amap)
            image_scores.append(score)
        
        return anomaly_maps, image_scores
    
    def upsample_anomaly_map(
        self,
        anomaly_map: np.ndarray,
        target_size: Tuple[int, int],
        mode: str = 'bilinear'
    ) -> np.ndarray:
        """
        Escala el mapa de anomalía al tamaño de la imagen/GT original.
        """
        import cv2
        
        interp = {
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC
        }.get(mode, cv2.INTER_LINEAR)
        
        upsampled = cv2.resize(
            anomaly_map.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=interp
        )
        
        return upsampled
    
    def normalize_anomaly_map(
        self,
        anomaly_map: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normaliza mapa de anomalía a [0, 1].
        """
        amap = anomaly_map.astype(np.float32)
        
        if method == 'minmax':
            amap_min, amap_max = amap.min(), amap.max()
            if amap_max > amap_min:
                return (amap - amap_min) / (amap_max - amap_min)
            return np.zeros_like(amap)
        
        elif method == 'robust':
            p_low = np.percentile(amap, 2)
            p_high = np.percentile(amap, 98)
            if p_high > p_low:
                normalized = (amap - p_low) / (p_high - p_low)
                return np.clip(normalized, 0, 1)
            return np.zeros_like(amap)
        
        raise ValueError(f"Método no válido: {method}")
    
    def evaluate(
        self,
        anomaly_map: np.ndarray,
        gt_mask: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evalúa el mapa de anomalía contra ground truth.
        
        Args:
            anomaly_map: Mapa de anomalía predicho
            gt_mask: Máscara ground truth binaria
            threshold: Umbral para binarización [0, 1]
            
        Returns:
            metrics: Diccionario con IoU, Dice, Precision, Recall, F1
        """
        # Resize si es necesario
        if anomaly_map.shape != gt_mask.shape:
            anomaly_map = self.upsample_anomaly_map(anomaly_map, gt_mask.shape)
        
        # Normalizar
        amap_norm = self.normalize_anomaly_map(anomaly_map)
        
        # Binarizar
        pred_mask = (amap_norm >= threshold).astype(bool)
        gt_binary = gt_mask.astype(bool)
        
        # Métricas
        tp = np.sum(pred_mask & gt_binary)
        fp = np.sum(pred_mask & ~gt_binary)
        fn = np.sum(~pred_mask & gt_binary)
        tn = np.sum(~pred_mask & ~gt_binary)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        dice = 2 * intersection / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        return {
            'IoU': iou,
            'Dice': dice,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        }
    
    @classmethod
    def load_ground_truth(cls, gt_path: str) -> np.ndarray:
        """
        Carga una máscara ground truth como array binario.
        """
        gt_image = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_image)
        return (gt_array > 127).astype(np.float32)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def list_available_classes(exported_dir: str) -> List[str]:
    """
    Lista las clases disponibles en el directorio exportado.
    """
    global_config_path = os.path.join(exported_dir, 'global_config.json')
    
    if os.path.exists(global_config_path):
        with open(global_config_path) as f:
            config = json.load(f)
        return [c for c in config.get('classes', []) if c != 'model_onnx']
    
    # Fallback: listar directorios con memory_bank.npy
    classes = []
    for d in os.listdir(exported_dir):
        class_dir = os.path.join(exported_dir, d)
        if os.path.isdir(class_dir):
            if os.path.exists(os.path.join(class_dir, 'memory_bank.npy')):
                classes.append(d)
    return sorted(classes)


def benchmark_inference(
    detector: MVTecONNXDetector,
    image: Image.Image,
    n_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark de velocidad de inferencia.
    """
    import time
    
    for _ in range(warmup):
        detector.predict(image)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        detector.predict(image)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'p50_ms': np.percentile(times, 50) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
        'p99_ms': np.percentile(times, 99) * 1000,
        'fps': 1 / times.mean()
    }


def evaluate_class(
    detector: MVTecONNXDetector,
    data_root: str,
    class_name: str,
    threshold: float = 0.5,
    defect_types: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evalúa el detector en todas las imágenes de test de una clase.
    
    Args:
        detector: Instancia de MVTecONNXDetector
        data_root: Ruta raíz del dataset MVTec AD
        class_name: Nombre de la clase
        threshold: Umbral para binarización
        defect_types: Lista de tipos de defectos a evaluar (None = todos)
        verbose: Si True, muestra progreso
        
    Returns:
        results: Diccionario con métricas por defect_type y promedio
    """
    test_path = os.path.join(data_root, class_name, 'test')
    gt_path = os.path.join(data_root, class_name, 'ground_truth')
    
    if defect_types is None:
        defect_types = [d for d in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, d)) and d != 'good']
    
    all_results = {}
    all_metrics = []
    
    for defect_type in defect_types:
        defect_test_path = os.path.join(test_path, defect_type)
        defect_gt_path = os.path.join(gt_path, defect_type)
        
        if not os.path.exists(defect_test_path):
            continue
        
        if verbose:
            print(f"   Evaluando {defect_type}...")
        
        metrics_list = []
        
        # Listar imágenes de test
        test_images = sorted([f for f in os.listdir(defect_test_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_file in test_images:
            # Cargar imagen
            img_path = os.path.join(defect_test_path, img_file)
            image = Image.open(img_path).convert('RGB')
            
            # Predecir
            anomaly_map, score = detector.predict(image)
            
            # Cargar GT
            gt_file = img_file.replace('.png', '_mask.png')
            gt_file_path = os.path.join(defect_gt_path, gt_file)
            
            if os.path.exists(gt_file_path):
                gt_mask = detector.load_ground_truth(gt_file_path)
                
                # Evaluar
                metrics = detector.evaluate(anomaly_map, gt_mask, threshold)
                metrics['score'] = score
                metrics_list.append(metrics)
        
        if metrics_list:
            # Promediar métricas para este defect_type
            avg_metrics = {}
            for key in ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'score']:
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
            avg_metrics['n_images'] = len(metrics_list)
            
            all_results[defect_type] = avg_metrics
            all_metrics.extend(metrics_list)
            
            if verbose:
                print(f"      IoU: {avg_metrics['IoU']:.3f}, F1: {avg_metrics['F1']:.3f}, "
                      f"n={avg_metrics['n_images']}")
    
    # Promedio global
    if all_metrics:
        global_avg = {}
        for key in ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'score']:
            values = [m[key] for m in all_metrics]
            global_avg[key] = np.mean(values)
        global_avg['n_images'] = len(all_metrics)
        all_results['_average'] = global_avg
        
        if verbose:
            print(f"   PROMEDIO: IoU: {global_avg['IoU']:.3f}, F1: {global_avg['F1']:.3f}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inferencia de anomalías MVTec AD con modelo ONNX'
    )
    
    parser.add_argument(
        '--exported_dir',
        type=str,
        required=True,
        help='Directorio con archivos exportados'
    )
    
    parser.add_argument(
        '--class_name',
        type=str,
        required=True,
        help='Nombre de la clase (e.g., bottle)'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        help='Ruta a la imagen de prueba'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        help='Ruta raíz del dataset MVTec AD (para evaluación completa)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='K para k-NN scoring'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Umbral de anomalía [0, 1]'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Ejecutar benchmark de velocidad'
    )
    
    parser.add_argument(
        '--evaluate_all',
        action='store_true',
        help='Evaluar todas las imágenes de test de la clase'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Ruta para guardar visualización'
    )
    
    parser.add_argument(
        '--list_classes',
        action='store_true',
        help='Listar clases disponibles y salir'
    )
    
    args = parser.parse_args()
    
    # Listar clases
    if args.list_classes:
        classes = list_available_classes(args.exported_dir)
        print(f"Clases disponibles en {args.exported_dir}:")
        for c in classes:
            print(f"  - {c}")
        return
    
    # Cargar detector
    detector = MVTecONNXDetector(
        exported_dir=args.exported_dir,
        class_name=args.class_name,
        k=args.k
    )
    
    # Evaluación completa
    if args.evaluate_all:
        if not args.data_root:
            parser.error("--data_root es requerido para --evaluate_all")
        
        print(f"\n📊 Evaluando clase: {args.class_name}")
        results = evaluate_class(
            detector=detector,
            data_root=args.data_root,
            class_name=args.class_name,
            threshold=args.threshold
        )
        return
    
    # Predicción simple
    if args.image_path:
        image = Image.open(args.image_path).convert('RGB')
        
        anomaly_map, score = detector.predict(image)
        
        print(f"\n📊 Resultados:")
        print(f"   Score: {score:.4f}")
        print(f"   Es anomalía: {'Sí' if score > args.threshold else 'No'} (τ={args.threshold})")
        
        if args.benchmark:
            print(f"\n⏱️ Benchmark (100 runs):")
            stats = benchmark_inference(detector, image)
            print(f"   Mean: {stats['mean_ms']:.2f} ms")
            print(f"   Std:  {stats['std_ms']:.2f} ms")
            print(f"   P95:  {stats['p95_ms']:.2f} ms")
            print(f"   FPS:  {stats['fps']:.1f}")
        
        if args.output:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image)
            axes[0].set_title('Imagen Original')
            axes[0].axis('off')
            
            amap_norm = detector.normalize_anomaly_map(anomaly_map)
            
            axes[1].imshow(amap_norm, cmap='jet')
            axes[1].set_title(f'Mapa de Anomalía\n(Score: {score:.4f})')
            axes[1].axis('off')
            
            amap_upsampled = detector.upsample_anomaly_map(
                amap_norm, 
                (image.height, image.width)
            )
            axes[2].imshow(image)
            axes[2].imshow(amap_upsampled, cmap='jet', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f'Clase: {args.class_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\n💾 Visualización guardada: {args.output}")


if __name__ == '__main__':
    main()

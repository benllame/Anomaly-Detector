"""
Visualización de Anomalías Detectadas vs Ground Truth
======================================================

Script para generar visualizaciones comparativas entre las
anomalías detectadas por el modelo y el ground truth real
para cada categoría y tipo de anomalía del dataset MVTec AD.

Uso:
    python visualize_anomalies.py --category bottle --n-samples 3
    python visualize_anomalies.py --all-categories --n-samples 2 --output-dir ./visualizations
"""

import argparse
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Importar componentes del módulo eval
from eval import (
    AnomalyEvaluator,
    DINOv2FeatureExtractor,
    MemoryBankDetector,
    MVTecDatasetEvaluator,
    load_images_from_folder,
    load_test_with_ground_truth,
    normalize_anomaly_map,
    resize_anomaly_map,
)
from PIL import Image


class AnomalyVisualizer:
    """
    Visualizador de anomalías para MVTec AD.

    Genera grids comparativos mostrando:
    - Imagen original
    - Ground Truth
    - Mapa de anomalía predicho
    - Predicción binarizada
    - Overlay comparativo
    """

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
        threshold: float = 0.5,
        output_dir: str = None,
    ):
        """
        Args:
            dataset_path: Ruta base al dataset MVTec AD
            model_path: Ruta al modelo DINOv2
            layer_idx: Índice de la capa a usar (-1 = última)
            n_good_images: Número de imágenes "good" para memory bank
            k: Número de vecinos para k-NN
            threshold: Umbral para binarización
            output_dir: Directorio para guardar visualizaciones
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.layer_idx = layer_idx
        self.n_good_images = n_good_images
        self.k = k
        self.threshold = threshold
        self.output_dir = output_dir or "./visualizations"

        # Crear directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)

        # Inicializar extractor
        print("🔧 Inicializando DINOv2 Feature Extractor...")
        self.extractor = DINOv2FeatureExtractor(
            model_path=model_path, layer_idx=layer_idx
        )

        # Evaluador para métricas
        self.evaluator = AnomalyEvaluator(threshold=threshold, auto_normalize=True)

        # Cache de memory banks por categoría
        self._memory_banks: Dict[str, MemoryBankDetector] = {}

    def _get_detector(self, category: str) -> MemoryBankDetector:
        """Obtiene o construye el detector para una categoría."""
        if category not in self._memory_banks:
            print(f"\n📦 Construyendo Memory Bank para '{category}'...")
            good_path = os.path.join(self.dataset_path, category, "train", "good")
            good_images = load_images_from_folder(
                good_path, n_images=self.n_good_images
            )

            detector = MemoryBankDetector(
                extractor=self.extractor, k=self.k, coreset_ratio=1.0
            )
            detector.build_memory_bank(good_images, verbose=True)
            self._memory_banks[category] = detector

        return self._memory_banks[category]

    def _get_anomaly_types(self, category: str) -> List[str]:
        """Obtiene los tipos de anomalía con ground truth disponible."""
        test_path = os.path.join(self.dataset_path, category, "test")
        gt_path = os.path.join(self.dataset_path, category, "ground_truth")

        anomaly_types = []
        if os.path.isdir(test_path):
            for folder in sorted(os.listdir(test_path)):
                folder_path = os.path.join(test_path, folder)
                gt_folder_path = os.path.join(gt_path, folder)
                if os.path.isdir(folder_path) and folder != "good":
                    if os.path.isdir(gt_folder_path):
                        anomaly_types.append(folder)
        return anomaly_types

    def visualize_single_image(
        self,
        test_image: Image.Image,
        gt_mask: np.ndarray,
        anomaly_map: np.ndarray,
        metrics: dict,
        title: str = None,
        save_path: str = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Visualiza una imagen con 5 paneles comparativos.

        Paneles:
        1. Imagen original
        2. Ground Truth (overlay rojo)
        3. Mapa de anomalía (heatmap)
        4. Predicción binarizada (overlay azul)
        5. Comparación GT vs Pred (verde=TP, rojo=FN, azul=FP)
        """
        # Preparar datos
        if anomaly_map.shape != gt_mask.shape:
            anomaly_map = resize_anomaly_map(anomaly_map, gt_mask.shape)

        amap_normalized = normalize_anomaly_map(anomaly_map, method="minmax")
        pred_mask = (amap_normalized >= self.threshold).astype(np.float32)

        # Crear figura
        fig = plt.figure(figsize=(25, 5))
        gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

        # Panel 1: Imagen original
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(test_image)
        ax1.set_title("Imagen Original", fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Ground Truth
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(test_image)
        gt_overlay = np.zeros((*gt_mask.shape, 4))
        gt_overlay[..., 0] = 1.0  # Red channel
        gt_overlay[..., 3] = gt_mask * 0.6  # Alpha
        ax2.imshow(gt_overlay, extent=(0, test_image.width, test_image.height, 0))
        n_gt = int(gt_mask.sum())
        ax2.set_title(
            f"Ground Truth\n({n_gt:,} píxeles)", fontsize=12, fontweight="bold"
        )
        ax2.axis("off")

        # Panel 3: Mapa de anomalía (heatmap)
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(test_image)
        im = ax3.imshow(
            amap_normalized,
            cmap="jet",
            alpha=0.6,
            vmin=0,
            vmax=1,
            extent=(0, test_image.width, test_image.height, 0),
        )
        ax3.set_title(f"Mapa de Anomalía\n[0.0 - 1.0]", fontsize=12, fontweight="bold")
        ax3.axis("off")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Panel 4: Predicción binarizada
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(test_image)
        pred_overlay = np.zeros((*pred_mask.shape, 4))
        pred_overlay[..., 2] = 1.0  # Blue channel
        pred_overlay[..., 3] = pred_mask * 0.6  # Alpha
        ax4.imshow(pred_overlay, extent=(0, test_image.width, test_image.height, 0))
        n_pred = int(pred_mask.sum())
        ax4.set_title(
            f"Predicción (τ={self.threshold})\n({n_pred:,} píxeles)",
            fontsize=12,
            fontweight="bold",
        )
        ax4.axis("off")

        # Panel 5: Comparación (TP=verde, FN=rojo, FP=azul)
        ax5 = fig.add_subplot(gs[4])
        ax5.imshow(test_image)

        gt_bool = gt_mask > 0
        pred_bool = pred_mask > 0

        comparison = np.zeros((*gt_mask.shape, 4))
        # True Positives - Verde
        tp_mask = gt_bool & pred_bool
        comparison[tp_mask, 1] = 1.0
        comparison[tp_mask, 3] = 0.7
        # False Negatives - Rojo (en GT pero no detectado)
        fn_mask = gt_bool & ~pred_bool
        comparison[fn_mask, 0] = 1.0
        comparison[fn_mask, 3] = 0.7
        # False Positives - Azul (detectado pero no en GT)
        fp_mask = ~gt_bool & pred_bool
        comparison[fp_mask, 2] = 1.0
        comparison[fp_mask, 3] = 0.7

        ax5.imshow(comparison, extent=(0, test_image.width, test_image.height, 0))

        # Calcular porcentajes
        tp = int(tp_mask.sum())
        fn = int(fn_mask.sum())
        fp = int(fp_mask.sum())

        ax5.set_title(
            f"Comparación\n🟢TP:{tp} 🔴FN:{fn} 🔵FP:{fp}",
            fontsize=11,
            fontweight="bold",
        )
        ax5.axis("off")

        # Métricas en la parte inferior
        metrics_text = (
            f"IoU: {metrics.get('IoU', 0):.3f} | "
            f"Dice: {metrics.get('Dice', 0):.3f} | "
            f"F1: {metrics.get('F1', 0):.3f} | "
            f"Precision: {metrics.get('Precision', 0):.3f} | "
            f"Recall: {metrics.get('Recall', 0):.3f} | "
            f"AU-PRO: {metrics.get('AU-PRO', 0):.3f}"
        )

        fig.text(
            0.5,
            0.02,
            metrics_text,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0.06, 1, 0.94])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"   💾 Guardado: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_anomaly_type(
        self,
        category: str,
        anomaly_type: str,
        n_samples: int = 5,
        save: bool = True,
        show: bool = False,
    ) -> List[plt.Figure]:
        """
        Visualiza múltiples imágenes de un tipo de anomalía.

        Args:
            category: Categoría del dataset
            anomaly_type: Tipo de anomalía
            n_samples: Número de imágenes a visualizar
            save: Si True, guarda las visualizaciones
            show: Si True, muestra las figuras

        Returns:
            Lista de figuras generadas
        """
        print(f"\n📸 Visualizando {category}/{anomaly_type}")

        # Cargar datos
        test_folder = os.path.join(self.dataset_path, category, "test", anomaly_type)
        gt_folder = os.path.join(
            self.dataset_path, category, "ground_truth", anomaly_type
        )

        if not os.path.isdir(gt_folder):
            print(f"   ⚠️ Sin ground truth para {anomaly_type}")
            return []

        test_data = load_test_with_ground_truth(
            test_folder=test_folder, gt_folder=gt_folder, n_images=n_samples
        )

        if not test_data:
            print(f"   ⚠️ No hay datos disponibles")
            return []

        # Obtener detector
        detector = self._get_detector(category)

        # Crear directorio de salida
        output_subdir = os.path.join(self.output_dir, category, anomaly_type)
        os.makedirs(output_subdir, exist_ok=True)

        figures = []

        for i, (test_img, gt_mask, filename) in enumerate(test_data):
            print(f"   🖼️ Procesando {filename}...")

            # Calcular mapa de anomalía
            _, amap_smooth, score = detector.compute_anomaly_map(test_img)

            # Evaluar métricas
            metrics = self.evaluator.evaluate(amap_smooth, gt_mask)

            # Título
            title = f"{category.upper()} / {anomaly_type} / {filename}"

            # Ruta de guardado
            save_path = os.path.join(
                output_subdir, f"{filename.replace('.png', '_viz.png')}"
            )

            # Visualizar
            fig = self.visualize_single_image(
                test_image=test_img,
                gt_mask=gt_mask,
                anomaly_map=amap_smooth,
                metrics=metrics,
                title=title,
                save_path=save_path if save else None,
                show=show,
            )
            figures.append(fig)

        return figures

    def visualize_category(
        self,
        category: str,
        n_samples_per_type: int = 3,
        save: bool = True,
        show: bool = False,
    ) -> Dict[str, List[plt.Figure]]:
        """
        Visualiza todos los tipos de anomalía de una categoría.

        Args:
            category: Categoría del dataset
            n_samples_per_type: Número de imágenes por tipo
            save: Si True, guarda las visualizaciones
            show: Si True, muestra las figuras

        Returns:
            Dict con figuras por tipo de anomalía
        """
        print(f"\n{'='*60}")
        print(f"📂 CATEGORÍA: {category.upper()}")
        print("=" * 60)

        anomaly_types = self._get_anomaly_types(category)
        print(f"   Tipos de anomalía: {len(anomaly_types)}")

        results = {}

        for anomaly_type in anomaly_types:
            figures = self.visualize_anomaly_type(
                category=category,
                anomaly_type=anomaly_type,
                n_samples=n_samples_per_type,
                save=save,
                show=show,
            )
            results[anomaly_type] = figures

        return results

    def visualize_all_categories(
        self,
        categories: List[str] = None,
        n_samples_per_type: int = 2,
        save: bool = True,
        show: bool = False,
    ) -> Dict[str, Dict[str, List[plt.Figure]]]:
        """
        Visualiza todas las categorías del dataset.

        Args:
            categories: Lista de categorías (None = todas)
            n_samples_per_type: Número de imágenes por tipo
            save: Si True, guarda las visualizaciones
            show: Si True, muestra las figuras

        Returns:
            Dict anidado con figuras por categoría y tipo
        """
        if categories is None:
            categories = [
                c
                for c in self.CATEGORIES
                if os.path.isdir(os.path.join(self.dataset_path, c))
            ]

        print("\n" + "=" * 80)
        print("🎨 VISUALIZACIÓN COMPLETA DEL DATASET MVTEC AD")
        print("=" * 80)
        print(f"   Categorías: {len(categories)}")
        print(f"   Muestras por tipo: {n_samples_per_type}")
        print(f"   Directorio de salida: {self.output_dir}")
        print("=" * 80)

        all_results = {}

        for i, category in enumerate(categories, 1):
            print(f"\n[{i}/{len(categories)}] ", end="")
            results = self.visualize_category(
                category=category,
                n_samples_per_type=n_samples_per_type,
                save=save,
                show=show,
            )
            all_results[category] = results

        print("\n" + "=" * 80)
        print("✅ VISUALIZACIÓN COMPLETA FINALIZADA")
        print(f"   Resultados guardados en: {self.output_dir}")
        print("=" * 80)

        return all_results

    def generate_summary_grid(
        self, category: str, n_per_type: int = 1, save: bool = True, show: bool = True
    ) -> plt.Figure:
        """
        Genera un grid resumen con una imagen por tipo de anomalía.

        Útil para tener una vista general de todos los tipos de defectos
        en una sola imagen.
        """
        anomaly_types = self._get_anomaly_types(category)
        n_types = len(anomaly_types)

        if n_types == 0:
            print(f"⚠️ No hay tipos de anomalía para {category}")
            return None

        detector = self._get_detector(category)

        # Calcular layout del grid
        n_cols = min(4, n_types)
        n_rows = (n_types + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 12, n_rows * 4))
        axes = np.atleast_2d(axes)

        for idx, anomaly_type in enumerate(anomaly_types):
            row = idx // n_cols
            col_base = (idx % n_cols) * 3

            # Cargar una imagen
            test_folder = os.path.join(
                self.dataset_path, category, "test", anomaly_type
            )
            gt_folder = os.path.join(
                self.dataset_path, category, "ground_truth", anomaly_type
            )

            test_data = load_test_with_ground_truth(test_folder, gt_folder, n_images=1)

            if not test_data:
                continue

            test_img, gt_mask, filename = test_data[0]
            _, amap_smooth, _ = detector.compute_anomaly_map(test_img)

            if amap_smooth.shape != gt_mask.shape:
                amap_smooth = resize_anomaly_map(amap_smooth, gt_mask.shape)
            amap_norm = normalize_anomaly_map(amap_smooth)
            pred_mask = amap_norm >= self.threshold

            # Panel 1: Original + GT
            ax1 = axes[row, col_base]
            ax1.imshow(test_img)
            gt_overlay = np.zeros((*gt_mask.shape, 4))
            gt_overlay[..., 0] = 1.0
            gt_overlay[..., 3] = gt_mask * 0.5
            ax1.imshow(gt_overlay, extent=(0, test_img.width, test_img.height, 0))
            ax1.set_title(f"{anomaly_type}\nGround Truth", fontsize=10)
            ax1.axis("off")

            # Panel 2: Mapa de anomalía
            ax2 = axes[row, col_base + 1]
            ax2.imshow(test_img)
            ax2.imshow(
                amap_norm,
                cmap="jet",
                alpha=0.5,
                extent=(0, test_img.width, test_img.height, 0),
            )
            ax2.set_title("Predicción", fontsize=10)
            ax2.axis("off")

            # Panel 3: Comparación
            ax3 = axes[row, col_base + 2]
            ax3.imshow(test_img)

            comparison = np.zeros((*gt_mask.shape, 4))
            gt_bool = gt_mask > 0
            pred_bool = pred_mask > 0

            # TP verde, FN rojo, FP azul
            comparison[gt_bool & pred_bool, 1] = 1.0
            comparison[gt_bool & pred_bool, 3] = 0.6
            comparison[gt_bool & ~pred_bool, 0] = 1.0
            comparison[gt_bool & ~pred_bool, 3] = 0.6
            comparison[~gt_bool & pred_bool, 2] = 1.0
            comparison[~gt_bool & pred_bool, 3] = 0.6

            ax3.imshow(comparison, extent=(0, test_img.width, test_img.height, 0))
            ax3.set_title("Comparación", fontsize=10)
            ax3.axis("off")

        # Ocultar ejes vacíos
        for idx in range(n_types, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            for c in range(3):
                axes[row, col * 3 + c].axis("off")

        fig.suptitle(
            f"Resumen de anomalías: {category.upper()}", fontsize=16, fontweight="bold"
        )

        # Leyenda
        fig.text(
            0.5,
            0.02,
            "🟢 True Positive | 🔴 False Negative | 🔵 False Positive",
            ha="center",
            fontsize=12,
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_summary_grid.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"💾 Grid resumen guardado: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualización de Anomalías MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python visualize_anomalies.py --category bottle --n-samples 3
  python visualize_anomalies.py --category cable --anomaly-type bent_wire --n-samples 5
  python visualize_anomalies.py --all-categories --n-samples 2
  python visualize_anomalies.py --category bottle --summary-grid
        """,
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
        "--output-dir",
        type=str,
        default="/home/bllancao/Portafolio/mvtec_anomaly_detection/visualizations",
        help="Directorio para guardar visualizaciones",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Categoría a visualizar"
    )
    parser.add_argument(
        "--anomaly-type",
        type=str,
        default=None,
        help="Tipo de anomalía específico a visualizar",
    )
    parser.add_argument(
        "--all-categories", action="store_true", help="Visualizar todas las categorías"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Número de muestras por tipo de anomalía",
    )
    parser.add_argument(
        "--n-good-images",
        type=int,
        default=None,
        help='Número de imágenes "good" para memory bank',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Umbral para binarización (0-1)"
    )
    parser.add_argument(
        "--summary-grid",
        action="store_true",
        help="Generar grid resumen de la categoría",
    )
    parser.add_argument(
        "--show", action="store_true", help="Mostrar visualizaciones en pantalla"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="No guardar visualizaciones"
    )

    args = parser.parse_args()

    # Crear visualizador
    visualizer = AnomalyVisualizer(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        n_good_images=args.n_good_images,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    save = not args.no_save

    # Modo: todas las categorías
    if args.all_categories:
        visualizer.visualize_all_categories(
            n_samples_per_type=args.n_samples, save=save, show=args.show
        )

    # Modo: grid resumen
    elif args.summary_grid and args.category:
        visualizer.generate_summary_grid(
            category=args.category, save=save, show=args.show
        )

    # Modo: tipo de anomalía específico
    elif args.category and args.anomaly_type:
        visualizer.visualize_anomaly_type(
            category=args.category,
            anomaly_type=args.anomaly_type,
            n_samples=args.n_samples,
            save=save,
            show=args.show,
        )

    # Modo: categoría completa
    elif args.category:
        visualizer.visualize_category(
            category=args.category,
            n_samples_per_type=args.n_samples,
            save=save,
            show=args.show,
        )

    else:
        print("⚠️ Especifica --category, --all-categories, o --summary-grid")
        parser.print_help()


if __name__ == "__main__":
    main()

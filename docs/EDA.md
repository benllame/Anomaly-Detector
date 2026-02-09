# EDA Report: screw

## Dataset Overview

### Class Distribution

**Train Set:**
{'good': 320}

**Test Set:**
{'scratch_head': 24, 'thread_top': 23, 'good': 41, 'thread_side': 23, 'manipulated_front': 24, 'scratch_neck': 25}

### Image Statistics

- **Total images analyzed:** 480
- **Height:** 1024 ± 0 pixels
- **Width:** 1024 ± 0 pixels
- **Aspect Ratio:** 1.00

### Pixel Statistics (Good Samples)

- **Red:** 184.64 ± 34.26
- **Green:** 184.64 ± 34.26
- **Blue:** 184.64 ± 34.26

## Key Findings

1. **Image Dimensions:** Las imágenes tienen tamaño consistente
2. **Color Distribution:** Canal predominante [analizar según stats]
3. **Class Imbalance:** [Calcular ratio good/defect]

## Recommendations

1. **Preprocessing:**
   - Resize a 640x640 para YOLO
   - Normalización RGB estándar

2. **Augmentation:**
   - Rotations, flips para aumentar variabilidad
   - Color jittering para robustez

3. **Training Strategy:**
   - Monitorear métricas por clase (precision/recall)
   - Considerar class weights si hay desbalance

## Generated Visualizations

- `class_distribution.png`
- `samples_visualization.png`
- `dimension_analysis.png`
- `pixel_statistics.png`
- `good_vs_defect_comparison.png`

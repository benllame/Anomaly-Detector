# 📊 Evaluation Report - MVTec AD Anomaly Detection

> **DINOv2-based anomaly detection system evaluation on the complete MVTec AD dataset**

---

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | DINOv2 ViT-B/14 |
| **Layer** | -1 (last) |
| **k-NN (k)** | 1 |
| **Threshold** | 0.6 |
| **Memory Bank Size** | 200 images/class |

---

## � Global Metrics

<table>
<tr>
<td>

### Summary
- 📷 **1,258** images evaluated
- 🏷️ **15** categories
- ⏱️ ~50ms/image (CPU)

</td>
<td>

| Metric | Value | Std Dev |
|:------:|:-----:|:-------:|
| **IoU** | 0.277 | ± 0.196 |
| **Dice** | 0.398 | ± 0.232 |
| **Precision** | 0.322 | ± 0.248 |
| **Recall** | 0.795 | ± 0.250 |
| **AU-PRO** | 0.831 | ± 0.172 |

</td>
</tr>
</table>

---

## � Top Performing Categories

| Rank | Category | IoU | Dice | AU-PRO | Performance |
|:----:|:---------|:---:|:----:|:------:|:-----------:|
| 🥇 | **Bottle** | 0.564 | 0.710 | 0.891 | ████████░░ |
| 🥈 | **Tile** | 0.460 | 0.593 | 0.850 | ███████░░░ |
| 🥉 | **Metal Nut** | 0.383 | 0.530 | 0.799 | ██████░░░░ |

---

## 📈 Complete Results by Category

<details>
<summary><b>🍾 Bottle</b> — IoU: 0.564 | AU-PRO: 0.891</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| broken_large | 20 | 0.601 | 0.749 | 0.879 |
| broken_small | 22 | 0.531 | 0.690 | 0.871 |
| contamination | 21 | 0.563 | 0.694 | 0.925 |

</details>

<details>
<summary><b>🔌 Cable</b> — IoU: 0.304 | AU-PRO: 0.750</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| bent_wire | 13 | 0.363 | 0.524 | 0.883 |
| cable_swap | 12 | 0.094 | 0.169 | 0.284 |
| combined | 11 | 0.270 | 0.420 | 0.624 |
| cut_inner_insulation | 14 | 0.351 | 0.510 | 0.851 |
| cut_outer_insulation | 10 | 0.185 | 0.305 | 0.762 |
| missing_cable | 12 | 0.542 | 0.701 | 0.853 |
| missing_wire | 10 | 0.258 | 0.393 | 0.916 |
| poke_insulation | 10 | 0.331 | 0.471 | 0.829 |

</details>

<details>
<summary><b>💊 Capsule</b> — IoU: 0.201 | AU-PRO: 0.877</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| crack | 23 | 0.164 | 0.262 | 0.882 |
| faulty_imprint | 22 | 0.198 | 0.312 | 0.895 |
| poke | 21 | 0.134 | 0.225 | 0.902 |
| scratch | 23 | 0.231 | 0.356 | 0.874 |
| squeeze | 20 | 0.282 | 0.421 | 0.829 |

</details>

<details>
<summary><b>🧶 Carpet</b> — IoU: 0.270 | AU-PRO: 0.891</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| color | 19 | 0.292 | 0.442 | 0.897 |
| cut | 17 | 0.499 | 0.660 | 0.915 |
| hole | 17 | 0.331 | 0.491 | 0.935 |
| metal_contamination | 17 | 0.111 | 0.198 | 0.892 |
| thread | 19 | 0.133 | 0.227 | 0.824 |

</details>

<details>
<summary><b>🔲 Grid</b> — IoU: 0.138 | AU-PRO: 0.851</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| bent | 12 | 0.149 | 0.254 | 0.856 |
| broken | 12 | 0.093 | 0.165 | 0.831 |
| glue | 11 | 0.167 | 0.280 | 0.858 |
| metal_contamination | 11 | 0.106 | 0.191 | 0.875 |
| thread | 11 | 0.175 | 0.294 | 0.839 |

</details>

<details>
<summary><b>🌰 Hazelnut</b> — IoU: 0.336 | AU-PRO: 0.900</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| crack | 18 | 0.437 | 0.576 | 0.900 |
| cut | 17 | 0.168 | 0.285 | 0.916 |
| hole | 18 | 0.294 | 0.444 | 0.885 |
| print | 17 | 0.442 | 0.606 | 0.900 |

</details>

<details>
<summary><b>👜 Leather</b> — IoU: 0.110 | AU-PRO: 0.933</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| color | 19 | 0.087 | 0.157 | 0.918 |
| cut | 19 | 0.064 | 0.118 | 0.948 |
| fold | 17 | 0.243 | 0.380 | 0.940 |
| glue | 19 | 0.129 | 0.224 | 0.914 |
| poke | 18 | 0.038 | 0.073 | 0.946 |

</details>

<details>
<summary><b>🔩 Metal Nut</b> — IoU: 0.383 | AU-PRO: 0.799</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| bent | 25 | 0.206 | 0.332 | 0.823 |
| color | 22 | 0.368 | 0.521 | 0.836 |
| flip | 23 | 0.549 | 0.705 | 0.725 |
| scratch | 23 | 0.422 | 0.581 | 0.810 |

</details>

<details>
<summary><b>💊 Pill</b> — IoU: 0.293 | AU-PRO: 0.877</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| color | 25 | 0.104 | 0.180 | 0.892 |
| combined | 17 | 0.286 | 0.430 | 0.797 |
| contamination | 21 | 0.351 | 0.502 | 0.872 |
| crack | 26 | 0.230 | 0.360 | 0.913 |
| faulty_imprint | 19 | 0.407 | 0.556 | 0.890 |
| pill_type | 9 | 0.369 | 0.526 | 0.819 |
| scratch | 24 | 0.394 | 0.544 | 0.897 |

</details>

<details>
<summary><b>🔧 Screw</b> — IoU: 0.091 | AU-PRO: 0.676</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| manipulated_front | 24 | 0.056 | 0.101 | 0.637 |
| scratch_head | 24 | 0.043 | 0.079 | 0.790 |
| scratch_neck | 25 | 0.175 | 0.289 | 0.922 |
| thread_side | 23 | 0.023 | 0.043 | 0.342 |
| thread_top | 23 | 0.156 | 0.242 | 0.663 |

</details>

<details>
<summary><b>🧱 Tile</b> — IoU: 0.460 | AU-PRO: 0.850</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| crack | 17 | 0.114 | 0.203 | 0.839 |
| glue_strip | 18 | 0.439 | 0.603 | 0.788 |
| gray_stroke | 16 | 0.424 | 0.587 | 0.909 |
| oil | 18 | 0.584 | 0.734 | 0.834 |
| rough | 15 | 0.763 | 0.861 | 0.894 |

</details>

<details>
<summary><b>🪥 Toothbrush</b> — IoU: 0.215 | AU-PRO: 0.867</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| defective | 30 | 0.215 | 0.333 | 0.867 |

</details>

<details>
<summary><b>📻 Transistor</b> — IoU: 0.287 | AU-PRO: 0.691</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| bent_lead | 10 | 0.126 | 0.219 | 0.832 |
| cut_lead | 10 | 0.252 | 0.392 | 0.796 |
| damaged_case | 10 | 0.545 | 0.693 | 0.864 |
| misplaced | 10 | 0.224 | 0.338 | 0.274 |

</details>

<details>
<summary><b>🪵 Wood</b> — IoU: 0.322 | AU-PRO: 0.802</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| color | 8 | 0.368 | 0.515 | 0.876 |
| combined | 11 | 0.343 | 0.470 | 0.751 |
| hole | 10 | 0.129 | 0.217 | 0.833 |
| liquid | 10 | 0.480 | 0.643 | 0.889 |
| scratch | 21 | 0.309 | 0.444 | 0.744 |

</details>

<details>
<summary><b>🧥 Zipper</b> — IoU: 0.280 | AU-PRO: 0.809</summary>

| Defect Type | Images | IoU | Dice | AU-PRO |
|:------------|:------:|:---:|:----:|:------:|
| broken_teeth | 19 | 0.301 | 0.444 | 0.756 |
| combined | 16 | 0.238 | 0.371 | 0.767 |
| fabric_border | 17 | 0.237 | 0.379 | 0.846 |
| fabric_interior | 16 | 0.286 | 0.435 | 0.819 |
| rough | 17 | 0.314 | 0.458 | 0.735 |
| split_teeth | 18 | 0.291 | 0.433 | 0.861 |
| squeezed_teeth | 16 | 0.292 | 0.438 | 0.886 |

</details>

---

## 📝 Metrics Explanation

| Metric | Description |
|--------|-------------|
| **IoU** | Intersection over Union - overlap between prediction and ground truth |
| **Dice** | Dice coefficient (F1 for segmentation) |
| **Precision** | Ratio of correctly detected anomaly pixels |
| **Recall** | Ratio of ground truth anomalies detected |
| **AU-PRO** | Area Under Per-Region Overlap curve (MVTec standard) |

---

<div align="center">

**[← Back to README](../README.md)**

</div>

# 🦉 Project Night Owl: Low-Light Object Detection via Generative Domain Adaptation

Standard object detection models cant detect greatly in low-light environments. Project Night Owl proposes a two-stage computer vision pipeline bridging Generative AI and Discriminative AI. 

By utilizing an unsupervised CycleGAN to translate low-illumination images into daylight images, we successfully recover hidden physical geometries without paired data, enabling standard YOLOv8 models to detect objects in pitch-black footage.

## 📊 Datasets

This project relies on two distinct datasets for the generative and discriminative pipelines:

1. **[Exclusively Dark (ExDark) Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)**
   * **Use:** Object Detection (YOLOv8)
   * **Details:** 7,363 low-light images across 12 object classes with bounding box annotations.
2. **[Low-Light (LOL) Dataset](https://huggingface.co/datasets/keras-io/low-light-image-enhancement)**
   * **Use:** Domain Translation (CycleGAN)
   * **Details:** Unpaired high/low illumination images used to train the daylight Generator.

## 🗂 Repository Structure

* `configs/` - YOLOv8 YAML configurations for baseline and enhanced datasets.
* `weights/` - Pre-trained model weights (CycleGAN generators and YOLO `.pt` files).
* `01_data_preprocessing/` - Scripts to convert ExDark absolute coordinates to normalized YOLO format.
* `02_gan_pipeline/` - PyTorch implementation of the CycleGAN and the dataset enhancement script.
* `03_yolo_pipeline/` - Baseline training, zero-shot testing, and enhanced dataset retraining notebooks.

## 🚀 Quickstart & Installation

Clone the repository and install the dependencies. I trained pipeline on google colab and local machine , so change directories whereever needed

```bash
git clone [https://github.com/YOUR_USERNAME/Night-Owl-Project.git](https://github.com/YOUR_USERNAME/Night-Owl-Project.git)
cd Night-Owl-Project
pip install torch torchvision ultralytics pillow matplotlib



⚙️ Execution Pipeline
To reproduce the results, execute the notebooks in the following sequential order:

Phase 1: Preparation & Baseline
Run 01_data_preprocessing/convert_exdark.ipynb to format the dataset.

Run 03_yolo_pipeline/yolo_training.ipynb to establish the dark-image baseline metrics.

Phase 2: Generative Enhancement
Run 02_gan_pipeline/GAN.ipynb to train the CycleGAN on the LOL dataset (Recommended: T4 GPU or higher).

Run 02_gan_pipeline/enhance_dataset.ipynb to process the entire ExDark dataset through the trained daylight Generator.

Phase 3: Final Evaluation
Run 03_yolo_pipeline/test_zeroshot.ipynb for qualitative visual comparisons.

Run 03_yolo_pipeline/yolo_training_after_enhancement.ipynb to train the final model and output the enhanced mAP scores.
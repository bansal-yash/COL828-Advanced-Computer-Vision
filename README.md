# COL828-Advanced-Computer-Vision
Course assignments of COL828:- Advanced Computer Vision course at IIT Delhi under Professor Chetan Arora

This repository contains assignments that explore state-of-the-art vision-language models and prompt learning techniques for medical image classification and detection tasks.

---

## 📁 Assignments

### [Assignment 1: Implant Classification](./Assignment_1_Implant_Classification)
- Applied vision-language models (CLIP, ViT) for medical implant classification on **Orthonet** and **Pacemakers** datasets.
- Implemented and compared multiple approaches:
  - Zero-shot CLIP inference without training
  - Fine-tuning with ImageNet21K, CLIP, and DINOv2 backbones
  - Prompt learning methods: **CoOp** (Context Optimization), **CoCoOp** (Conditional CoOp), and **MaPLe** (Multi-modal Prompt Learning)
- Evaluated models using accuracy, F1-score, precision, and recall metrics.
- Demonstrated the effectiveness of prompt learning over traditional fine-tuning for few-shot medical image classification.

---

### [Assignment 2: Cancer Detection in Mammography](./Assignment_2_Cancer_Detection)
- Performed object detection on mammography images to identify benign and malignant tumors using **Grounding DINO**.
- Explored zero-shot detection capabilities of pre-trained Grounding DINO without additional training.
- Implemented prompt learning techniques (CoOp, CoCoOp) to improve detection by learning optimal text prompts.
- Applied **FixMatch** semi-supervised learning to leverage unlabeled data for enhanced model performance.
- Evaluated detection quality using **mAP** (mean Average Precision) and conducted qualitative analysis with bounding box visualizations.

---

Each assignment demonstrates the application of cutting-edge vision-language models and prompt engineering techniques to challenging medical imaging tasks, combining transfer learning, few-shot learning, and modern detection frameworks.

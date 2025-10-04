# COL828: Advanced Computer Vision – Assignment 1

**Implant Classification using Transformer Models**

## 📋 Requirements

* Python **3.9** or above
* Install all dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Dataset Setup

* Ensure the original **Orthonet** and **Pacemakers** datasets are placed in the respective folders:
  * `orthonet_data/`
  * `pacemakers_data/`
* Maintain the **original folder structure** of the datasets.

## 📁 Model Saving & Naming Convention

* All trained models are saved inside `--save_dir` provided during training.
* Naming convention for `.pth` files is strict and must be followed:
  * **Orthonet fine-tune** → `fine_tune_imagenet21k.pth`, `fine_tune_clip.pth`, `fine_tune_dinov2.pth`
  * **Orthonet CoOp/CoCoOp/MaPLe** → `coop.pth`, `cocoop.pth`, `maple.pth`
  * **Pacemakers fine-tune** → `fine_tune_imagenet21k.pth`, `fine_tune_clip.pth`, `fine_tune_dinov2.pth`
  * **Pacemakers CoOp/CoCoOp/MaPLe** → `*_stage_1.pth`, `*_stage_2.pth`

## 🚀 Running Experiments

### 1. Orthonet Dataset

#### Fine-tuning

**Train**:
```bash
python3 orthonet_fine_tune_train.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

**Test**:
```bash
python3 orthonet_fine_tune_test.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

#### Zero-shot CLIP

```bash
python3 orthonet_zero_shot.py --data_dir orthonet_data
```

#### CoOp

**Train**:
```bash
python3 orthonet_coop_train.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

**Test**:
```bash
python3 orthonet_coop_test.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

#### CoCoOp

**Train**:
```bash
python3 orthonet_cocoop_train.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

**Test**:
```bash
python3 orthonet_cocoop_test.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

#### MaPLe

**Train**:
```bash
python3 orthonet_maple_train.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

**Test**:
```bash
python3 orthonet_maple_test.py --data_dir orthonet_data --save_dir Trained_Models/orthonet
```

### 2. Pacemakers Dataset

#### Fine-tuning

**Train**:
```bash
python3 pacemakers_fine_tune_train.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

**Test**:
```bash
python3 pacemakers_fine_tune_test.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

#### Zero-shot CLIP

```bash
python3 pacemakers_zero_shot.py --data_dir pacemakers_data
```

#### CoOp

**Train**:
```bash
python3 pacemakers_coop_train.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

This will save:
* `coop_stage_1.pth`
* `coop_stage_2.pth`

**Test**:
```bash
python3 pacemakers_coop_test.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

#### CoCoOp

**Train**:
```bash
python3 pacemakers_cocoop_train.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

This will save:
* `cocoop_stage_1.pth`
* `cocoop_stage_2.pth`

**Test**:
```bash
python3 pacemakers_cocoop_test.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

#### MaPLe

**Train**:
```bash
python3 pacemakers_maple_train.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

This will save:
* `maple_stage_1.pth`
* `maple_stage_2.pth`
* `maple.pth`

**Test**:
```bash
python3 pacemakers_maple_test.py --data_dir pacemakers_data --save_dir Trained_Models/pacemakers
```

## ✅ Notes

* Always provide correct `--data_dir` and `--save_dir` paths.
* Ensure `.pth` file naming conventions are strictly followed, otherwise test scripts may not detect the models.
* Results (accuracy, F1-score, etc.) will be displayed during the test runs.
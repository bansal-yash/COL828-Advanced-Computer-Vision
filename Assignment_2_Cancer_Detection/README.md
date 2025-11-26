# Mammography Object Detection

## Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Replace Transformers Library File

After installation, replace the `modeling_grounding_dino.py` file in the transformers library with the provided custom file:

```bash
# Find transformers installation path
python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))"

# Copy the custom file (replace <TRANSFORMERS_PATH> with the path from above)
cp modeling_grounding_dino.py <TRANSFORMERS_PATH>/models/grounding_dino/modeling_grounding_dino.py
```

## Training

Training is performed using Jupyter notebooks located in `Train_Notebooks/`:

- **CoOp Training**: `coop.ipynb`
- **CoCoOp Training**: `cocoop.ipynb`
- **FixMatch Training**: `fixmatch.ipynb`

Open and run the respective notebook for your desired training method.

## Testing

### CoOp Testing

```bash
python test_coop.py --dataset_root /path/to/dataset --model_path /path/to/models --n_ctx 4
```

**Arguments**:
- `--dataset_root`: Path to dataset directory (default: `/mammography`)
- `--model_path`: Path to trained model files (default: `./trained_models`)
- `--n_ctx`: Number of learnable context tokens (default: `4`)

### CoCoOp Testing

```bash
python test_cocoop.py --dataset_root /path/to/dataset --model_path /path/to/models --n_ctx 4 --feature_level highest
```

**Arguments**:
- `--dataset_root`: Path to dataset directory (default: `/mammography`)
- `--model_path`: Path to trained model files (required)
- `--n_ctx`: Number of learnable context tokens (default: `4`)
- `--feature_level`: Feature level - `highest`, `lowest`, or `middle` (required)

### Zero-shot Testing

```bash
python zero_shot.py --dataset_root /path/to/dataset
```

**Arguments**:
- `--dataset_root`: Path to dataset directory (default: `/mammography`)
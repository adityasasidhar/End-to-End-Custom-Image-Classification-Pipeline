# Custom Image Classification Pipeline

This project is a custom image classification pipeline built with PyTorch, specifically designed for the CIFAR-10 dataset. It uses a modified ResNet-18 architecture optimized for 32x32 images and features a training strategy with progressive layer unfreezing.

As stated in the project description: *"This is a fun learning attempt at building custom models, I was bored and the year was new, so i thought why not ?? hehehehe"*

## 🚀 Features

- **Custom ResNet-18**: Modified the initial convolution and removed the maxpooling layer to better handle the 32x32 resolution of CIFAR-10.
- **CIFAR-10 Dataset**: Integrated data loading with standard augmentations (Random Crop, Horizontal Flip).
- **Progressive Unfreezing**: Training starts with a frozen backbone, then gradually unfreezes later layers (`layer4`, then `layer3`) to fine-tune performance.
- **Inference Script**: Easy-to-use script for predicting the class of any local image.
- **Config-Driven**: Manage hyperparameters easily via YAML files.

## 📁 Project Structure

```text
.
├── configs/
│   └── default.yaml    # Main configuration file
├── data/               # CIFAR-10 dataset (auto-downloaded)
│   ├── cifar-10-batches-py/
│   └── cifar-10-python.tar.gz
├── src/
│   ├── dataset.py      # Data loading and augmentation
│   ├── eval.py         # Performance evaluation
│   ├── infer.py        # Single image prediction
│   ├── model.py        # ResNet-18 modifications
│   ├── training.py     # Training & progressive unfreezing
│   ├── utils.py        # Shared utilities
│   ├── resnet_architecture.py # Architecture inspector
│   ├── image.jpg       # Sample image for inference
│   └── model.pth       # Saved model weights
├── LICENSE             # MIT License
├── pyproject.toml      # Project metadata & dependencies
├── README.md           # You are here!
└── uv.lock             # Dependency lockfile
```

## ⚙️ Configuration

The project uses YAML files for configuration. You can modify parameters in `configs/default.yaml`:
- **Data**: Batch size, number of workers, data path.
- **Model**: Backbone selection, pre-training, freezing options.
- **Training**: Epochs, learning rates, unfreezing schedule.
- **Runtime**: Device selection (CPU/CUDA) and random seed.

## 🛠️ Installation

This project uses `uv` for dependency management.

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CustomImageclassificationPipeline
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## 📈 Usage

### Training

To start training the model on CIFAR-10:

```bash
uv run python src/training.py
```

The script will:
- Train according to parameters in `configs/default.yaml`.
- Download the CIFAR-10 dataset to the `data/` folder if not present.
- Save the best model weights to `model.pth`.

### Evaluation

To evaluate the trained model on the test set:

```bash
uv run python src/eval.py
```

### Inference

To predict the class of an image:

```bash
uv run python src/infer.py path/to/your/image.jpg
```

Example output:
```text
Prediction: dog
Confidence: 0.8942
```

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

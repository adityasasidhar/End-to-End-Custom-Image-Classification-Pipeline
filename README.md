# Custom Image Classification Pipeline

This project is a custom image classification pipeline built with PyTorch, specifically designed for the CIFAR-10 dataset. It uses a modified ResNet-18 architecture optimized for 32x32 images and features a training strategy with progressive layer unfreezing.

As stated in the project description: *"This is a fun learning attempt at building custom models, I was bored and the year was new, so i thought why not ?? hehehehe"*

## 🚀 Features

- **Custom ResNet-18**: Modified the initial convolution and removed the maxpooling layer to better handle the 32x32 resolution of CIFAR-10.
- **CIFAR-10 Dataset**: Integrated data loading with standard augmentations (Random Crop, Horizontal Flip).
- **Progressive Unfreezing**: Training starts with a frozen backbone, then gradually unfreezes later layers (`layer4`, then `layer3`) to fine-tune performance.
- **Inference Script**: Easy-to-use script for predicting the class of any local image.

## 📁 Project Structure

```text
.
├── configs/            # Configuration files (if any)
├── data/               # Dataset storage
├── src/
│   ├── dataset.py      # CIFAR-10 DataLoaders and Transforms
│   ├── infer.py        # Inference script for single images
│   ├── model.py        # Custom Model definition
│   ├── training.py     # Main training loop with unfreezing logic
│   └── resnet_architecture.py # Utility to inspect base ResNet
├── pyproject.toml      # Project dependencies and metadata
└── uv.lock             # Lockfile for dependencies
```

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
- Download the CIFAR-10 dataset to the `data/` folder if not present.
- Train for 15 epochs.
- Progressively unfreeze layers at epoch 5 and 11.
- Save the best model weights to `model.pth`.

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

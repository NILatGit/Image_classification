# Image_classification
# Image Classification without CNN

Welcome to the Image Classification without CNN project! This repository demonstrates how to classify images using non-convolutional architectures—leveraging techniques like patch-based embedding, MLP mixers, and transformer-inspired layers. Perfect for exploring alternative approaches beyond standard CNNs.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Model Architectures](#model-architectures)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Usage](#usage)
10. [Results](#results)
11. [Customization](#customization)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [License](#license)
15. [Contact](#contact)

---

## Project Overview

This notebook implements image classification without convolutional neural networks. Instead, it explores:

* **Patch Embeddings:** Splitting images into fixed-size patches and linearly projecting them.
* **MLP Mixer:** Alternating token-mixing and channel-mixing MLP layers.
* **Transformer Blocks (optional):** Self-attention layers for capturing global context.

By comparing these methods, you’ll learn how non-CNN architectures handle spatial information and classification tasks.

---

## Features

* **Patch-based encoding:** Convert image to sequence of patch tokens.
* **MLP Mixer implementation:** Simple yet powerful fully-connected mixer layers.
* **Transformer Block option:** Integrate multi-head self-attention.
* **Flexible depth/width:** Easily adjust number of layers and hidden dimensions.
* **Training with PyTorch:** End-to-end examples using PyTorch.

---

## Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision
* numpy
* matplotlib
* tqdm

> **Tip:** Use a virtual environment or Conda environment to isolate dependencies.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/image-classification-no-cnn.git
   cd image-classification-no-cnn
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

This notebook uses CIFAR-10 by default. To prepare your own dataset:

1. Organize images into `data/train/<class_name>/` and `data/test/<class_name>/` directories.
2. Ensure images are of consistent size (e.g., 32×32 or 64×64). Notebook includes resizing transforms.
3. Update the `DATA_DIR` path in the first code cell.

---

## Model Architectures

1. **Patch Embedding Layer**

   * Splits image into non-overlapping patches.
   * Flattens and linearly projects each patch to an embedding vector.

2. **MLP Mixer**

   * **Token-mixing MLP:** Learns spatial correlations across patches.
   * **Channel-mixing MLP:** Learns relationships across feature channels.

3. **Transformer Block (Optional)**

   * Multi-head self-attention for capturing global dependencies.
   * Feed-forward network with residual connections.

Hyperparameters you can adjust:

* `patch_size`, `embed_dim`, `num_layers`, `mlp_dim`, `num_heads` (for transformer).

---

## Training

1. Configure hyperparameters in the notebook’s parameter cell.
2. Run cells sequentially:

   * Data loading and augmentation
   * Model definitions
   * Training loop with loss and accuracy logging
   * Checkpoint saving

**Training Tips:**

* Use a small model first (`embed_dim=64`, `num_layers=2`).
* Gradually increase complexity if GPU memory allows.
* Monitor both training and validation accuracy to avoid overfitting.

---

## Evaluation

* After training, evaluate on the test set:

  ```python
  test_acc = evaluate(model, test_loader)
  print(f"Test Accuracy: {test_acc:.2f}%")
  ```

* Visualize confusion matrix and misclassified examples using provided utility functions.

---

## Usage

To run the notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute model_image_classification_withoutCNN.ipynb
```

Or open it in Jupyter Lab/Colab for interactive exploration.

---

## Results

An example test accuracy on CIFAR-10 after 20 epochs with MLP Mixer (embed\_dim=128, num\_layers=4):

> **Test Accuracy:** 70.3%

Sample misclassified images and confusion matrix plots are displayed in the notebook.

---

## Customization

* **Change Dataset:** Swap CIFAR-10 for CIFAR-100 or your own image folder.
* **Alternate Patch Sizes:** Experiment with larger patches (e.g., 16×16).
* **Swap Architectures:** Replace MLP Mixer with pure Transformer or other sequence models.

---

## Future Work

* Explore hybrid CNN-Mixer architectures.
* Implement hierarchical patching for multi-scale features.
* Compare performance on larger datasets like ImageNet.

---

## Contributing

Contributions and improvements are welcome:

1. Fork the repository
2. Create a branch for your feature
3. Commit changes and push
4. Submit a pull request with detailed description

Please follow existing code style and include tests if possible.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

Developed by Nil (Studying Computer Engineering)

* GitHub: [NILatGit](https://github.com/NILatGit)
* Email: [nskarmakar.cse.ug@jadavpurunniversity.in](mailto:nskarmakar.cse.ug@jadavpurunniversity.in)

Happy experimenting with non-convolutional image classification!


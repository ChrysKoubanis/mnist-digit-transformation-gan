# MNIST Digit Transformation: Autoencoder vs. Conditional GAN

An end-to-end deep learning project exploring **Image-to-Image Translation** on the MNIST dataset. The objective is to transform an input digit N into its sequential successor (N+1) mod 10 (e.g., translating a drawn 1 into a 2, or a 9 into a 0).

This project evaluates and compares three distinct modeling paradigms:
1. **Classical Baseline:** Principal Component Analysis (PCA) dimensional reduction.
2. **Deep Learning Baseline:** Non-adversarial Convolutional Autoencoder (MSE loss).
3. **Adversarial Approach:** Conditional GAN (cGAN / Pix2Pix architecture with combined Binary Cross-Entropy and L1 Loss).

---

## 📸 Key Results & Comparison

| Approach | Visual Sharpness | Blur Artifacts | Target Accuracy ("Judge" Classifier) |
| :--- | :--- | :--- | :--- |
| **PCA (32 Components)** | Very Low | Severe | N/A (Reconstruction only) |
| **Convolutional Autoencoder** | Moderate | High (Averaged pixel loss) | ~72.4% |
| **Conditional GAN (Run 6)** | High | Minimal (Sharp edges) | **>94.0%** |

---

## 📁 Repository Structure

```text
neural-networks-assignment-3/
├── before GAN/                 # Baseline Autoencoder pipeline
│   ├── data_loader.py          # Preprocesses MNIST digit pairs
│   ├── model.py                # Encoder-Decoder CNN architecture
│   └── train.py                # Standard MSE training loop
│
├── after GAN/                  # Conditional GAN pipeline
│   ├── data_loader.py          # Pair dataset generator
│   ├── model.py                # Generator network (U-Net style bottleneck)
│   ├── discriminator.py        # PatchGAN / Conv Discriminator
│   ├── train_gan.py            # Adversarial training loop with custom tf.function
│   ├── live_demo.py            # Interactive digit transformation demo
│   └── compare_models.py       # Multi-run GAN loss evaluator
│
├── comparison/                 # Comprehensive validation suite
│   ├── compare_models.py       # Loss trend & side-by-side visual generator
│   ├── final_validation.py     # CNN 'Judge' classifier & PCA comparison
│   ├── gan_generator_6.keras   # Best performing GAN model weights
│   └── *.txt                   # Training logs
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/ChrysKoubanis/mnist-digit-transformation-gan.git]


```

Create and activate a virtual environment:

```bash
python -m venv venv

```



* **Windows :**
```
.\venv\Scripts\Activate

```



Install dependencies:

```bash
pip install -r requirements.txt

```

---

### 2. Running Evaluation & Visualizations

To reproduce the benchmark comparisons using pre-trained model weights without retraining:

* **Run side-by-side visual and mathematical comparisons:**
```bash
python comparison/compare_models.py

```


* **Run automated CNN "Judge" validation & PCA baseline:**
```bash
python comparison/final_validation.py

```



---

### 3. Training Models

* **To train the baseline autoencoder from scratch:**
```bash
python "before GAN/train.py"

```


* **To train the conditional GAN from scratch:**
```bash
python "after GAN/train_gan.py"

```



---

## 🧪 Methodological Overview

* **Generator Loss:** Formulated as $L_G = L_{\text{GAN}} + \lambda L_{L1}$, where $\lambda = 10$. The $L1$ pixel loss preserves exact digit structural placement, while the GAN adversarial loss enforces high-frequency sharp details.
* **Discriminator:** Features a Convolutional Patch-based architecture with label smoothing (`0.9` for real samples) to prevent discriminator overconfidence.
* **Evaluation Metric:** An independent CNN classifier ("The Judge") trained on standard MNIST tests whether generated digits are correctly categorized as target numbers.
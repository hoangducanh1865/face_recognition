# Face Recognition - Siamese Network

A comprehensive face verification system using Siamese Neural Networks built with TensorFlow/Keras.

## 📁 Project Structure

```
FaceRecognition/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and environment settings
│   ├── layers.py                # Custom Keras layers (L1Dist)
│   ├── model.py                 # Siamese Network model definition
│   ├── data.py                  # Data loading, preprocessing, augmentation
│   ├── trainer.py               # Training logic and checkpoints
│   ├── verifier.py              # Face verification functionality
│   ├── utils.py                 # Utility functions
│   └── main.py                  # Main entry point
├── data/                        # Training data
│   ├── anchor/                  # Anchor face images
│   ├── positive/                # Positive (same person) images
│   └── negative/                # Negative (different person) images
├── application_data/            # Verification data
│   ├── input_image/             # Current input for verification
│   └── verification_images/     # Reference images for verification
├── checkpoints_backup/          # Training checkpoints
├── siamesemodelv2.keras         # Trained model file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- TensorFlow 2.16+
- OpenCV
- Webcam (for data collection and verification)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd face_recognition/FaceRecognition
   ```

2. **Create a virtual environment:**

   ```bash
   conda create -n face_recognition python=3.10
   conda activate face_recognition
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Application

```bash
cd FaceRecognition
python -m src.main [OPTIONS]
```

### Available Modes

#### 1. Show Model Info (Default)

```bash
python -m src.main --mode info
```

#### 2. Train the Model

**Local training:**

```bash
python -m src.main --mode train --environment local --epochs 50
```

**Kaggle training:**

```bash
python -m src.main --mode train --environment kaggle --epochs 50
```

**Resume from checkpoint:**

```bash
python -m src.main --mode train --use-checkpoint --epochs 20
```

**Train with data augmentation:**

```bash
python -m src.main --mode train --augment --epochs 50
```

#### 3. Collect Training Images

```bash
python -m src.main --mode collect --camera 0
```

Controls:

- `a` - Save as anchor image
- `p` - Save as positive image
- `q` - Quit

#### 4. Run Face Verification

```bash
python -m src.main --mode verify --camera 0
```

Controls:

- `v` - Capture and verify face
- `q` - Quit

**With custom thresholds:**

```bash
python -m src.main --mode verify --detection-threshold 0.7 --verification-threshold 0.6
```

#### 5. Evaluate Model

```bash
python -m src.main --mode evaluate
```

### Command Line Options

| Option                     | Description                                            | Default |
| -------------------------- | ------------------------------------------------------ | ------- |
| `--mode`                   | Operation mode: train, verify, collect, evaluate, info | info    |
| `--environment`            | Runtime environment: local, kaggle                     | local   |
| `--use-checkpoint`         | Load from checkpoint instead of model file             | False   |
| `--epochs`                 | Number of training epochs                              | 50      |
| `--camera`                 | Camera index for webcam                                | 0       |
| `--detection-threshold`    | Threshold for positive detection                       | 0.5     |
| `--verification-threshold` | Threshold for verification                             | 0.5     |
| `--augment`                | Augment training data before training                  | False   |

## 📊 Using with Kaggle

### Training on Kaggle

1. Upload your data to Kaggle as a dataset
2. Create a new notebook and add the dataset
3. Run the training with:

```python
import os
os.chdir("/kaggle/working")

# Clone or upload the src folder
from src.main import FaceRecognitionApp

app = FaceRecognitionApp(environment="kaggle", use_checkpoint=False)
history = app.train(epochs=50)
```

### Downloading Trained Model

After training on Kaggle:

```python
from src.utils import export_model_for_download

export_model_for_download(
    model_path="/kaggle/working/siamesemodelv2.keras",
    checkpoint_dir="/kaggle/working/training_checkpoints",
    output_dir="/kaggle/working"
)
```

## 🔧 Configuration

The project automatically detects the environment (local or Kaggle) and configures paths accordingly.

### Local Environment Paths

- Data: `FaceRecognition/data/`
- Model: `FaceRecognition/siamesemodelv2.keras`
- Checkpoints: `FaceRecognition/checkpoints_backup/`

### Kaggle Environment Paths

- Data: `/kaggle/input/dataset/data/`
- Model: `/kaggle/working/siamesemodelv2.keras`
- Checkpoints: `/kaggle/working/training_checkpoints/`

## 🏗️ Architecture

### Siamese Network

The model uses a Siamese architecture with:

1. **Embedding Network:**

   - Conv2D(64, 10x10) → MaxPool → Conv2D(128, 7x7) → MaxPool
   - Conv2D(128, 4x4) → MaxPool → Conv2D(256, 4x4)
   - Flatten → Dense(4096, sigmoid)

2. **L1 Distance Layer:**

   - Computes |embedding1 - embedding2|

3. **Classification:**
   - Dense(1, sigmoid) → Same person probability

### Input/Output

- Input: Two 100x100x3 RGB images
- Output: Probability (0-1) that images are the same person

## 📝 API Reference

### FaceRecognitionApp

```python
from src.main import FaceRecognitionApp

# Initialize
app = FaceRecognitionApp(
    environment="local",  # or "kaggle"
    use_checkpoint=False
)

# Train
history = app.train(epochs=50, augment_data=False)

# Verify
app.verify(camera_index=0, detection_threshold=0.5, verification_threshold=0.5)

# Collect images
app.collect(camera_index=0)

# Evaluate
results = app.evaluate()
```

### Using Individual Components

```python
from src.config import Config, Environment
from src.model import SiameseModel
from src.data import DataLoader
from src.trainer import Trainer
from src.verifier import FaceVerifier

# Create configuration
config = Config(Environment.LOCAL)

# Create and train model
model = SiameseModel(config)
data_loader = DataLoader(config)
train_data, test_data = data_loader.load_dataset()

trainer = Trainer(model, config)
trainer.train(train_data, epochs=50)

# Verify faces
verifier = FaceVerifier(model, config)
results, verified = verifier.verify()
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Memory Error:**

   - The GPU memory growth is set automatically, but you can restart the kernel if issues persist

2. **Camera Not Found:**

   - Try different camera indices (0, 1, 2...)
   - Check camera permissions

3. **Model Not Found:**

   - Ensure `siamesemodelv2.keras` is in the correct location
   - Use `--use-checkpoint` if only checkpoints are available

4. **TensorFlow Import Error (Kaggle):**
   - This is usually a protobuf conflict; restart the kernel

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Based on the Siamese Network paper: "Siamese Neural Networks for One-shot Image Recognition"
- LFW Dataset: Labeled Faces in the Wild

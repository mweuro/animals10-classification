# 🐾 Classification on Animals10 Dataset

This project implements image classification on the **Animals10 dataset** using deep learning models. The solution includes data preprocessing, model training, and evaluation pipelines.

## 🚀 Quick Start

### 1. Setup Environment

First, create a virtual environment in the project directory (tested with Python 3.10) and activate it:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
```

Upgrade pip to the latest version:

```bash
pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements-cpu.txt
```

For GPU *support* use `requirements-gpu.txt` instead. 

### 2. Dataset preparation

Download and split the dataset using the Make command:

```bash
make download_and_split
```

### 3. Model training

Run the fine-tuning training loop:
```python
python main.py
```
#### Outputs:
- 📊 Training logs: `logs/training_log.csv`
- 💾 Best model: `models/resnet18.pth`

### 4. Visualization

Generate training metrics plots:

```python
python src/visualizations.py
```

## 🔬 Future Enhancements

- 📈 Model Comparison: Benchmark ResNet18 against other architectures
- 🏗️ Custom Architecture: Train from scratch with custom CNN designs
- 🔍 Ablation Studies: Analyze impact of different components
- ⚡ Performance Optimization: Improve training speed and accuracy

## 📊 Results

Training progress and model performance metrics are automatically logged and can be visualized for analysis.
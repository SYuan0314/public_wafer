# Wafer Defect Detection and Visualization Toolki

An advanced deep learning-based wafer defect detection system, integrating interactive data visualization and analysis tool.

## 📋 Project Overview

This project aims to develop an automated defect detection and analysis system for wafer manufacturing using machine learning and deep learning technique. The system includes multiple modular tools, providing a complete solution from data processing to result visualization.

## 🚀 Main Features

### 🔧 Core Modules

- **Interactive Grid Rotation Tool** (`wafer_flip.py`): For wafer data rotation, transformation, and visualization
- **Matrix GUI Generator** (`wafer_defect_GUI.py`): Interactive visualization interface for wafer defect data
- **Deep Learning Models**: Defect classification models based on EfficientNet and MobileNet

### 📊 Visualization Features

- Interactive grid rotation (90°/180°/270° rotation)
- Real-time coordinate display and transformation
- Defect type color coding
- Adjustable matrix size (25x25 to 65x65)
- Dataset loading and browsin

## 🛠️ Installation & Environment Setup

### System Requirements
- Python 3.8+
- Matplotlib
- NumPy
- PyTorch (for deep learning models)

### Install Dependencies
```bash
pip install matplotlib numpy torch torchvision
```

## 💻 Usage Instructions

### 1. Interactive Grid Rotation Tool
```bash
python wafer_flip.py
```
- Click buttons to perform grid rotation
- Observe coordinate transformation process
- Use the reset function to return to the initial st

### 2. Wafer Defect Visualization GUI
```bash
python wafer_defect_GUI.py
```
- Adjust matrix size (25-65)
- Generate random defect patterns
- Load real wafer data from dataset
- View color coding for different defect type

### 3. Deep Learning Model Training
See the Jupyter Notebooks for detailed model training and evaluation:
- `Local_EffientNet_b0.ipynb` - EfficientNet implementation
- `mobilenet_v2_l.ipynb` - MobileNet implementation

## 📁 File Structure

```
public_wafer/
├── wafer_flip.py              # Interactive grid rotation tool
├── wafer_defect_GUI.py        # Wafer defect GUI inter
├── wafer_flip.ipynb           # Grid rotation demo notebook
├── Local_EffientNet_b0.ipynb  # EfficientNet model notebook
├── mobilenet_v2_l.ipynb       # MobileNet model notebook
├── mobilenet_edgetpu_v2_l.pth # Pretrained model weight
└── README.md                  # Project documentation
```

## 🎯 Technical Highlights

- **Interactive Design**: Intuitive GUI for data explora
- **Multi-angle Analysis**: Supports data rotation to enhance model training
- **Real-time Visualization**: Instantly displays coordinate transformations and defect distributions
- **Modular Architecture**: Independent functional modules for easy maintenance and extension

## 🔍 Application Scenarios

- Semiconductor manufacturing quality control
- Wafer defect pattern recognition
- Process parameter optimization analysis
- Data augmentation and preprocessing
- Engineer training and educatio

  ## 📈 Future Development

- [ ] Support for more defect types
- [ ] Batch processing functionality
- [ ] Integration of additional deep learning models
- [ ] API interface
- [ ] GUI user experience optimizat

## 🤝 Contribution

Contributions are welcome! Please submit issues and pull requests to help improve this project.

## 📄 License

This project is licensed under the MIT Licens

  

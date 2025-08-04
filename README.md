# 晶圓缺陷檢測與可視化工具

一個基於深度學習的晶圓缺陷檢測系統，結合互動式數據可視化和分析工具。

## 📋 專案概述

本專案旨在開發晶圓製造過程中的缺陷檢測與分析系統，通過機器學習和深度學習技術實現自動化的缺陷識別。系統包含多個工具模組，提供從數據處理到結果可視化的完整解決方案。

## 🚀 主要功能

### 🔧 核心模組

- **互動式網格旋轉工具** (`wafer_flip.py`) - 用於晶圓數據的旋轉變換和可視化
- **矩陣GUI生成器** (`wafer_defect_GUI.py`) - 晶圓缺陷數據的互動式可視化界面
- **深度學習模型** - 基於EfficientNet和MobileNet的缺陷分類模型

### 📊 可視化特性

- 互動式網格旋轉 (90°/180°/270°旋轉功能)
- 即時座標顯示和變換
- 缺陷類型顏色編碼
- 可調整矩陣大小 (25x25 到 65x65)
- 數據集載入和瀏覽功能

## 🛠️ 安裝與環境設置

### 系統需求
- Python 3.8+
- Matplotlib
- NumPy
- PyTorch (用於深度學習模型)

### 安裝依賴
```bash
pip install matplotlib numpy torch torchvision
```

## 💻 使用說明

### 1. 互動式網格旋轉工具
```python
python wafer_flip.py
```
- 點擊按鈕進行網格旋轉操作
- 觀察座標變換過程
- 使用重置功能回到初始狀態

### 2. 晶圓缺陷可視化GUI
```python
python wafer_defect_GUI.py
```
- 調整矩陣大小 (25-65)
- 生成隨機缺陷模式
- 從數據集載入真實晶圓數據
- 查看不同缺陷類型的顏色編碼

### 3. 深度學習模型訓練
查看Jupyter Notebooks獲取詳細的模型訓練和評估過程：
- `Local_EffientNet_b0.ipynb` - EfficientNet模型實現
- `mobilenet_v2_l.ipynb` - MobileNet模型實現

## 📁 檔案結構

```
public_wafer/
├── wafer_flip.py              # 互動式網格旋轉工具
├── wafer_defect_GUI.py        # 晶圓缺陷GUI界面
├── wafer_flip.ipynb           # 網格旋轉演示notebook
├── Local_EffientNet_b0.ipynb  # EfficientNet模型notebook
├── mobilenet_v2_l.ipynb       # MobileNet模型notebook
├── mobilenet_edgetpu_v2_l.pth # 預訓練模型權重
└── README.md                  # 專案說明文件
```

## 🎯 技術特色

- **互動式設計**: 提供直觀的GUI界面進行數據探索
- **多角度分析**: 支援數據旋轉變換以增強模型訓練
- **實時可視化**: 即時顯示座標變換和缺陷分布
- **模組化架構**: 各功能模組獨立，便於維護和擴展

## 🔍 應用場景

- 半導體製造品質控制
- 晶圓缺陷模式識別
- 製程參數優化分析
- 數據增強和預處理
- 工程師培訓和教學

## 📈 未來發展

- [ ] 支援更多缺陷類型
- [ ] 增加批量處理功能
- [ ] 整合更多深度學習模型
- [ ] 提供API接口
- [ ] 優化GUI使用體驗

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進本專案。

## 📄 授權

本專案採用MIT授權條款。

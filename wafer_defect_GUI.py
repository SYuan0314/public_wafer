# Simple Wafer GUI - Functionality Only
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
import numpy as np
import os
from typing import Optional, List
import torch
import torch.nn as nn
import timm
import cv2

class CategoricalCNN(nn.Module):
    """Model architecture from mobilenet_v2_l.ipynb"""
    def __init__(self, num_classes, num_categories=3, embedding_dim=16, backbone_name='mobilenet_edgetpu_v2_l'):
        super().__init__()
        self.num_classes = num_classes
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim)

        # Backbone network
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            in_chans=embedding_dim
        )

        # Get feature dimensions
        num_features = self.backbone.feature_info.channels(-1)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.long()
        embedded_x = self.embedding(x)
        embedded_x = embedded_x.permute(0, 3, 1, 2)
        features = self.backbone(embedded_x)
        last_feature_map = features[-1]
        pooled_features = self.global_pool(last_feature_map).flatten(1)
        output = self.classifier(pooled_features)
        return output

class SimpleWaferGUI:
    """Simple wafer GUI with basic functionality only"""
    
    def __init__(self):
        # Data
        self.size = 25
        self.matrix_values: Optional[List[List[int]]] = None
        self.dataset: Optional[np.ndarray] = None
        self.current_index = 0
        self.max_index = 0
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'mobilenet_edgetpu_v2_l.pth'
        
        # Results storage for table display
        self.inference_results = [0] * 8  # 8 binary values for the table
        
        # Create GUI
        self.fig = plt.figure(figsize=(12, 8))
        
        # Layout: left controls, right matrix, bottom table
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[4, 1])
        self.control_ax = self.fig.add_subplot(gs[0, 0])
        self.matrix_ax = self.fig.add_subplot(gs[0, 1])
        self.table_ax = self.fig.add_subplot(gs[1, :])
        
        # Setup
        self._setup_controls()
        self._setup_table()
        self._load_dataset()
        self._load_model()
        self._setup_click_handler()
    
    def _setup_controls(self):
        """Setup basic controls"""
        ax = self.control_ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Basic text
        ax.text(0, 0.8, 'Index:', fontsize=12)
        ax.text(0, 0.75, f'Range: 0 ~ 114044', fontsize=10)
        
        # Input box
        textbox_ax = plt.axes((0.08, 0.5, 0.15, 0.05))
        self.textbox = widgets.TextBox(textbox_ax, '', initial='0')
        self.textbox.on_submit(self._on_text_submit)
        
        # Load button
        button_ax = plt.axes((0.08, 0.45, 0.15, 0.05))
        self.load_button = widgets.Button(button_ax, 'Load')
        self.load_button.on_clicked(self._on_load_click)
        
        # Inference button
        inference_ax = plt.axes((0.08, 0.36, 0.15, 0.05))
        self.inference_button = widgets.Button(inference_ax, 'Inference', color='#4CAF50')
        self.inference_button.on_clicked(self._on_inference_click)
    
    def _setup_table(self):
        """Setup 2x9 table at bottom"""
        ax = self.table_ax
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 3)
        ax.axis('off')
        
        # Table labels - first cell is 'Result', then the 8 pattern names
        labels = ['Result', 'Center', 'Donut', 'Edge_Loc', 'Edge_ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
        
        # Draw 2x9 grid with wider cells
        cell_width = 1.0
        cell_height = 1.0
        
        for i in range(2):
            for j in range(9):
                # Draw cell
                rect = patches.Rectangle((j * cell_width, (1-i) * cell_height), 
                                       cell_width, cell_height,
                                       facecolor='white', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # Add label in first row, values in second row
                if i == 0 and j < len(labels):
                    ax.text(j * cell_width + cell_width/2, 
                           (1-i) * cell_height + cell_height/2, 
                           labels[j], ha='center', va='center', 
                           fontsize=10, rotation=0)
                elif i == 1:
                    if j == 0:
                        # First cell in second row shows 'Result'
                        ax.text(j * cell_width + cell_width/2, 
                               (1-i) * cell_height + cell_height/2, 
                               'Result', ha='center', va='center', 
                               fontsize=10, fontweight='bold', color='red')
                    elif j <= len(self.inference_results):
                        # Show binary results (1 or 0) for positions 1-8
                        result_idx = j - 1  # Adjust index since first cell is 'Result'
                        ax.text(j * cell_width + cell_width/2, 
                               (1-i) * cell_height + cell_height/2, 
                               str(self.inference_results[result_idx]), ha='center', va='center', 
                               fontsize=12, fontweight='bold', color='blue')
        
        # Remove aspect ratio constraint for wider cells
        # ax.set_aspect('equal')
    
    def _create_pattern_mapping(self):
        """Create mapping from 38 classes to 8 defect patterns based on bit positions"""
        # Mapping from 8-bit patterns to class indices
        bit_to_class = {
            "00000000": 0,
            "10000000": 1,
            "01000000": 2,
            "00100000": 3,
            "00010000": 4,
            "00001000": 5,
            "00000100": 6,
            "00000010": 7,
            "00000001": 8,
            "10100000": 9,
            "10010000": 10,
            "10001000": 11,
            "10000010": 12,
            "01100000": 13,
            "01010000": 14,
            "01001000": 15,
            "01000010": 16,
            "00101000": 17,
            "00100010": 18,
            "00011000": 19,
            "00010010": 20,
            "00001010": 21,
            "10101000": 22,
            "10100010": 23,
            "10011000": 24,
            "10010010": 25,
            "10001010": 26,
            "01101000": 27,
            "01100010": 28,
            "01011000": 29,
            "01010010": 30,
            "01001010": 31,
            "00101010": 32,
            "00011010": 33,
            "10101010": 34,
            "10011010": 35,
            "01101010": 36,
            "01011010": 37
        }
        
        # Create reverse mapping: class_index -> which patterns it belongs to
        class_to_patterns = {}
        for bit_pattern, class_idx in bit_to_class.items():
            patterns = []
            for i, bit in enumerate(bit_pattern):
                if bit == '1':
                    patterns.append(i)
            class_to_patterns[class_idx] = patterns
            
        return class_to_patterns
    
    def _map_to_patterns(self, class_probs, pattern_mapping):
        """Convert 38 class probabilities to 8 pattern probabilities"""
        pattern_probs = [0.0] * 8
        
        # Sum probabilities for each pattern
        for class_idx, prob in enumerate(class_probs):
            if class_idx in pattern_mapping:
                patterns = pattern_mapping[class_idx]
                # Distribute probability among the patterns this class belongs to
                if len(patterns) > 0:
                    prob_per_pattern = prob / len(patterns)
                    for pattern_idx in patterns:
                        pattern_probs[pattern_idx] += prob_per_pattern
        
        return pattern_probs
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = CategoricalCNN(
                num_classes=38,
                num_categories=38,  # Updated from notebook
                embedding_dim=16,
                backbone_name='mobilenet_edgetpu_v2_l'
            )
            
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _load_dataset(self):
        """Load dataset"""
        try:
            if os.path.exists('../Resize_Dataset.npz'):
                self.dataset = np.load('../Resize_Dataset.npz')
                self.max_index = len(self.dataset['image']) - 1
                self._setup_controls()  # Refresh with correct max_index
                if len(self.dataset['image']) > 0:
                    self._load_map(0)
        except Exception as e:
            print(f"Error: {e}")
    
    def _load_map(self, index: int):
        """Load map by index"""
        if self.dataset is None:
            return
        
        try:
            if 0 <= index < len(self.dataset['image']):
                self.current_index = index
                image = self.dataset['image'][index]
                self.size = image.shape[0]
                self.matrix_values = image.tolist()
                self.textbox.set_val(str(index))
                self._draw_matrix()
        except Exception as e:
            print(f"Error: {e}")
    
    def _draw_matrix(self):
        """Draw matrix"""
        ax = self.matrix_ax
        ax.clear()
        
        if self.matrix_values is None:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Basic title
        ax.set_title(f'Index: {self.current_index} Size: {self.size}x{self.size}')
        
        # Draw cells
        for i in range(self.size):
            for j in range(self.size):
                draw_y = self.size - 1 - i
                value = self.matrix_values[i][j]
                
                # Custom colors
                if value == 0:
                    color = '#FFFFFF'  # White
                elif value == 1:
                    color = "#84F1B3"  # Green
                else:
                    color = "#C9463D"  # Red
                
                rect = patches.Rectangle((j, draw_y), 1, 1,
                                       facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Basic axes setup
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        
        plt.draw()
    
    def _setup_click_handler(self):
        """Setup click handler"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_matrix_click)
    
    def _on_matrix_click(self, event):
        """Handle click"""
        if event.inaxes != self.matrix_ax or self.matrix_values is None:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        col = int(x)
        row = self.size - 1 - int(y)
        
        if 0 <= row < self.size and 0 <= col < self.size:
            current = self.matrix_values[row][col]
            new_value = (current + 1) % 3
            self.matrix_values[row][col] = new_value
            self._draw_matrix()
    
    def _on_text_submit(self, text: str):
        """Handle text input"""
        try:
            index = int(text.strip())
            if 0 <= index <= self.max_index:
                self._load_map(index)
        except ValueError:
            pass
    
    def _on_load_click(self, event):
        """Handle load button"""
        self._on_text_submit(self.textbox.text)
    
    def _on_inference_click(self, event):
        """Handle inference button - perform model inference on current wafer map"""
        if self.model is None:
            print("Model not loaded!")
            return
            
        if self.matrix_values is None:
            print("No wafer map loaded!")
            return
        
        try:
            # Prepare input data
            wafer_map = np.array(self.matrix_values, dtype=np.float32)
            
            # Resize to 65x65 as expected by model
            if wafer_map.shape != (65, 65):
                wafer_map = cv2.resize(wafer_map, (65, 65), interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.from_numpy(wafer_map).unsqueeze(0).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Convert to numpy - full 38 class probabilities
            probs_np = probabilities.cpu().numpy()[0]
            
            # Map 38 classes to 8 defect patterns
            pattern_mapping = self._create_pattern_mapping()
            pattern_probs = self._map_to_patterns(probs_np, pattern_mapping)
            
            # Convert probabilities to binary (1 or 0) using threshold
            threshold = 0.5  # You can adjust this threshold
            self.inference_results = [1 if prob > threshold else 0 for prob in pattern_probs]
            
            # Refresh the table to show results
            self._setup_table()
            plt.draw()
            
            print(f"Inference completed. Top prediction: {np.argmax(probs_np)} (confidence: {np.max(probs_np):.3f})")
            
        except Exception as e:
            print(f"Error during inference: {e}")

def main():
    gui = SimpleWaferGUI()
    plt.show()

if __name__ == "__main__":
    main()
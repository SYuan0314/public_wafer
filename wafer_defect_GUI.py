# Simple Wafer GUI - Functionality Only
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
import numpy as np
import os
from typing import Optional, List

class SimpleWaferGUI:
    """Simple wafer GUI with basic functionality only"""
    
    def __init__(self):
        # Data
        self.size = 25
        self.matrix_values: Optional[List[List[int]]] = None
        self.dataset: Optional[np.ndarray] = None
        self.current_index = 0
        self.max_index = 0
        
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
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 2)
        ax.axis('off')
        
        # Table labels
        labels = ['Center', 'Donut', 'Edge_Loc', 'Edge_ring', 'Loc', 'Near_Full', 'Scratch', 'Random', '']
        
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
                
                # Add label in first row
                if i == 0 and j < len(labels):
                    ax.text(j * cell_width + cell_width/2, 
                           (1-i) * cell_height + cell_height/2, 
                           labels[j], ha='center', va='center', 
                           fontsize=10, rotation=0)
        
        # Remove aspect ratio constraint for wider cells
        # ax.set_aspect('equal')
    
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
        """Handle inference button"""
        print("Inference button clicked!")
        # TODO: Add inference logic here

def main():
    gui = SimpleWaferGUI()
    plt.show()

if __name__ == "__main__":
    main()
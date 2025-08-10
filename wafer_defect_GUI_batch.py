import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.widgets as widgets
import numpy as np
import os
from typing import Optional, List
import torch
import cv2

from model import CategoricalCNN

class ScrollableFrame(ttk.Frame):
    """A scrollable frame with a vertical scrollbar only."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class SimpleWaferGUI:
    """GUI with Controls-Table-Wafers layout and 5x5 grid."""
    
    def __init__(self, parent_frame):
        # Data
        self.batch_size = 25
        self.matrix_values_batch: Optional[List[np.ndarray]] = None
        self.dataset: Optional[np.ndarray] = None
        self.current_start_index = 0
        self.max_index = 0
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'mobilenet_edgetpu_v2_l.pth'
        
        # Results storage
        self.inference_results = [[0] * 8 for _ in range(self.batch_size)]
        
        self.wafer_cmap = ListedColormap(['#FFFFFF', '#84F1B3', '#C9463D'])

        # GUI Elements - Figure is now tall again
        self.fig = Figure(figsize=(12, 28), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig.suptitle("Wafer Defect Detection - Batch Viewer", fontsize=16)
        self.table_cell_texts: List[List[matplotlib.text.Text]] = []

        # New top-down layout: Controls, Table, Wafers
        main_gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 8, 10], left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1)
        
        # --- Top Controls Area ---
        control_gs = main_gs[0].subgridspec(1, 8, wspace=0.1)
        self.control_axes = {
            'label': self.fig.add_subplot(control_gs[0, 0]),
            'range': self.fig.add_subplot(control_gs[0, 1]),
            'textbox': self.fig.add_subplot(control_gs[0, 2]),
            'load_button': self.fig.add_subplot(control_gs[0, 3]),
            'inference_button': self.fig.add_subplot(control_gs[0, 4])
        }

        # --- Middle Table Area ---
        self.table_ax = self.fig.add_subplot(main_gs[1])

        # --- Bottom Wafer Map Area (5x5) ---
        matrix_gs = main_gs[2].subgridspec(5, 5, wspace=0.1, hspace=0.1)
        self.matrix_axes = [[self.fig.add_subplot(matrix_gs[i, j]) for j in range(5)] for i in range(5)]
        
        self._setup_controls()
        self._create_table()
        self._load_dataset()
        self._load_model()
        self._update_table_text()
    
    def _setup_controls(self):
        for ax in self.control_axes.values():
            ax.axis('off')
        self.control_axes['label'].text(0.5, 0.5, 'Start Index:', fontsize=12, ha='center', va='center')
        self.range_label = self.control_axes['range'].text(0.5, 0.5, f'Range: 0 ~ {self.max_index}', fontsize=10, ha='center', va='center')
        self.textbox = widgets.TextBox(self.control_axes['textbox'], '', initial='0')
        self.textbox.on_submit(self._on_text_submit)
        self.load_button = widgets.Button(self.control_axes['load_button'], 'Load Batch')
        self.load_button.on_clicked(self._on_load_click)
        self.inference_button = widgets.Button(self.control_axes['inference_button'], 'Batch Inference', color='#4CAF50')
        self.inference_button.on_clicked(self._on_inference_click)

    def _create_table(self):
        ax = self.table_ax
        ax.clear()
        ax.set_title("Inference Results", fontsize=14, pad=20)
        ax.set_xlim(0, 9)
        ax.set_ylim(0, self.batch_size + 1)
        ax.axis('off')
        labels = ['Index', 'Center', 'Donut', 'Edge_Loc', 'Edge_ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
        cell_width, cell_height = 1.0, 1.0
        self.table_cell_texts = []
        for i in range(self.batch_size + 1):
            row_texts = []
            for j in range(9):
                facecolor = 'lightgray' if i == 0 else 'white'
                rect = matplotlib.patches.Rectangle((j * cell_width, (self.batch_size - i) * cell_height), 
                                       cell_width, cell_height,
                                       facecolor=facecolor, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                text_content, ha, va, fweight = '', 'center', 'center', 'normal'
                fsize = 12 if i == 0 else 11
                if i == 0:
                    text_content = labels[j]
                    fweight = 'bold'
                else:
                    row_texts.append(None)
                text_obj = ax.text(j * cell_width + cell_width / 2, 
                                   (self.batch_size - i) * cell_height + cell_height / 2, 
                                   text_content, ha=ha, va=va, fontsize=fsize, fontweight=fweight)
                if i > 0:
                    row_texts[j] = text_obj
            if i > 0:
                self.table_cell_texts.append(row_texts)

    def _update_table_text(self):
        for i in range(self.batch_size):
            wafer_index = self.current_start_index + i
            self.table_cell_texts[i][0].set_text(str(wafer_index))
            for j in range(8):
                result = self.inference_results[i][j]
                self.table_cell_texts[i][j+1].set_text(str(result))
                self.table_cell_texts[i][j+1].set_color('blue' if result == 1 else 'black')
                self.table_cell_texts[i][j+1].set_fontweight('bold')
        self.canvas.draw()

    def _draw_maps_grid(self):
        """Draw 5x5 grid using imshow."""
        if self.matrix_values_batch is None: return
        for i in range(5):
            for j in range(5):
                ax = self.matrix_axes[i][j]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                map_index = i * 5 + j
                if map_index < len(self.matrix_values_batch):
                    wafer_map = self.matrix_values_batch[map_index]
                    ax.set_title(f'Idx: {self.current_start_index + map_index}', fontsize=10)
                    ax.imshow(wafer_map, cmap=self.wafer_cmap, interpolation='none', vmin=0, vmax=2)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        self.canvas.draw()

    def _create_pattern_mapping(self):
        bit_to_class = {
            "00000000": 0, "10000000": 1, "01000000": 2, "00100000": 3, "00010000": 4,
            "00001000": 5, "00000100": 6, "00000010": 7, "00000001": 8, "10100000": 9,
            "10010000": 10, "10001000": 11, "10000010": 12, "01100000": 13, "01010000": 14,
            "01001000": 15, "01000010": 16, "00101000": 17, "00100010": 18, "00011000": 19,
            "00010010": 20, "00001010": 21, "10101000": 22, "10100010": 23, "10011000": 24,
            "10010010": 25, "10001010": 26, "01101000": 27, "01100010": 28, "01011000": 29,
            "01010010": 30, "01001010": 31, "00101010": 32, "00011010": 33, "10101010": 34,
            "10011010": 35, "01101010": 36, "01011010": 37
        }
        class_to_patterns = {}
        for bit_pattern, class_idx in bit_to_class.items():
            class_to_patterns[class_idx] = [i for i, bit in enumerate(bit_pattern) if bit == '1']
        return class_to_patterns

    def _map_to_patterns(self, class_probs, pattern_mapping):
        pattern_probs = [0.0] * 8
        for class_idx, prob in enumerate(class_probs):
            if class_idx in pattern_mapping:
                patterns = pattern_mapping[class_idx]
                if len(patterns) > 0:
                    prob_per_pattern = prob / len(patterns)
                    for pattern_idx in patterns:
                        pattern_probs[pattern_idx] += prob_per_pattern
        return pattern_probs

    def _load_model(self):
        try:
            self.model = CategoricalCNN(num_classes=38, num_categories=38, embedding_dim=16)
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
        dataset_path = '../Resize_Dataset.npz'
        try:
            if os.path.exists(dataset_path):
                self.dataset = np.load(dataset_path)
                self.max_index = len(self.dataset['image']) - 1
                self.range_label.set_text(f'Range: 0 ~ {self.max_index}')
                if len(self.dataset['image']) > 0:
                    self._load_maps_batch(0)
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
    
    def _load_maps_batch(self, start_index: int):
        if self.dataset is None: return
        self.current_start_index = start_index
        self.matrix_values_batch = []
        end_index = min(start_index + self.batch_size, len(self.dataset['image']))
        for i in range(start_index, end_index):
            self.matrix_values_batch.append(self.dataset['image'][i])
        while len(self.matrix_values_batch) < self.batch_size:
            self.matrix_values_batch.append(np.zeros_like(self.matrix_values_batch[0]))
        self.textbox.set_val(str(start_index))
        self._draw_maps_grid()
        self.inference_results = [[0] * 8 for _ in range(self.batch_size)]
        self._update_table_text()

    def _on_text_submit(self, text: str):
        try:
            index = int(text.strip())
            if 0 <= index <= self.max_index:
                self._load_maps_batch(index)
        except ValueError:
            pass
    
    def _on_load_click(self, event):
        self._on_text_submit(self.textbox.text)
    
    def _on_inference_click(self, event):
        if self.model is None or self.matrix_values_batch is None:
            print("Model or data not loaded!")
            return
        try:
            batch_input = []
            for wafer_map in self.matrix_values_batch:
                if wafer_map.shape != (65, 65):
                    wafer_map = cv2.resize(wafer_map, (65, 65), interpolation=cv2.INTER_NEAREST)
                batch_input.append(wafer_map)
            input_tensor = torch.from_numpy(np.array(batch_input, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            probs_np = probabilities.cpu().numpy()
            pattern_mapping = self._create_pattern_mapping()
            batch_results = []
            for i in range(probs_np.shape[0]):
                pattern_probs = self._map_to_patterns(probs_np[i], pattern_mapping)
                threshold = 0.5
                binary_results = [1 if prob > threshold else 0 for prob in pattern_probs]
                batch_results.append(binary_results)
            self.inference_results = batch_results
            self._update_table_text()
            print("Batch inference completed.")
        except Exception as e:
            print(f"Error during batch inference: {e}")

def main():
    root = tk.Tk()
    root.title("Wafer Defect GUI")
    root.geometry("1200x900")
    scrollable_app_frame = ScrollableFrame(root)
    scrollable_app_frame.pack(fill="both", expand=True)
    SimpleWaferGUI(scrollable_app_frame.scrollable_frame)
    root.mainloop()

if __name__ == "__main__":
    main()
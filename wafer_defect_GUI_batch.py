import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.widgets as widgets
import numpy as np
import os
import math
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
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class SimpleWaferGUI:
    """GUI that dynamically rebuilds its layout based on the number of loaded images."""
    
    def __init__(self, parent_frame):
        # Data
        self.num_images = 0
        self.master_image_list: List[np.ndarray] = []
        self.inference_results: List[List[int]] = []
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'mobilenet_edgetpu_v2_l.pth'
        
        self.wafer_cmap = ListedColormap(['#FFFFFF', '#84F1B3', '#C9463D'])

        # GUI Elements
        self.fig = Figure(figsize=(12, 8), dpi=100) # Start with a modest size
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self._load_model()
        self._setup_initial_view()

    def _setup_initial_view(self):
        """Sets up the initial view with only a load button."""
        self.fig.clear()
        self.fig.suptitle("Wafer Defect Detection", fontsize=16)
        
        # Create a single axis for the load button
        ax = self.fig.add_subplot(111)
        ax.axis('off')
        
        # Place a text prompt and the load button
        ax.text(0.5, 0.6, "Please load one or more .npz files to begin.", ha='center', va='center', fontsize=12)
        
        button_ax = self.fig.add_axes([0.4, 0.4, 0.2, 0.1]) # Manually place button
        self.load_npz_button = widgets.Button(button_ax, 'Load NPZ File(s)', color='#FFC107')
        self.load_npz_button.on_clicked(self._on_load_npz_click)
        
        self.canvas.draw()

    def _on_load_npz_click(self, event):
        filepaths = filedialog.askopenfilenames(
            title="Select NPZ files",
            filetypes=(("NumPy NPZ files", "*.npz"), ("All files", "*.*"))
        )
        if not filepaths:
            return

        self.master_image_list = []
        for fpath in filepaths:
            try:
                data = np.load(fpath)
                if 'image' in data:
                    self.master_image_list.extend(data['image'])
                else:
                    print(f"Warning: 'image' key not found in {os.path.basename(fpath)}")
            except Exception as e:
                print(f"Error loading {os.path.basename(fpath)}: {e}")

        self.num_images = len(self.master_image_list)
        if self.num_images == 0:
            print("No images were loaded.")
            self._setup_initial_view()
            return
        
        print(f"Successfully loaded {self.num_images} images from {len(filepaths)} file(s).")
        self.inference_results = [[0] * 8 for _ in range(self.num_images)]
        self._rebuild_gui_layout()

    def _rebuild_gui_layout(self):
        """Clears and rebuilds the entire GUI layout based on the number of images."""
        self.fig.clear()

        # --- Calculate dynamic layout sizes ---
        # Define how many images per row
        IMG_COLS = 5
        # Calculate rows needed for images
        img_rows = math.ceil(self.num_images / IMG_COLS)
        
        # Estimate figure height based on content
        # Ratios: controls:table:images
        # Give more relative height to table and images as num_images grows
        fig_height = 3 + (self.num_images * 0.4) + (img_rows * 2.2)
        self.fig.set_size_inches(12, fig_height)

        main_gs = self.fig.add_gridspec(3, 1, height_ratios=[1, self.num_images * 0.8, img_rows * 2], 
                                        left=0.05, right=0.95, top=0.98, bottom=0.02, hspace=0.15)

        # --- Top Controls Area ---
        control_gs = main_gs[0].subgridspec(1, 4, wspace=0.2)
        self.control_axes = {
            'load_npz': self.fig.add_subplot(control_gs[0, 0]),
            'info': self.fig.add_subplot(control_gs[0, 1]),
            'inference_button': self.fig.add_subplot(control_gs[0, 2])
        }

        # --- Middle Table Area ---
        self.table_ax = self.fig.add_subplot(main_gs[1])

        # --- Bottom Wafer Map Area ---
        matrix_gs = main_gs[2].subgridspec(img_rows, IMG_COLS, wspace=0.1, hspace=0.1)
        self.matrix_axes = [self.fig.add_subplot(matrix_gs[i, j]) for i in range(img_rows) for j in range(IMG_COLS)]

        # --- Populate the new layout ---
        self._setup_dynamic_controls()
        self._create_dynamic_table()
        self._draw_dynamic_maps()
        self._update_dynamic_table()
        self.canvas.draw()

    def _setup_dynamic_controls(self):
        for ax in self.control_axes.values():
            ax.axis('off')
        self.load_npz_button = widgets.Button(self.control_axes['load_npz'], 'Load More NPZ...', color='#FFC107')
        self.load_npz_button.on_clicked(self._on_load_npz_click)
        self.control_axes['info'].text(0.5, 0.5, f'Loaded {self.num_images} images', fontsize=12, ha='center', va='center')
        self.inference_button = widgets.Button(self.control_axes['inference_button'], 'Run Inference', color='#4CAF50')
        self.inference_button.on_clicked(self._on_inference_click)

    def _create_dynamic_table(self):
        ax = self.table_ax
        ax.clear()
        ax.set_title("Inference Results", fontsize=14, pad=20)
        ax.set_xlim(0, 9)
        ax.set_ylim(0, self.num_images + 1)
        ax.axis('off')
        labels = ['Index', 'Center', 'Donut', 'Edge_Loc', 'Edge_ring', 'Loc', 'Near_Full', 'Scratch', 'Random']
        self.table_cell_texts = []
        for i in range(self.num_images + 1):
            row_texts = []
            for j in range(9):
                facecolor = 'lightgray' if i == 0 else 'white'
                rect = matplotlib.patches.Rectangle((j, (self.num_images - i)), 1, 1, facecolor=facecolor, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                text_content, fweight = ('', 'normal')
                fsize = 12 if i == 0 else 11
                if i == 0:
                    text_content, fweight = (labels[j], 'bold')
                else:
                    row_texts.append(None)
                text_obj = ax.text(j + 0.5, (self.num_images - i) + 0.5, text_content, ha='center', va='center', fontsize=fsize, fontweight=fweight)
                if i > 0:
                    row_texts[j] = text_obj
            if i > 0:
                self.table_cell_texts.append(row_texts)

    def _update_dynamic_table(self):
        for i in range(self.num_images):
            self.table_cell_texts[i][0].set_text(str(i))
            for j in range(8):
                result = self.inference_results[i][j]
                self.table_cell_texts[i][j+1].set_text(str(result))
                self.table_cell_texts[i][j+1].set_color('blue' if result == 1 else 'black')
                self.table_cell_texts[i][j+1].set_fontweight('bold')
        self.canvas.draw()

    def _draw_dynamic_maps(self):
        for i in range(len(self.matrix_axes)):
            ax = self.matrix_axes[i]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            if i < self.num_images:
                wafer_map = self.master_image_list[i]
                ax.set_title(f'Idx: {i}', fontsize=10)
                ax.imshow(wafer_map, cmap=self.wafer_cmap, interpolation='none', vmin=0, vmax=2)
            else:
                ax.axis('off') # Hide unused axes
        self.canvas.draw()

    def _on_inference_click(self, event):
        if self.model is None or not self.master_image_list:
            print("Model or data not loaded!")
            return
        try:
            batch_input = []
            for wafer_map in self.master_image_list:
                if wafer_map.shape != (65, 65):
                    wafer_map = cv2.resize(wafer_map, (65, 65), interpolation=cv2.INTER_NEAREST)
                batch_input.append(wafer_map)
            
            input_tensor = torch.from_numpy(np.array(batch_input, dtype=np.float32)).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            probs_np = probabilities.cpu().numpy()
            pattern_mapping = self._create_pattern_mapping()
            
            for i in range(self.num_images):
                pattern_probs = self._map_to_patterns(probs_np[i], pattern_mapping)
                self.inference_results[i] = [1 if prob > 0.5 else 0 for prob in pattern_probs]
            
            self._update_dynamic_table()
            print(f"Inference completed on {self.num_images} images.")
        except Exception as e:
            print(f"Error during batch inference: {e}")

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

    def _create_pattern_mapping(self):
        bit_to_class = {"00000000":0,"10000000":1,"01000000":2,"00100000":3,"00010000":4,"00001000":5,"00000100":6,"00000010":7,"00000001":8,"10100000":9,"10010000":10,"10001000":11,"10000010":12,"01100000":13,"01010000":14,"01001000":15,"01000010":16,"00101000":17,"00100010":18,"00011000":19,"00010010":20,"00001010":21,"10101000":22,"10100010":23,"10011000":24,"10010010":25,"10001010":26,"01100000":27,"01100010":28,"01011000":29,"01010010":30,"01001010":31,"00101010":32,"00011010":33,"10101010":34,"10011010":35,"01101010":36,"01011010":37}
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

def main():
    root = tk.Tk()
    root.title("Dynamic Wafer Defect GUI")
    root.geometry("1200x900")
    scrollable_app_frame = ScrollableFrame(root)
    scrollable_app_frame.pack(fill="both", expand=True)
    SimpleWaferGUI(scrollable_app_frame.scrollable_frame)
    root.mainloop()

if __name__ == "__main__":
    main()

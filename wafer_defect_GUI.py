# 矩陣GUI生成器
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import TextBox, Button
import numpy as np
import os

# 常數定義
DEFAULT_SIZE = 25
MIN_SIZE = 25
MAX_SIZE = 65
FIGURE_SIZE = (14, 10)
ALPHA_VALUE = 0.8

# 值對應顏色
VALUE_COLORS = {
    0: '#ffffff',  # 白色
    1: '#4caf50',  # OK的綠色
    2: '#f44336'   # 紅色
}

class MatrixGUI:
    def __init__(self):
        self.size = DEFAULT_SIZE
        self.matrix_values = None
        self.dataset = None
        self.current_image_index = 0
        
        # 創建圖形和軸
        self.fig, (self.control_ax, self.matrix_ax) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.3)
        
        # 設置控制面板
        self.setup_controls()
        
        # 載入數據集（在控件創建後）
        self.load_dataset()
        
        # 初始化矩陣顯示
        self.generate_matrix()
        
        # 設置點擊事件處理
        self.setup_click_handler()
        
    def setup_controls(self):
        """設置控制面板"""
        self.control_ax.set_xlim(0, 1)
        self.control_ax.set_ylim(0, 1)
        self.control_ax.set_title('Control Panel', fontsize=14, fontweight='bold')
        self.control_ax.axis('off')
        
        # === 矩陣大小設定 ===
        y_pos = 0.85
        self.control_ax.text(0.05, y_pos, 'Matrix Size Settings:', fontsize=12, fontweight='bold')
        
        y_pos -= 0.06
        self.control_ax.text(0.05, y_pos, f'Range: {MIN_SIZE} - {MAX_SIZE}', fontsize=10, style='italic', color='gray')
        
        # 大小輸入框
        textbox_ax = plt.axes([0.12, 0.72, 0.15, 0.05])
        self.textbox = TextBox(textbox_ax, '', initial=str(DEFAULT_SIZE))
        self.textbox.on_submit(self.on_size_change)
        
        # 隨機生成按鈕
        button_ax = plt.axes([0.29, 0.72, 0.15, 0.05])
        self.button = Button(button_ax, 'Random')
        self.button.on_clicked(self.on_generate_random)
        
        # === 數據集載入 ===
        y_pos = 0.65
        self.control_ax.text(0.05, y_pos, 'Dataset Operations:', fontsize=12, fontweight='bold')
        
        y_pos -= 0.06
        self.control_ax.text(0.05, y_pos, 'Image Index:', fontsize=10)
        
        # 圖片索引輸入框
        index_textbox_ax = plt.axes([0.12, 0.52, 0.15, 0.05])
        self.index_textbox = TextBox(index_textbox_ax, '', initial='0')
        
        # 載入按鈕
        load_button_ax = plt.axes([0.29, 0.52, 0.15, 0.05])
        self.load_button = Button(load_button_ax, 'Load Image')
        self.load_button.on_clicked(self.load_from_dataset)
        
        # === 顏色說明 ===
        y_pos = 0.42
        self.control_ax.text(0.05, y_pos, 'Color Legend:', fontsize=12, fontweight='bold')
        
        y_pos -= 0.05
        self.control_ax.text(0.05, y_pos, '■ 0: Normal', fontsize=10, color='gray')
        y_pos -= 0.04
        self.control_ax.text(0.05, y_pos, '■ 1: Type 1', fontsize=10, color='#4caf50')
        y_pos -= 0.04
        self.control_ax.text(0.05, y_pos, '■ 2: Type 2', fontsize=10, color='#f44336')
        
        # === 編輯說明 ===
        y_pos -= 0.08
        self.control_ax.text(0.05, y_pos, 'Edit Mode:', fontsize=12, fontweight='bold')
        y_pos -= 0.05
        self.control_ax.text(0.05, y_pos, '點擊格子可切換值', fontsize=10, style='italic', color='blue')
        y_pos -= 0.04
        self.control_ax.text(0.05, y_pos, '(0 → 1 → 2 → 0)', fontsize=9, color='gray')
        
        # === 狀態顯示 ===
        y_pos = 0.18
        self.control_ax.text(0.05, y_pos, 'Status:', fontsize=12, fontweight='bold')
        
        # 數據集狀態
        self.dataset_text = self.control_ax.text(0.05, 0.12, '', fontsize=9, 
                                                 bbox=dict(boxstyle='round,pad=0.3', 
                                                 facecolor='lightyellow', alpha=0.7))
        
        # 當前狀態
        self.status_text = self.control_ax.text(0.05, 0.05, f'Size: {self.size}x{self.size}', 
                                               fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='lightblue', alpha=0.7))
    
    def on_size_change(self, text):
        """處理尺寸變更"""
        try:
            new_size = int(text)
            if MIN_SIZE <= new_size <= MAX_SIZE:
                self.size = new_size
                self.generate_matrix()
                self.update_status()
            else:
                self.show_error(f"Size must be between {MIN_SIZE} and {MAX_SIZE}")
        except ValueError:
            self.show_error("Please enter a valid integer")
    
    def on_generate_random(self, event):
        """處理隨機生成按鈕點擊"""
        # 使用當前設定的size生成隨機矩陣
        text = self.textbox.text
        try:
            new_size = int(text)
            if MIN_SIZE <= new_size <= MAX_SIZE:
                self.size = new_size
                self.generate_matrix()
                self.update_status()
            else:
                self.show_error(f"Size must be between {MIN_SIZE} and {MAX_SIZE}")
        except ValueError:
            # 如果輸入無效，使用當前size
            self.generate_matrix()
            self.update_status()
    
    def show_error(self, message):
        """顯示錯誤訊息"""
        self.status_text.set_text(f'Error: {message}')
        self.status_text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        plt.draw()
    
    def update_status(self, message=None):
        """更新狀態顯示"""
        base_text = f'Size: {self.size}x{self.size}'
        display_text = f'{base_text} | {message}' if message else base_text
        color = 'lightcyan' if message else 'lightgreen'
        
        self.status_text.set_text(display_text)
        self.status_text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        plt.draw()
    
    def load_dataset(self):
        """載入數據集"""
        try:
            if os.path.exists('../Resize_Dataset.npz'):
                self.dataset = np.load('../Resize_Dataset.npz')
                self.update_dataset_status(f'Dataset loaded: {self.dataset["image"].shape[0]} images')
            else:
                self.update_dataset_status('Dataset not found')
        except Exception as e:
            self.update_dataset_status(f'Error: {str(e)}')
    
    def update_dataset_status(self, message):
        """更新數據集狀態"""
        if hasattr(self, 'dataset_text') and self.dataset_text is not None:
            self.dataset_text.set_text(message)
            plt.draw()
    
    def load_from_dataset(self, event):
        """從數據集載入圖片"""
        if self.dataset is None:
            self.show_error('Dataset not loaded')
            return
        
        try:
            index = int(self.index_textbox.text)
            if 0 <= index < len(self.dataset['image']):
                self.current_image_index = index
                image = self.dataset['image'][index]
                self.size = image.shape[0]  # 65x65
                self.matrix_values = image.tolist()
                self.update_status()
                self.draw_matrix()
            else:
                self.show_error(f'Index must be 0-{len(self.dataset["image"])-1}')
        except ValueError:
            self.show_error('Please enter a valid index')
    
    def generate_matrix(self):
        """生成並顯示隨機矩陣"""
        import random
        # 生成隨機值矩陣（0, 1, 2）
        self.matrix_values = [[random.randint(0, 2) for _ in range(self.size)] 
                             for _ in range(self.size)]
        self.draw_matrix()
    
    def draw_matrix(self):
        """繪製矩陣"""
        self.matrix_ax.clear()
        
        # 設置標題
        self.matrix_ax.set_title(f'{self.size}x{self.size} Matrix', 
                                fontsize=14, fontweight='bold')
        
        # 繪製格子和值
        for i in range(self.size):
            for j in range(self.size):
                # 計算繪製位置
                draw_y = self.size - 1 - i
                
                # 取得格子的值
                value = self.matrix_values[i][j] if self.matrix_values else 0
                color = VALUE_COLORS.get(value, '#ffffff')
                
                # 創建矩形
                rect = patches.Rectangle((j, draw_y), 1, 1, 
                                       linewidth=1, edgecolor='gray', 
                                       facecolor=color, alpha=ALPHA_VALUE)
                self.matrix_ax.add_patch(rect)
                
                # 顯示數值（只在較小矩陣時顯示，避免過於密集）
                if self.size <= 32:
                    self.matrix_ax.text(j + 0.5, draw_y + 0.5, str(value),
                                      ha='center', va='center', fontsize=6, 
                                      fontweight='bold', color='black')
        # 設置軸屬性
        self.matrix_ax.set_xlim(0, self.size)
        self.matrix_ax.set_ylim(0, self.size)
        self.matrix_ax.set_xlabel('X Coordinate', fontsize=10)
        self.matrix_ax.set_ylabel('Y Coordinate', fontsize=10)
        
        # 設置刻度（每5個顯示一次以避免過於密集）
        tick_step = max(1, self.size // 10)
        self.matrix_ax.set_xticks(range(0, self.size + 1, tick_step))
        self.matrix_ax.set_yticks(range(0, self.size + 1, tick_step))
        
        # 添加網格（降低透明度以免干擾小格子）
        self.matrix_ax.grid(True, alpha=0.2)
        
        # 確保比例正確
        self.matrix_ax.set_aspect('equal')
        
        plt.draw()
    
    def setup_click_handler(self):
        """設置點擊事件處理器"""
        self.matrix_ax.figure.canvas.mpl_connect('button_press_event', self.on_matrix_click)
    
    def on_matrix_click(self, event):
        """處理矩陣點擊事件"""
        # 檢查點擊是否在矩陣區域內
        if event.inaxes != self.matrix_ax:
            return
        
        # 獲取點擊座標
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # 轉換為矩陣索引
        col = int(x)
        row = self.size - 1 - int(y)  # 因為y軸是倒置的
        
        # 檢查索引是否有效且矩陣已初始化
        if (0 <= row < self.size and 0 <= col < self.size and 
            self.matrix_values is not None):
            # 循環切換值 (0 -> 1 -> 2 -> 0)
            current_value = self.matrix_values[row][col]
            new_value = (current_value + 1) % 3
            self.matrix_values[row][col] = new_value
            
            # 重新繪製矩陣
            self.draw_matrix()
            
            # 更新狀態顯示
            self.update_edit_status(row, col, current_value, new_value)
    
    def update_edit_status(self, row, col, old_value, new_value):
        """更新編輯狀態顯示"""
        edit_msg = f'編輯: ({col},{row}) {old_value}→{new_value}'
        self.update_status(edit_msg)

def main():
    """主函數"""
    matrix_gui = MatrixGUI()
    plt.show()

if __name__ == "__main__":
    main()
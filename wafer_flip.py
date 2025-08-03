# 互動式旋轉網格 GUI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import matplotlib.patches as patches

# 常數定義
BUTTON_WIDTH = 0.15
BUTTON_HEIGHT = 0.08
BUTTON_Y = 0.05
ALPHA_VALUE = 0.7
FIGURE_SIZE = (10, 8)
BOTTOM_MARGIN = 0.2

class InteractiveGrid:
    def __init__(self, size=5):
        self.size = size
        self.current_rotation = 0
        
        # 初始化座標矩陣和顏色矩陣
        self.reset_grid()

        # 創建圖形和軸
        self.fig, self.ax = plt.subplots(figsize=FIGURE_SIZE)
        self.fig.subplots_adjust(bottom=BOTTOM_MARGIN)
        
        # 創建按鈕
        self.create_buttons()
        
        # 初始化顯示
        self.update_display()
        
    def get_cell_color(self, i, j):
        """取得格子顏色：最底下的row (i==size-1) 為黑色，其餘為白色"""
        if i == self.size - 1:
            return "#000000"  # 最底下的row黑色
        else:
            return "#ffffff"  # 其他全白
        
    def reset_grid(self):
        """重置為原始網格"""
        # 初始化座標矩陣 - 保存格子的原始座標和目前顯示座標
        self.original_coords = np.zeros((self.size, self.size, 2), dtype=int)
        self.display_coords = np.zeros((self.size, self.size, 2), dtype=int)
        # 初始化顏色矩陣 - 最底下的row為黑色，其餘為白色
        self.colors = np.zeros((self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                self.original_coords[i, j] = [j, self.size - 1 - i]
                self.display_coords[i, j] = [j, self.size - 1 - i]
                self.colors[i, j] = self.get_cell_color(i, j)
    
    def create_buttons(self):
        """創建控制按鈕"""
        # 按鈕位置
        ax_btn90 = plt.axes([0.1, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
        ax_btn180 = plt.axes([0.3, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
        ax_btn270 = plt.axes([0.5, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
        ax_reset = plt.axes([0.7, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
        
        # 創建按鈕
        self.btn_90 = Button(ax_btn90, 'rotate 90°')
        self.btn_180 = Button(ax_btn180, 'rotate 180°')
        self.btn_270 = Button(ax_btn270, 'rotate 270°')
        self.btn_reset = Button(ax_reset, 'reset grid')
        
        # 綁定事件
        self.btn_90.on_clicked(self.rotate_90)
        self.btn_180.on_clicked(self.rotate_180)
        self.btn_270.on_clicked(self.rotate_270)
        self.btn_reset.on_clicked(self.reset)
    
    def rotate_grid(self, angle):
        """旋轉格子內部的座標與顏色，座標軸不動，顏色也跟著旋轉"""
        new_display_coords = np.zeros_like(self.display_coords)
        new_colors = np.zeros_like(self.colors)
        for i in range(self.size):
            for j in range(self.size):
                # 旋轉格子內部的顯示座標
                x, y = self.display_coords[i, j]
                if angle == 90:
                    new_x, new_y = y, self.size - 1 - x
                    target_i, target_j = j, self.size - 1 - i
                elif angle == 180:
                    new_x, new_y = self.size - 1 - x, self.size - 1 - y
                    target_i, target_j = self.size - 1 - i, self.size - 1 - j
                elif angle == 270:
                    new_x, new_y = self.size - 1 - y, x
                    target_i, target_j = self.size - 1 - j, i
                else:
                    new_x, new_y = x, y
                    target_i, target_j = i, j
                new_display_coords[target_i, target_j] = [new_x, new_y]
                # 顏色也要跟著格子移動
                new_colors[target_i, target_j] = self.colors[i, j]
        self.display_coords = new_display_coords
        self.colors = new_colors
        self.current_rotation = (self.current_rotation + angle) % 360
    
    def get_rotated_tick_labels(self, axis='x'):
        """軸標籤固定不變"""
        return [str(i) for i in range(self.size + 1)]

    def update_display(self):
        """更新顯示"""
        self.ax.clear()
        
        # 繪製有顏色的格子
        for i in range(self.size):
            for j in range(self.size):
                # 計算正確的繪製位置 (矩陣的i=0應該在視覺上方，即matplotlib的y較大處)
                draw_y = self.size - 1 - i
                # 創建有顏色的矩形
                rect = patches.Rectangle((j, draw_y), 1, 1, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor=self.colors[i, j], alpha=ALPHA_VALUE)
                self.ax.add_patch(rect)
                
                # 顯示旋轉後的座標
                display_x, display_y = self.display_coords[i, j]
                display_text = f"({display_x},{display_y})"
                self.ax.text(j + 0.5, draw_y + 0.5, display_text,
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=ALPHA_VALUE))
        
        # 設置軸屬性
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_title(f'Interactive Grid Rotation (Cumulative Rotation: {self.current_rotation}°)')
        
        # 設置旋轉後的軸標籤
        self.ax.set_xticks(range(self.size + 1))
        self.ax.set_yticks(range(self.size + 1))
        self.ax.set_xticklabels(self.get_rotated_tick_labels('x'))
        self.ax.set_yticklabels(self.get_rotated_tick_labels('y'))
        
        self.ax.grid(True, alpha=0.3)
        
        plt.draw()
    
    def rotate_90(self, event):
        """旋轉90度"""
        self.rotate_grid(90)
        self.update_display()
    
    def rotate_180(self, event):
        """旋轉180度"""
        self.rotate_grid(180)
        self.update_display()
    
    def rotate_270(self, event):
        """旋轉270度"""
        self.rotate_grid(270)
        self.update_display()
    
    def reset(self, event):
        """重置網格"""
        self.current_rotation = 0
        self.reset_grid()
        self.update_display()

# 創建互動式網格 - 可以調整大小
if __name__ == "__main__":
    # 可以修改這個變數來改變網格大小
    grid_size = 5  # 可以改成任何大小，例如 3, 4, 6, 7 等
    interactive_grid = InteractiveGrid(grid_size)
    plt.show()
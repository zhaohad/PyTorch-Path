import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class InteractiveMatplot:
    def __init__(self):
        self.start_x = 0
        self.start_y = 0
        self.is_pressed = False

    def getAxes(self, width, height) -> Axes:
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self.fig.canvas.mpl_connect('scroll_event', self.call_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.call_move)
        self.fig.canvas.mpl_connect('button_release_event', self.call_move)
        self.fig.canvas.mpl_connect('motion_notify_event', self.call_move)
        self.axes = self.fig.add_axes([0, 0, 1, 1])

        self.axes.axis("off")
        return self.axes

    def show(self):
        plt.show()

    # 鼠标拖动 处理事件
    def call_move(self, event):
        mouse_x = event.x
        mouse_y = event.y
        ax_temp = event.inaxes
        if event.name == 'button_press_event':
            if ax_temp and event.button == 1:
                if ax_temp.get_legend():
                    legend_bbox = ax_temp.get_legend().get_window_extent()
                    left_bottom = legend_bbox.get_points()[0]
                    right_top = legend_bbox.get_points()[1]

                    if left_bottom[0] <= mouse_x <= right_top[0] and left_bottom[1] <= mouse_y <= right_top[1]:
                        # 在图例上按下鼠标
                        self.is_pressed = False
                        return
                # 没有图例的情况
                # 在 Axes 上按下鼠标
                self.is_pressed = True
                self.start_x = event.xdata
                self.start_y = event.ydata
                return
        elif event.name == 'button_release_event':
            if ax_temp and event.button == 1:
                self.is_pressed = False
        elif event.name == 'motion_notify_event':
            if ax_temp and event.button == 1 and self.is_pressed:
                if ax_temp.get_legend():
                    legend_bbox = ax_temp.get_legend().get_window_extent()
                    left_bottom = legend_bbox.get_points()[0]
                    right_top = legend_bbox.get_points()[1]

                    if left_bottom[0] <= mouse_x <= right_top[0] and left_bottom[1] <= mouse_y <= right_top[1]:
                        # 在图例上按下鼠标
                        self.is_pressed = False
                        return

                # 没有图例的情况
                x_min, x_max = ax_temp.get_xlim()
                y_min, y_max = ax_temp.get_ylim()
                w = x_max - x_min
                h = y_max - y_min
                mx = event.xdata - self.start_x
                my = event.ydata - self.start_y
                ax_temp.set(xlim=(x_min - mx, x_min - mx + w))
                ax_temp.set(ylim=(y_min - my, y_min - my + h))
                self.fig.canvas.draw_idle()
        return

    def call_scroll(self, event):
        ax_temp = event.inaxes
        # 计算放大缩小后， xlim 和ylim
        if ax_temp:
            x_min, x_max = ax_temp.get_xlim()
            y_min, y_max = ax_temp.get_ylim()
            w = x_max - x_min
            h = y_max - y_min
            cur_x = event.xdata
            cur_y = event.ydata
            cur_X = (cur_x - x_min) / w
            cur_Y = (cur_y - y_min) / h
            if event.button == 'down':
                w = w * 1.1
                h = h * 1.1
            elif event.button == 'up':
                w = w / 1.1
                h = h / 1.1
            new_x = cur_x - w * cur_X
            new_y = cur_y - h * cur_Y
            ax_temp.set(xlim=(new_x, new_x + w))
            ax_temp.set(ylim=(new_y, new_y + h))
            self.fig.canvas.draw_idle()

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, MaxNLocator
import matplotlib.image as mpimg

# 创建一个示例图形
# fig, ax = plt.subplots()

fig = plt.figure(figsize=(500 / 100, 400 / 100), dpi=100)
print(type(fig))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(mpimg.imread("/home/zhaohad/dev/demos/github/PyTorch-Path/data/img.png"), extent=[0, 1, 0, 1], aspect='auto')
ax.axis("off")

startx = 0
starty = 0
mPress = False

# 鼠标拖动 处理事件
def call_move(event):
    global mPress
    global startx
    global starty
    mouse_x = event.x
    mouse_y = event.y
    axtemp = event.inaxes
    if event.name == 'button_press_event':
        if axtemp and event.button == 1:
            if axtemp.get_legend():
                legend_bbox = axtemp.get_legend().get_window_extent()
                left_bottom = legend_bbox.get_points()[0]
                right_top = legend_bbox.get_points()[1]

                if left_bottom[0] <= mouse_x <= right_top[0] and left_bottom[1] <= mouse_y <= right_top[1]:
                    # 在图例上按下鼠标
                    mPress = False
                    return
            # 没有图例的情况
            # 在 Axes 上按下鼠标
            mPress = True
            startx = event.xdata
            starty = event.ydata
            return
    elif event.name == 'button_release_event':
        if axtemp and event.button == 1:
            mPress = False
    elif event.name == 'motion_notify_event':
        if axtemp and event.button == 1 and mPress:
            if axtemp.get_legend():
                legend_bbox = axtemp.get_legend().get_window_extent()
                left_bottom = legend_bbox.get_points()[0]
                right_top = legend_bbox.get_points()[1]

                if left_bottom[0] <= mouse_x <= right_top[0] and left_bottom[1] <= mouse_y <= right_top[1]:
                    # 在图例上按下鼠标
                    mPress = False
                    return

            # 没有图例的情况
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            w = x_max - x_min
            h = y_max - y_min
            mx = event.xdata - startx
            my = event.ydata - starty
            axtemp.set(xlim=(x_min - mx, x_min - mx + w))
            axtemp.set(ylim=(y_min - my, y_min - my + h))
            fig.canvas.draw_idle()  # 绘图动作实时反映在图像上

    return


# 滚轮滚动 处理事件
def call_scroll(event):
    # print(event.name)
    axtemp = event.inaxes
    # print('event:', event)
    # print(event.xdata, event.ydata)
    # 计算放大缩小后， xlim 和ylim
    if axtemp:
        x_min, x_max = axtemp.get_xlim()
        y_min, y_max = axtemp.get_ylim()
        w = x_max - x_min
        h = y_max - y_min
        curx = event.xdata
        cury = event.ydata
        curXposition = (curx - x_min) / w
        curYposition = (cury - y_min) / h
        if event.button == 'down':
            w = w * 1.1
            h = h * 1.1
        elif event.button == 'up':
            w = w / 1.1
            h = h / 1.1
        newx = curx - w * curXposition
        newy = cury - h * curYposition
        axtemp.set(xlim=(newx, newx + w))
        axtemp.set(ylim=(newy, newy + h))
        fig.canvas.draw_idle()  # 绘图动作实时反映在图像上


fig.canvas.mpl_connect('scroll_event', call_scroll)
fig.canvas.mpl_connect('button_press_event', call_move)
fig.canvas.mpl_connect('button_release_event', call_move)
fig.canvas.mpl_connect('motion_notify_event', call_move)


plt.show()


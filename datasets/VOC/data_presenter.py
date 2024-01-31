import os

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from VOC2012Dataset import VOC2012Dataset

from VOC2012Dataset import RawData
from utils.interactive_matplot import InteractiveMatplot


class DataPresenter:

    def show_raw_data(self, data: RawData):
        img = plt.imread(data.img_path)
        plot = InteractiveMatplot()
        ax = plot.getAxes(img.shape[1], img.shape[0])
        self.show_data_in_axes(ax, data)
        plot.show()

    def show_data_in_axes(self, ax: Axes, data: RawData):
        ax.imshow(mpimg.imread(data.img_path), aspect='auto')
        ax.axis("off")

        colorMap = {}

        colorI = 0
        for obj in data.objects:
            x = obj.bndbox.xmin
            y = obj.bndbox.ymin
            w = obj.bndbox.xmax - x
            h = obj.bndbox.ymax - y
            name = obj.name
            if name not in colorMap:
                colorMap[name] = self._get_next_color(colorI)
                colorI += 1
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=colorMap[name], facecolor='none')
            ax.set_title(os.path.basename(data.img_path))
            ax.text(x + 6, y + 6, name, color=colorMap[name], fontsize=12, va="top")
            ax.add_patch(rect)

    def show_25_data(self, datas: list[RawData]):
        N = 5
        M = 5
        fig, axs = plt.subplots(N, M)
        for i in range(N):
            for j in range(M):
                data = datas[i * N + j]
                self.show_data_in_axes(axs[i, j], data)
        plt.show()

    def show_class(self, dataset: VOC2012Dataset, cls: str):
        self.show_25_data(self._find_classes(dataset, cls))
        pass

    def _find_classes(self, dataset: VOC2012Dataset, cls: str) -> list[RawData]:
        res = []
        for data in dataset.train:
            for obj in data.objects:
                if obj.name.__eq__(cls):
                    print(data)
                    res.append(data)
                    break
        return res


    def _get_next_color(self, i):
        color_list = ['red', "green", "blue", "cyan", "magenta", "yellow", "black", "white"]
        c = color_list[i]
        return c


import matplotlib.pyplot as plt
from torch import Tensor
import matplotlib.patches as patches


class DataPresenter:
    def show_img(self, img: Tensor, boxes: Tensor = None):
        img_np = img.permute(1, 2, 0).numpy()
        plt.imshow(img_np)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.unbind(0)
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)

        plt.show()

    def _get_next_color(self, i):
        color_list = ['red', "green", "blue", "cyan", "magenta", "yellow", "black", "white"]
        c = color_list[i]
        return c


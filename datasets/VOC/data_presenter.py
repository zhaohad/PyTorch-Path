import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils.interactive_matplot import InteractiveMatplot

from VOC2012Dataset import RawData


class DataPresenter:
    def show_raw_data(self, data: RawData):
        img = plt.imread(data.img_path)
        plot = InteractiveMatplot()
        ax = plot.getAxes(img.shape[1], img.shape[0])
        ax.imshow(mpimg.imread(data.img_path), extent=[0, 1, 0, 1], aspect='auto')
        ax.axis("off")
        plot.show()

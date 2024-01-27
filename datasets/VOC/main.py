import os
from VOC2012Dataset import VOC2012Dataset
from data_presenter import DataPresenter

VOC_DATA_ROOT_PATH = "data/VOCdevkit/VOC2012"


def main():
    presenter = DataPresenter()
    data = VOC2012Dataset(VOC_DATA_ROOT_PATH)
    presenter.show_raw_data(data.train[0])


if __name__ == "__main__":
    main()

import os
from VOC2012Dataset import VOC2012Dataset
from data_presenter import DataPresenter

VOC_DATA_ROOT_PATH = "data/VOCdevkit/VOC2012"


def main():
    presenter = DataPresenter()
    data = VOC2012Dataset(VOC_DATA_ROOT_PATH, train_val="train")
    # presenter.show_class(data, "dog")
    # presenter.show_raw_data(data.val[0])
    print(data.__len__())


if __name__ == "__main__":
    main()

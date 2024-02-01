
import os

import torch
from torch.utils.data import DataLoader

from datasets.VOC.VOC2012Dataset import VOC2012Dataset
from datasets.VOC.data_presenter import DataPresenter

VOC_DATA_ROOT_PATH = "data/VOCdevkit/VOC2012"
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    voc_dataset = VOC2012Dataset(VOC_DATA_ROOT_PATH, train_val="train")
    # presenter = DataPresenter()
    # presenter.show_class(voc_dataset, 'dog')

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_data_loader = DataLoader(voc_dataset,
                                   # batch_sampler=train_batch_sampler,
                                   batch_size=8,
                                   pin_memory=True,
                                   # num_workers=nw,
                                   num_workers=1,
                                   collate_fn=voc_dataset.collate_fn)

    print(f"len(voc_dataset.train) = {len(voc_dataset.train)}")
    print(f"len(train_data_loader) = {len(train_data_loader)}")
    for batch, a in enumerate(train_data_loader):
        # print(a)
        # print(type(a))
        # print(a[0])
        # print(type(a[0]))
        # print(a[0][0])
        # print(type(a[0][0]))
        break


if __name__ == "__main__":
    main()

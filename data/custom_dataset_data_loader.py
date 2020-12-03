import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt, dataset_type="aligned"):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    from data.plain_dataset import PlainDataset
    if dataset_type == "aligned":
        dataset = AlignedDataset()
    elif dataset_type == "plain":
        dataset = PlainDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, dataset_type):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, dataset_type)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataLoader(opt, dataset_type="aligned"):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, dataset_type)
    return data_loader

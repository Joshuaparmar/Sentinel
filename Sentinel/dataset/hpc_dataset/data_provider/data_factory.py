from Sentinel.dataset.hpc_dataset.data_provider.dataset import Dataset_OLCF, Dataset_OLCF_anomaly, Dataset_OLCF_Cluster, Dataset_OLCF_MMD
from torch.utils.data import DataLoader

data_dict = {
    'olcf': Dataset_OLCF,
    'olcf_anomaly': Dataset_OLCF_anomaly,
    'olcf_cluster': Dataset_OLCF_Cluster,
    'olcf_mmd': Dataset_OLCF_MMD,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

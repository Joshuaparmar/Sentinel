import os



def store_featurelist(dataset, df):
    if os.path.isfile(f"{dataset}/features.txt"):
        raise Exception(f"{dataset}/features.txt already exists")
    feature_list = extract_featurelist(df)
    with open(f"{dataset}/features.txt", "w") as f:
        for item in feature_list:
            f.write("%s\n" % item)
    return feature_list

def extract_featurelist(df):
    feature_list = []
    for i in df.columns:
        excluded_values = [
            "attack",
            "timestamp",
            "label",
            "timestamp_memory",
            "timestamp_disk",
            "timestamp_slurm",
            "instance",
            "timestamp_network",
            "timestamp_cpu",
            "timestamp_gpu",
            "timestamp_power",
            "timestamp_job",
            "timestamp_job_event",
            "timestamp_job_resrc_usage",
            "timestamp_job_resrc_used",
            "timestamp_job_resrc_alloc",
            "node",
            "node_state",
            "hostname",
            "yymmdd",
        ]
        if i not in excluded_values:
            feature_list.append(i)
    return feature_list


def get_feature_map(dataset,flag=None):
    filename = 'features.txt'
    if flag == 'train_cluster' or flag == 'val_cluster' or flag == 'test_cluster':
        filename = 'features_cluster.txt'
    if flag == 'train_mmd' or flag == 'val_mmd' or flag == 'test_mmd':
        filename = 'features_mmd_sample.txt'
    if not os.path.isfile(f"{dataset}/{filename}"):
        print(f"{dataset}/{filename} not found")
        return None
    else:
        feature_file = open(f"{dataset}/{filename}", "r")
        feature_list = []
        for ft in feature_file:
            if "#" not in ft:
                feature_list.append(ft.strip())
        return feature_list


import numpy as np
import torch
import contextlib

__all__ = ['cifar_iid']

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    #num_items=8
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_overlap(dataset, num_users, overlap_ratio):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    #num_items=8
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    class_indices = {}
    for i, (image, label) in enumerate(dataset):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)

    # overlap_ratio = 0.2

    for i in range(num_users):
        dict_users[i] = []
        for label in class_indices:
            N = len(class_indices[label])
            N_overlap = int(N * overlap_ratio)
            # start = int((i/num_users) * (N-N_overlap) + N_overlap)
            # end = int(((i+1)/num_users) * (N-N_overlap) + N_overlap)
            start = int((i/num_users) * (N-N_overlap))
            end = int((i/num_users) * (N-N_overlap) + N_overlap)
            # print(N_overlap, start, end)
            # breakpoint()
            # samples = class_indices[label][:N_overlap] + class_indices[label][start:end]
            samples = class_indices[label][start:end]
            dict_users[i].append(samples)

        dict_users[i] = set(np.concatenate(dict_users[i]))
            # dict_users[i].append(class_indices[label][start:end])

    #breakpoint()
        # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# def cifar_overlap(dataset, num_users, overlap_ratio):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     #num_items=8
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     class_indices = {}
#     for i, (image, label) in enumerate(dataset):
#         if label not in class_indices:
#             class_indices[label] = []
#         class_indices[label].append(i)

#     # overlap_ratio = 0.2

#     for i in range(num_users):
#         dict_users[i] = []
#         for label in class_indices:
#             N = len(class_indices[label])
#             # N_overlap = int(N * overlap_ratio)
#             N_overlap = int(N * overlap_ratio)
#             start = int((i/num_users) * (N-N_overlap) + N_overlap)
#             end = int(((i+1)/num_users) * (N-N_overlap) + N_overlap)
#             # print(N_overlap, start, end)
#             # breakpoint()
#             samples = class_indices[label][:N_overlap] + class_indices[label][start:end]
#             dict_users[i].append(samples)

#         dict_users[i] = set(np.concatenate(dict_users[i]))
#             # dict_users[i].append(class_indices[label][start:end])

#         # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         # all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


def cifar_noniid(dataset, num_clients, class_per_client=1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 250
    # num_shards, num_imgs = 200, 250
    
    num_shards = num_clients * class_per_client
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)

    labels = []
    for element in dataset:
        labels.append(int(element[1]))
    # print(type(labels[0]))
    labels = np.array(labels)
    # labels=labels.astype('int64')
    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, class_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        # breakpoint()
        for rand in rand_set:
            try:
                # dict_users[i] = set(np.concatenate((list(dict_users[i]), idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0))
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            except:
                breakpoint()

        dict_users[i] = set(dict_users[i])
        
    return dict_users



def cifar_dirichlet_unbalanced(dataset, n_nets, alpha=0.5):
    '''
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    '''
    #X_train=dataset[:][0]
    y_train=torch.zeros(len(dataset),dtype=torch.long)
    print(y_train.dtype)
    for a in range(len(dataset)):
        y_train[a]=(dataset[a][1])
    n_train = len(dataset)
    #X_train.shape[0]
    '''
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    '''
    #elif partition == "hetero-dir":
    min_size = 0
    K = len(dataset.class_to_idx)
    N = len(dataset)
    N = y_train.shape[0]
    net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    #traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return net_dataidx_map
    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def cifar_dirichlet_balanced(dataset, n_nets, alpha=0.5):
    with temp_seed(0):
        y_train=torch.zeros(len(dataset),dtype=torch.long)

        for a in range(len(dataset)):
            y_train[a]=(dataset[a][1])
        n_train = len(dataset)

        min_size = 0
        K = len(dataset.class_to_idx)
        N = len(dataset)
        N = y_train.shape[0]
        print(N)
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}
        assigned_ids = []
        idx_batch = [[] for _ in range(n_nets)]
        num_data_per_client=int(N/n_nets)
        for i in range(n_nets):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k]=proportions[k]
            weights[assigned_ids] = 0.0
            idx_batch[i] = (torch.multinomial(weights, num_data_per_client, replacement=False)).tolist()
            assigned_ids+=idx_batch[i]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    #traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return net_dataidx_map
    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def cifar_toyset(dataset, num_users=3, num_valid_classes=3, limit_number_per_class = 500, toy_noniid_rate=0.1, non_iid = True):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 200, 250
    # num_shards, num_imgs = 200, 250
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)

    labels = []
    for element in dataset:
        labels.append(int(element[1]))
    # print(type(labels[0]))
    labels = np.array(labels)
    # labels=labels.astype('int64')
    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #idxs = idxs_labels[0, :]


    num_total_classes = len(dataset.classes)
    samples_per_classes = int(len(dataset)/num_total_classes)

    selected_classes_idxs_labels = idxs_labels[:,idxs_labels[1]<num_valid_classes]
    idxs = selected_classes_idxs_labels[0, :]
    valid_num_per_class = min(samples_per_classes, limit_number_per_class)
    if non_iid:
        gap = int(toy_noniid_rate/num_users*valid_num_per_class)
    else:
        gap = int(1/num_users*valid_num_per_class)

    print("gap, valid_num_per_class, samples_per_classes : ",gap, valid_num_per_class, samples_per_classes)

    # divide and assign
    for i in range(num_users):
        for j in range(num_valid_classes):
            dict_users[i] = (np.concatenate(
                    (dict_users[i], idxs[samples_per_classes*j + gap*i:samples_per_classes*j + gap*(i+1)]), axis=0)) 
            if i==j:
                dict_users[i] = (np.concatenate(
                    (dict_users[i], idxs[samples_per_classes*j + gap*num_users:samples_per_classes*(j) + valid_num_per_class  ]), axis=0)) 
                
                
        dict_users[i] = set(dict_users[i])
        #dict_users[i] = set(list(dict_users[i])[:limit_number_per_class])
    for i in range(num_users):
        for c in range(num_total_classes):
            this_label_num = sum(labels[list(dict_users[i])] == c)
            print("client_idx, class_idx, num_samples_class :",i,c,this_label_num)
    return dict_users
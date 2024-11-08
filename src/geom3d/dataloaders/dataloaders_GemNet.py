import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from geom3d.datasets.dataset_GemNet_utils import get_id_data_list
from geom3d.datasets.dataset_GemNet_utils import get_id_data_list_for_material


class BatchGemNet(Data):
    def __init__(self, **kwargs):
        super(BatchGemNet, self).__init__(**kwargs)
        return

    @staticmethod
    
    def from_data_list(data_list, cutoff, int_cutoff, triplets_only):
        print("data_list: ", data_list)
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        
        index_keys = [
            "id_undir", "id_swap",
            "id_c", "id_a",
            "id3_expand_ba", "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            index_keys += [
                "id4_int_b", "id4_int_a",
                "id4_reduce_ca", "id4_expand_db", "id4_reduce_cab", "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca", "id4_expand_intm_db", "id4_reduce_intm_ab", "id4_expand_intm_ab",
            ]

        batch = BatchGemNet()

        # Initialize lists for each key
        for key in keys:
            batch[key] = []

        # Initialize batch and track max_num_nodes
        batch.batch = []
        max_num_nodes = 0

        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            max_num_nodes = max(max_num_nodes, num_nodes)

            # Track the batch index for each node
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))

            for key in keys:
                item = data[key]

                # Pad the tensor to the max_num_nodes
                pad_size = max_num_nodes - num_nodes
                item = torch.nn.functional.pad(item, (0, 0, 0, pad_size), value=0)

                batch[key].append(item)

        # Concatenate along the node dimension
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=0)

        # Concatenate along the batch dimension
        batch.batch = torch.cat(batch.batch, dim=0)

        index_batch = get_id_data_list(
            data_list, cutoff=cutoff, int_cutoff=int_cutoff, index_keys=index_keys, triplets_only=triplets_only)

        for index_key in index_keys:
            batch[index_key] = torch.tensor(index_batch[index_key], dtype=torch.int64)

        return batch.contiguous()
    
    @property
    def num_graphs(self):
        '''Returns the number of graphs in the batch.'''
        return self.batch[-1].item() + 1


class DataLoaderGemNet(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, cutoff, int_cutoff, triplets_only, **kwargs):
        # super(DataLoaderGemNet, self).__init__(
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda data_list: BatchGemNet.from_data_list(data_list, cutoff, int_cutoff, triplets_only),
            **kwargs)



# class BatchGemNet(Data):
#     def __init__(self, **kwargs):
#         super(BatchGemNet, self).__init__(**kwargs)
#         return

#     @staticmethod
#     def from_data_list(data_list, cutoff, int_cutoff, triplets_only):
#         print("data_list: ", data_list)
#         keys = [set(data.keys) for data in data_list]
#         keys = list(set.union(*keys))
        
#         index_keys = [
#             "id_undir", "id_swap",
#             "id_c", "id_a",
#             "id3_expand_ba", "id3_reduce_ca",
#             "Kidx3",
#         ]
#         if not triplets_only:
#             index_keys += [
#                 "id4_int_b", "id4_int_a",
#                 "id4_reduce_ca", "id4_expand_db", "id4_reduce_cab", "id4_expand_abd",
#                 "Kidx4",
#                 "id4_reduce_intm_ca", "id4_expand_intm_db", "id4_reduce_intm_ab", "id4_expand_intm_ab",
#             ]

#         batch = BatchGemNet()

#         for key in keys:
#             batch[key] = []
#         batch.batch = []
#         for i, data in enumerate(data_list):
#             num_nodes = data.x.size()[0]
#             batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
#             for key in data.keys:
#                 item = data[key]
#                 batch[key].append(item)

#         for key in keys:
#             batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
#         batch.batch = torch.cat(batch.batch, dim=-1)

#         index_batch = get_id_data_list(
#             data_list, cutoff=cutoff, int_cutoff=int_cutoff, index_keys=index_keys, triplets_only=triplets_only)

#         for index_key in index_keys:
#             batch[index_key] = torch.tensor(index_batch[index_key], dtype=torch.int64)

#         return batch.contiguous()
    
#     @property
#     def num_graphs(self):
#         '''Returns the number of graphs in the batch.'''
#         return self.batch[-1].item() + 1


# class DataLoaderGemNet(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle, num_workers, cutoff, int_cutoff, triplets_only, **kwargs):
#         # super(DataLoaderGemNet, self).__init__(
#         super().__init__(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             collate_fn=lambda data_list: BatchGemNet.from_data_list(data_list, cutoff, int_cutoff, triplets_only),
#             **kwargs)
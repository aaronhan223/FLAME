from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import numpy as np
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import pdb


def data_prepare(args, task, modeltype, data=None):
    """
    Prepare the data for training or evaluation.

    Args:
        args (object): The arguments object.
        mode (str): The mode, either 'train' or 'eval'.
        modeltype (str): The model type.
        data (list, optional): The data to be used. Defaults to None.

    Returns:
        dataset (object): The dataset object.
        sampler (object): The sampler object.
        dataloader (object): The dataloader object.
    """
    train_dataset = EmbedDataset(args, 'train', task, modeltype, data=data)
    val_dataset = EmbedDataset(args, 'val', task, modeltype, data=data)
    test_dataset = EmbedDataset(args, 'test', task, modeltype, data=data)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs_embed, collate_fn=EMBEDcollate_fn)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.train_bs_embed, collate_fn=EMBEDcollate_fn)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_bs_embed, collate_fn=EMBEDcollate_fn)

    return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset


class EmbedDataset(Dataset):
    def __init__(self, args, mode, task, modeltype, data=None):
        if task == 'risk':
            self.task = 'cancer_risk_5yr'
        elif task == 'density':
            self.task = 'tissue_density'
        else:
            self.task = task
        
        if data is not None:
            self.data = data
        else:
            self.data = load_data(args.embed_path, mode, debug=args.debug, task=self.task)

        self.modeltype = modeltype
        self.model_name=args.model_name
        self.use_pt_text_embeddings=args.use_pt_text_embeddings
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_detail = self.data[idx]
        idx = data_detail['name']
        label= data_detail["label"] - 1 if self.task=='tissue_density' else data_detail["label"] # 0-indexed labels for density
        img_feats = np.array(data_detail["img_feats"])
        embed_2d_cc = np.array(data_detail["embed_2d_cc"])
        embed_2d_mlo = np.array(data_detail["embed_2d_mlo"])
        embed_c_view_cc = np.array(data_detail["embed_c_view_cc"])
        embed_c_view_mlo = np.array(data_detail["embed_c_view_mlo"])
        return_dict = {'idx':idx, 'label':label}
        
        if 'allviews' in self.modeltype:
            return_dict['img_feats'] = img_feats
            return return_dict
        if 'cc' in self.modeltype:
            return_dict['cc'] = embed_c_view_cc
        if 'mlo' in self.modeltype:
            return_dict['mlo'] = embed_c_view_mlo
        if '2dcc' in self.modeltype:
            return_dict['2dcc'] = embed_2d_cc
        if '2dmlo' in self.modeltype:
            return_dict['2dmlo'] = embed_2d_mlo

        return return_dict


def load_data(file_path, mode, debug=False, text=False, task='ihm'):
    """
    Load data from a file.

    Args:
        file_path (str): The path to the file.
        mode (str): The mode of the data.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        text (bool, optional): Whether the data is text. Defaults to False.
        task (str, optional): The task of the data. Defaults to 'ihm'.

    Returns:
        data: The loaded data.
    """
    dataPath = os.path.join(file_path, mode + '_' + task + '.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if debug and not text:
                data = data[:100]
    return data

def EMBEDcollate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch = {key: [d[key] for d in batch] for key in batch[0]}
    idx = torch.tensor(batch['idx'])
    label = torch.tensor(batch['label'])
    
    embed_2d_cc = torch.stack([torch.as_tensor(v) for v in batch['2dcc']]) if '2dcc' in batch else None
    embed_2d_mlo = torch.stack([torch.as_tensor(v) for v in batch['2dmlo']]) if '2dmlo' in batch else None
    embed_c_view_cc = torch.stack([torch.as_tensor(v) for v in batch['cc']]) if 'cc' in batch else None
    embed_c_view_mlo = torch.stack([torch.as_tensor(v) for v in batch['mlo']]) if 'mlo' in batch else None
    img_feats = torch.stack([torch.as_tensor(v) for v in batch['img_feats']]) if 'img_feats' in batch else None

    
    return idx, label, embed_2d_cc, embed_2d_mlo, embed_c_view_cc, embed_c_view_mlo, img_feats
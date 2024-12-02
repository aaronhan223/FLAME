import os
from typing import List
import random
import logging
import dill as pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


def read_txt(filename):
    logging.info(f"Reading from {filename}")
    data = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            data.append(line)
    return data


def load_pickle(filename):
    logging.info(f"Data loaded from {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def read(file, dtype='float'):
    with open(file) as file:
        header = file.readline().split(' ')
        count = int(header[0])
        dim = int(header[1])
        matrix = np.empty((count, dim), dtype=dtype)
        for i in range(count):
            matrix[i] = np.fromstring(file.readline(), sep=' ', dtype=dtype)
    return matrix


def to_index(sequence: List[str], vocab, prefix="", suffix=""):
    """ convert code to index (each timestamp contains one token) """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    sequence = torch.tensor(sequence)
    return sequence


def collate_fn(data):
    data = {k: [d[k] for d in data] for k in data[0]}
    data["age"] = torch.stack(data["age"])
    data["gender"] = torch.stack(data["gender"])
    data["ethnicity"] = torch.stack(data["ethnicity"])
    # padding
    data["timestamps"] = torch.nn.utils.rnn.pad_sequence(
        data["timestamps"], batch_first=True, padding_value=0
    )
    data["types"] = torch.nn.utils.rnn.pad_sequence(
        data["types"], batch_first=True, padding_value=0
    )
    data["codes"] = torch.nn.utils.rnn.pad_sequence(
        data["codes"], batch_first=True, padding_value=0
    )
    data["mortality"] = torch.stack(data["mortality"])
    data["readmission"] = torch.stack(data["readmission"])
    return data


class Vocabulary(object):

    def __init__(self, init_words=None):
        if init_words is None:
            init_words = ["<pad>", "<cls>", "<eos>", "<sep>", "<unk>"]
        self.init_words = init_words
        self.word2idx = {word: idx for idx, word in enumerate(init_words)}
        self.idx2word = {idx: word for idx, word in enumerate(init_words)}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            try:
                return self.word2idx["<unk>"]
            except KeyError:
                raise KeyError(f"word {word} not in vocab and <unk> not in vocab")
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class eICUTokenizer:
    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
        self.code_vocabs, self.code_vocabs_size, self.code_embeddings = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()
        self.age_vocabs, self.age_vocabs_size = self._load_age_vocabs()
        self.gender_vocabs, self.gender_vocabs_size = self._load_gender_vocabs()
        self.ethnicity_vocabs, self.ethnicity_vocabs_size = self._load_ethnicity_vocabs()

    def _load_code_vocabs(self):
        vocab_dir = os.path.join(self.processed_data_path, f"vocab.pkl")
        vocabs = load_pickle(vocab_dir)
        vocabs_size = len(vocabs)
        embeddings = read(os.path.join(self.processed_data_path, f"embeddings.txt"))
        return vocabs, vocabs_size, embeddings

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in [
            # "pasthistory",
            # "admissiondx",
            # "admissiondrug",
            "diagnosis",
            "treatment",
            "medication",
            "lab",
            # "physicalexam",
        ]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_age_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in range(18, 90):
            word = word // 10 * 10
            vocabs.add_word(str(word))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_gender_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["Female", "Male", "Other", "Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_ethnicity_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Other/Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def __call__(
            self,
            age: str,
            gender: str,
            ethnicity: str,
            timestamps: List[int],
            types: List[str],
            codes: List[str]
    ):
        age = str(int(age) // 10 * 10)
        age = torch.tensor(self.age_vocabs(age))
        gender = torch.tensor(self.gender_vocabs(gender))
        ethnicity = torch.tensor(self.ethnicity_vocabs(ethnicity))
        timestamps = torch.tensor([0] + timestamps + [timestamps[-1]])
        types = to_index(types, self.type_vocabs, prefix="<cls>", suffix="<eos>")
        codes = to_index(codes, self.code_vocabs, prefix="<cls>", suffix="<eos>")
        return age, gender, ethnicity, timestamps, types, codes


class eICUDataset(Dataset):
    def __init__(self, split, processed_data_path=None, return_raw=False):
        self.split = split
        self.all_icu_stay_dict = load_pickle(os.path.join(processed_data_path, "icu_stay_dict.pkl"))
        self.split_icu_ids = []
        self.split_icu_ids.extend(read_txt(os.path.join(processed_data_path, f"{split}_icu_ids.txt")))
        self.return_raw = return_raw
        self.tokenizer = eICUTokenizer(processed_data_path)

    def __len__(self):
        return len(self.split_icu_ids)

    def subsample(self, size):
        random.shuffle(self.split_icu_ids)
        self.split_icu_ids = self.split_icu_ids[:size]

    def __getitem__(self, index):
        icu_id = self.split_icu_ids[index]
        icu_stay = self.all_icu_stay_dict[icu_id]

        icu_id = icu_id
        age = str(icu_stay.age)
        gender = icu_stay.gender
        ethnicity = icu_stay.ethnicity
        timestamps = icu_stay.trajectory[0]
        types = icu_stay.trajectory[1]
        codes = icu_stay.trajectory[2]
        mortality = float(icu_stay.mortality)
        readmission = float(icu_stay.readmission)

        if not self.return_raw:
            age, gender, ethnicity, timestamps, types, codes = self.tokenizer(
                age, gender, ethnicity, timestamps, types, codes
            )
            mortality = torch.tensor(mortality)
            readmission = torch.tensor(readmission)

        return_dict = dict()
        return_dict["id"] = icu_id
        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["timestamps"] = timestamps
        return_dict["types"] = types
        return_dict["codes"] = codes
        return_dict["mortality"] = mortality
        return_dict["readmission"] = readmission

        return return_dict


def data_prepare(args):
    train_dataset = eICUDataset(split='train', processed_data_path=args.eicu_path)
    val_dataset = eICUDataset(split='val', processed_data_path=args.eicu_path)
    test_dataset = eICUDataset(split='test', processed_data_path=args.eicu_path)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs_eicu, collate_fn=collate_fn)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.tokenizer


if __name__ == "__main__":

    import argparse
    from tqdm import tqdm
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pdb
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Alignment text and ts data")
        parser.add_argument(
            "--eicu_path", type=str, default="./datasets/eicu/processed", help="A path to dataset folder"
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=8,
            help="Batch size for the training dataloader.",
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=32,
            help="Batch size for the evaluation dataloader.",
        )
        args = parser.parse_args()
        return args
    
    args = parse_args()
    train, valid, test = data_prepare(args=args)
    for i, batch in enumerate(tqdm(train)):
        y = batch
        pdb.set_trace()
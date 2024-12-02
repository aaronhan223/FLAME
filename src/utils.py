import csv
import json
import logging
import os
import dill as pickle
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_csv(filename):
    logging.info(f"Reading from {filename}")
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.DictReader(file, delimiter=",")
        for row in csv_reader:
            data.append(row)
    header = list(data[0].keys())
    return header, data


def read_txt(filename):
    logging.info(f"Reading from {filename}")
    data = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            data.append(line)
    return data


def write_txt(filename, data):
    logging.info(f"Writing to {filename}")
    with open(filename, "w") as file:
        for line in data:
            file.write(line + "\n")
    return


def read_json(filename):
    logging.info(f"Reading from {filename}")
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def write_json(filename, data):
    logging.info(f"Writing to {filename}")
    with open(filename, "w") as file:
        json.dump(data, file)
    return


def create_directory(directory):
    if not os.path.exists(directory):
        logging.info(f"Creating directory {directory}")
        os.makedirs(directory)


def load_pickle(filename):
    logging.info(f"Data loaded from {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filename):
    logging.info(f"Data saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_events(events):
    """
    in: [(t0, _, i0), (t0, _, i1), (t1, _, i2), (t2, _, i3)]
    out: [(t0, _, [i0, i1]), (t1, _, [i2]), (t2, _, [i3])]
    """
    assert len(set([e[1] for e in events])) <= 1
    events = sorted(events, key=lambda x: x[0])
    out = []
    for timestamp, _type, _id in events:
        if len(out) == 0 or out[-1][0] != timestamp:
            out.append((timestamp, _type, [_id]))
        else:
            out[-1][-1].append(_id)
    return out

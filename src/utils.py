import csv
import json
import logging
import os
import dill as pickle
import random

import numpy as np
import torch


# Maps each base task name in --task to the args attribute that holds its
# dash-separated modality string (e.g. 'ihm' -> args.ihm_mod = 'TS-Text').
TASK_TO_MOD_ARG = {
    'ihm': 'ihm_mod',
    'los': 'los_mod',
    'pheno': 'pheno_mod',
    'mortality': 'mor_mod',
    'readmission': 'rad_mod',
    'birads': 'birads_mod',
    'risk': 'risk_mod',
    'density': 'density_mod',
    'diag': 'diag_mod',
}


def mods_for_task(args, task=None):
    """Return the underscore-joined per-task modality string for ``task``.

    Replaces the old hand-listed ``task_mods_dict`` lookups: for
    ``args.task = 'ihm-diag'`` this yields ``f"{args.ihm_mod}_{args.diag_mod}"``.

    Order of base tasks in ``args.task`` is preserved (so file paths from
    older runs that used a particular order still resolve identically as long
    as the user keeps that order).
    """
    task_str = task if task is not None else args.task
    parts = []
    for t in task_str.split('-'):
        if t not in TASK_TO_MOD_ARG:
            raise KeyError(
                f"Unknown task '{t}' in args.task='{task_str}'. "
                f"Known: {sorted(TASK_TO_MOD_ARG)}"
            )
        parts.append(getattr(args, TASK_TO_MOD_ARG[t]))
    return '_'.join(parts)


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

def check_encoder_updates(encoder, step_name=""):
    """
    Check if encoder parameters are being updated during training.
    Call this function before and after optimizer.step() to compare.
    
    Args:
        encoder: dict of encoder modules (one per task)
        step_name: str to identify when this check is called
    
    Returns:
        dict: snapshot of encoder parameters for comparison
    """
    encoder_snapshots = {}
    
    for task_name, enc in encoder.items():
        params_snapshot = {}
        has_grad = {}
        
        for name, param in enc.named_parameters():
            # Store a copy of the parameter
            params_snapshot[name] = param.data.clone().detach()
            # Check if gradient exists
            has_grad[name] = param.grad is not None
            
        encoder_snapshots[task_name] = {
            'params': params_snapshot,
            'has_grad': has_grad
        }
    
    if step_name:
        print(f"\n=== Encoder Check: {step_name} ===")
        for task_name, snapshot in encoder_snapshots.items():
            n_params = len(snapshot['params'])
            n_with_grad = sum(snapshot['has_grad'].values())
            print(f"Task '{task_name}': {n_params} params, {n_with_grad} with gradients")
    
    return encoder_snapshots

def compare_encoder_snapshots(before, after, tolerance=1e-10):
    """
    Compare two encoder snapshots to check if parameters changed.
    
    Args:
        before: snapshot from check_encoder_updates() before update
        after: snapshot from check_encoder_updates() after update
        tolerance: minimum difference to consider as change
    
    Returns:
        dict: summary of changes per task
    """
    changes = {}
    
    for task_name in before.keys():
        changed_params = []
        unchanged_params = []
        
        for param_name, param_before in before[task_name]['params'].items():
            param_after = after[task_name]['params'][param_name]
            diff = torch.abs(param_after - param_before).max().item()
            
            if diff > tolerance:
                changed_params.append((param_name, diff))
            else:
                unchanged_params.append(param_name)
        
        changes[task_name] = {
            'changed': changed_params,
            'unchanged': unchanged_params,
            'n_changed': len(changed_params),
            'n_unchanged': len(unchanged_params)
        }
    
    print("\n=== Encoder Parameter Update Summary ===")
    for task_name, change_info in changes.items():
        total = change_info['n_changed'] + change_info['n_unchanged']
        print(f"\nTask '{task_name}':")
        print(f"  Changed: {change_info['n_changed']}/{total} parameters")
        print(f"  Unchanged: {change_info['n_unchanged']}/{total} parameters")
        
        if change_info['changed']:
            print(f"  Sample changes (first 3):")
            for param_name, diff in change_info['changed'][:3]:
                print(f"    {param_name}: max diff = {diff:.6e}")
        else:
            print(f"  ⚠️  WARNING: No parameters were updated!")
    
    return changes

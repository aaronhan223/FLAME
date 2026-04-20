import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch


STATE_DICT_CANDIDATE_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "module",
)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_tensor_dict(obj) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    tensor_count = sum(torch.is_tensor(v) for v in obj.values())
    return tensor_count >= max(1, int(0.5 * len(obj)))


def _extract_state_dict(checkpoint_obj) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, torch.nn.Module):
        return checkpoint_obj.state_dict()

    if _is_tensor_dict(checkpoint_obj):
        return checkpoint_obj

    if isinstance(checkpoint_obj, dict):
        for key in STATE_DICT_CANDIDATE_KEYS:
            if key in checkpoint_obj:
                candidate = checkpoint_obj[key]
                if isinstance(candidate, torch.nn.Module):
                    return candidate.state_dict()
                if _is_tensor_dict(candidate):
                    return candidate

        for _, value in checkpoint_obj.items():
            if _is_tensor_dict(value):
                return value

    raise ValueError(
        "Unable to extract state_dict from checkpoint object. "
        "Expected a saved model, raw state_dict, or a dict containing one."
    )


def _collect_2d_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = OrderedDict()
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim != 2:
            continue
        if "weight" not in name.lower():
            continue
        out[name] = tensor.detach().float().cpu()
    return out


def _common_weight_names(weight_dicts: List[Dict[str, torch.Tensor]]) -> List[str]:
    if not weight_dicts:
        return []
    common = set(weight_dicts[0].keys())
    for d in weight_dicts[1:]:
        common &= set(d.keys())
    return sorted(common)


def _shape_consistent(
    tensors: List[torch.Tensor], concat_dim: int
) -> bool:
    if len(tensors) <= 1:
        return True
    ref_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        shape = list(tensor.shape)
        if len(shape) != len(ref_shape):
            return False
        for idx in range(len(shape)):
            if idx == concat_dim:
                continue
            if shape[idx] != ref_shape[idx]:
                return False
    return True


def _eigenvalues_from_concatenated_weights(
    weight_dicts: List[Dict[str, torch.Tensor]],
    common_names: Iterable[str],
    concat_dim: int,
) -> Dict[str, torch.Tensor]:
    eigvals = OrderedDict()

    for name in common_names:
        per_ckpt = [wd[name] for wd in weight_dicts]
        if not _shape_consistent(per_ckpt, concat_dim=concat_dim):
            continue

        concat_w = torch.cat(per_ckpt, dim=concat_dim)

        singular_vals = torch.linalg.svdvals(concat_w)
        vals = torch.square(singular_vals)
        eigvals[name] = vals

    return eigvals


def _plot_eigenvalue_curves(
    eigvals_per_layer: Dict[str, torch.Tensor],
    out_path: str,
    title: str,
    max_components: int,
    max_layers: int,
):
    if len(eigvals_per_layer) == 0:
        raise ValueError("No compatible 2D weight tensors found for plotting.")

    plt.figure(figsize=(12, 7))

    plotted = 0
    for layer_name, vals in eigvals_per_layer.items():
        if plotted >= max_layers:
            break
        k = min(max_components, vals.numel())
        x = torch.arange(1, k + 1).numpy()
        y = vals[:k].numpy()
        plt.plot(x, y, linewidth=1.5, label=layer_name)
        plotted += 1

    plt.yscale("log")
    plt.xlabel("Eigenvector index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if plotted <= 20:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_summary(
    output_dir: str,
    eig_in: Dict[str, torch.Tensor],
    eig_out: Dict[str, torch.Tensor],
    checkpoint_paths: List[str],
):
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Checkpoint eigen-analysis summary\n")
        file.write("===================================\n\n")
        file.write("Checkpoints used:\n")
        for path in checkpoint_paths:
            file.write(f"- {path}\n")
        file.write("\n")

        file.write(f"Layers plotted (concat along input dim): {len(eig_in)}\n")
        file.write(f"Layers plotted (concat along output dim): {len(eig_out)}\n\n")

        def _write_block(name: str, block: Dict[str, torch.Tensor]):
            file.write(f"{name}\n")
            file.write("-" * len(name) + "\n")
            for layer_name, vals in block.items():
                top = vals[0].item() if vals.numel() > 0 else float("nan")
                total = vals.sum().item() if vals.numel() > 0 else float("nan")
                file.write(
                    f"{layer_name}: n={vals.numel()}, top_eig={top:.6e}, trace={total:.6e}\n"
                )
            file.write("\n")

        _write_block("Input-dim concatenation", eig_in)
        _write_block("Output-dim concatenation", eig_out)



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot eigenspectra from multiple PyTorch checkpoints "
            "with the same architecture."
        )
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of paths to .pt checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save plots and summary.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=256,
        help="Maximum eigen components per layer to plot (default: 256).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=30,
        help="Maximum number of layers to plot (default: 30).",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    checkpoint_paths = [os.path.abspath(path) for path in args.checkpoints]
    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    os.makedirs(args.output_dir, exist_ok=True)

    weight_dicts = []
    for path in checkpoint_paths:
        checkpoint_obj = _torch_load_cpu(path)
        state_dict = _extract_state_dict(checkpoint_obj)
        weight_dicts.append(_collect_2d_weights(state_dict))

    common_names = _common_weight_names(weight_dicts)
    if len(common_names) == 0:
        raise ValueError("No shared 2D weight tensors found across checkpoints.")

    eigvals_input_concat = _eigenvalues_from_concatenated_weights(
        weight_dicts=weight_dicts,
        common_names=common_names,
        concat_dim=1,
    )
    eigvals_output_concat = _eigenvalues_from_concatenated_weights(
        weight_dicts=weight_dicts,
        common_names=common_names,
        concat_dim=0,
    )

    input_plot = os.path.join(args.output_dir, "eigs_concat_input_dim.png")
    output_plot = os.path.join(args.output_dir, "eigs_concat_output_dim.png")

    _plot_eigenvalue_curves(
        eigvals_per_layer=eigvals_input_concat,
        out_path=input_plot,
        title="Eigenvalue Spectrum per Layer (weights concatenated along input dim)",
        max_components=args.max_components,
        max_layers=args.max_layers,
    )
    _plot_eigenvalue_curves(
        eigvals_per_layer=eigvals_output_concat,
        out_path=output_plot,
        title="Eigenvalue Spectrum per Layer (weights concatenated along output dim)",
        max_components=args.max_components,
        max_layers=args.max_layers,
    )

    _write_summary(
        output_dir=args.output_dir,
        eig_in=eigvals_input_concat,
        eig_out=eigvals_output_concat,
        checkpoint_paths=checkpoint_paths,
    )

    print("Saved:")
    print(f"- {input_plot}")
    print(f"- {output_plot}")
    print(f"- {os.path.join(args.output_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()

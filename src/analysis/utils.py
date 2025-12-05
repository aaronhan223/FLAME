import torch
import torch.nn as nn
from src.analysis.evaluation import evaluate_model
import re

def normalize_layer_name(name):
    """Remove task-specific substrings from layer name for comparison."""
    task_patterns = ['_IHM', '_LOS', '_PHENO', '_RAD', '_MOR']
    normalized = name
    for pattern in task_patterns:
        normalized = normalized.replace(pattern, '')
    return normalized

def layerwise_svd(task_vector, rank=None, lora_only=False, log=False):
    svd_results = {}
    tv = {}
    if isinstance(task_vector, nn.Module):
        for name, param in task_vector.state_dict().items():
            tv[name] = param
    else:
        tv = {n:p for n, p in task_vector.items()}
    for name, delta in tv.items():
        if lora_only and "lora_" not in name:
            continue
        if delta.ndim < 2:  # skip biases, layernorm weights etc.
            continue
        
        # Flatten into matrix for SVD
        delta_2d = delta.detach().cpu()
        if delta_2d.ndim > 2:
            print(name, delta_2d.shape)
            delta_2d = delta_2d.view(delta_2d.size(0), -1)
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
        if rank is not None:
            U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
        
        svd_results[name] = (U, S, Vh)
        if log:
            print(f"{name}: shape={delta.shape}, top singular values={S[:5]}")
    
    return svd_results

def layerwise_rank_analysis(model, tol=1e-5):
    """
    Computes the numerical rank (low-rank structure) of each parameter tensor in the model.
    
    Args:
        model: torch.nn.Module
        tol: threshold for singular values to consider as non-zero (for numerical rank)
    
    Returns:
        dict mapping layer names -> rank info
    """
    rank_info = {}

    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Only analyze weight matrices (not biases, batchnorm params, etc.)
            W = param.detach().cpu()
            # Flatten convolutional kernels: (out_channels, in_channels, kH, kW) → (out_channels, -1)
            if W.dim() > 2:
                W = W.view(W.size(0), -1)
            
            # Compute singular values via SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            # Count significant singular values
            rank = torch.sum(S > tol).item()
            
            rank_info[name] = {
                "shape": tuple(param.shape),
                "rank": rank,
                "rank_ratio": rank / min(W.shape),  # normalized rank
            }

    return rank_info

def layerwise_concat_rank(model1, model2, tol=1e-5):
    """
    Computes numerical rank for concatenated parameters layer-by-layer
    between two models with the same architecture.
    """
    
    rank_info = {}
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        # Normalize names by removing task substrings
        norm_name1 = normalize_layer_name(name1)
        norm_name2 = normalize_layer_name(name2)
        
        assert norm_name1 == norm_name2, f"Layer mismatch: {name1} (normalized: {norm_name1}) vs {name2} (normalized: {norm_name2})"
        
        if p1.dim() >= 2:
            W1 = p1.detach().cpu()
            W2 = p2.detach().cpu()
            
            # Flatten convolutional kernels into matrices
            if W1.dim() > 2:
                W1 = W1.view(W1.size(0), -1)
                W2 = W2.view(W2.size(0), -1)
            
            # Concatenate along output dimension (rows)
            W_concat = torch.cat([W1, W2], dim=0)
            
            # Compute SVD
            S = torch.linalg.svdvals(W_concat)
            rank = torch.sum(S > tol).item()
            
            rank_info[name1] = {
                "shape": tuple(W1.shape),
                "concat_rank": rank,
                "concat_rank_ratio": rank / min(W_concat.shape),
            }

    return rank_info

def layerwise_concat_svd(model1, model2, tol=1e-5, select_rank=None, concat_dim=0):
    model1_params = {normalize_layer_name(n):p for n, p in model1.state_dict().items()}
    model2_params = {normalize_layer_name(n):p for n, p in model2.state_dict().items()}
    concat1_params = {}
    concat2_params = {}
    U_dict = {}
    S_dict = {}
    Vh_dict = {}
    for (name, p2) in model2_params.items():
        if name in model1_params:
            p1 = model1_params[name]
            if p1.dim()>=2 and p2.dim()>=2:
                W1 = p1.detach().cpu()
                W2 = p2.detach().cpu()
                if W1.dim() > 2 and W2.dim() > 2:
                    W1 = W1.view(W1.size(0), -1)
                    W2 = W2.view(W2.size(0), -1)
                W_concat = torch.cat([W1, W2], dim=concat_dim)
                U, S, Vh = torch.linalg.svd(W_concat, full_matrices=False)
                if select_rank is None:
                    rank = select_rank
                else:
                    rank = torch.sum(S > tol).item()
                if concat_dim == 0:
                    U_dict[name] = (U[:U.shape[0]//2, :rank], U[U.shape[0]//2:, :rank])
                    S_dict[name] = S[:rank]
                    Vh_dict[name] = Vh[:rank, :]
                    Wk1_concat = torch.matmul(U[:U.shape[0]//2, :rank], torch.diag(S[:rank])) @ Vh[:rank, :]
                    Wk2_concat = torch.matmul(U[U.shape[0]//2:, :rank], torch.diag(S[:rank])) @ Vh[:rank, :]
                else:
                    U_dict[name] = U[:, :rank]
                    S_dict[name] = S[:rank]
                    Vh_dict[name] = (Vh[:rank, :Vh.shape[1]//2], Vh[:rank, Vh.shape[1]//2:])
                    Wk1_concat = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :Vh.shape[1]//2]
                    Wk2_concat = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, Vh.shape[1]//2:]
                concat1_params[name] = Wk1_concat
                concat2_params[name] = Wk2_concat
        else:
            concat1_params[name] = p1
            concat2_params[name] = p2
    return concat1_params, concat2_params, U_dict, S_dict, Vh_dict

def copy_weights(model1, model2, start_copy_layer=None, end_copy_layer=None):
    # Build lookup table for model1 parameters
    if isinstance(model1, nn.Module):
        model1_params = {normalize_layer_name(n): p for n, p in model1.named_parameters()}
    else:
        model1_params = {normalize_layer_name(n): p for n, p in model1.items()}
    # Iterate through model2 parameters and copy if match found
    for i, (n2, p2) in enumerate(model2.named_parameters()):
        if start_copy_layer is not None and i < start_copy_layer:
            continue
        if end_copy_layer is not None and i >= end_copy_layer:
            break
        norm = normalize_layer_name(n2)
        if norm in model1_params:
            p2.data.copy_(model1_params[norm].data)

def register_forward_hook(model, layer_name):
    """
    Register a forward hook to capture the output of a specific layer.
    
    Args:
        model: torch.nn.Module - the model to hook into
        layer_name: str - the name of the layer to hook (from model.named_modules())
    
    Returns:
        tuple: (activations_dict, hook_handle)
            - activations_dict: dict that will store the layer output
            - hook_handle: handle to remove the hook later
    
    Example:
        activations, handle = register_forward_hook(model, 'layer1.conv1')
        output = model(input_data)
        layer_output = activations['layer1.conv1']
        handle.remove()  # Clean up when done
    """
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Find the layer by name
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {[n for n, _ in model.named_modules()]}")
    
    # Register the hook
    handle = target_module.register_forward_hook(get_activation(layer_name))
    
    return activations, handle

def create_composite_model(model_source, model_target,
                           cutoff_layer_source,   # list of layers (3 modalities)
                           start_layer_target):   # list of target layers (3 modalities)
    """
    cutoff_layer_source:  ["modality_layers.TS_IHM.0.2",
                           "modality_layers.Text_IHM.0.2",
                           "modality_layers.CXR_IHM.0.2"]

    start_layer_target:   ["cross_layers.0.0",
                           "cross_layers.0.0",
                           "cross_layers.0.0"]
    (or modality-specific cross layers)
    """

    # Will store activations from source, index by position in cutoff list
    source_activations = [None] * len(cutoff_layer_source)

    # -------------------------------- SOURCE HOOKS --------------------------------
    def make_source_hook(i):
        def hook(module, input, output):
            source_activations[i] = output.detach()   # (B, N, D)
        return hook

    # -------------------------------- TARGET PRE-HOOKS ----------------------------
    # Replace ONLY context argument of cross-attn block
    def make_target_pre_hook(i):
        def pre_hook(module, input, kwargs):
            ctx = source_activations[i]
            if ctx is None:
                return input, kwargs
            
            # Replace the context kwarg
            new_kwargs = dict(kwargs)
            new_kwargs["context"] = ctx
            
            return input, new_kwargs
        return pre_hook

    # -------------------------------- REGISTER HOOKS ------------------------------
    source_handles = []
    target_handles = []

    for idx, layer_name in enumerate(cutoff_layer_source):
        if layer_name not in dict(model_source.named_modules()):
            raise ValueError(f"Source layer '{layer_name}' not found.")
        module = dict(model_source.named_modules())[layer_name]
        source_handles.append(module.register_forward_hook(make_source_hook(idx)))

    for idx, layer_name in enumerate(start_layer_target):
        if layer_name not in dict(model_target.named_modules()):
            raise ValueError(f"Target layer '{layer_name}' not found.")
        module = dict(model_target.named_modules())[layer_name]
        target_handles.append(module.register_forward_pre_hook(make_target_pre_hook(idx)))

    # -------------------------------- COMPOSITE FORWARD ---------------------------
    def composite_forward(input_data):
        # reset activations
        for i in range(len(source_activations)):
            source_activations[i] = None

        # rename LOS→IHM for source model
        renamed = {k.replace("LOS", "IHM"): v for k, v in input_data.items()}

        # run source model → collects latent for all modalities
        with torch.no_grad():
            _ = model_source(renamed)

        # run target model → pre-hooks inject latent for each modality
        return model_target(input_data)

    # -------------------------------- CLEANUP -------------------------------------
    def cleanup():
        for h in source_handles: h.remove()
        for h in target_handles: h.remove()

    return composite_forward, cleanup



def evaluate_composite_model(args, model_source, model_target, encoder_source, encoder_target, 
                             cutoff_layer_source, start_layer_target, device='cuda'):
    """
    Evaluate a composite model that combines early layers from source and late layers from target.
    
    Args:
        args: Arguments containing evaluation settings
        model_source: Source model (e.g., IHM)
        model_target: Target model (e.g., LOS)
        encoder_source: Encoder for source model
        encoder_target: Encoder for target model
        cutoff_layer_source: Where to extract from source
        start_layer_target: Where to inject into target
        device: Device to run on
    
    Returns:
        Evaluation metrics
    """
    
    composite_forward, cleanup = create_composite_model(
        model_source, model_target, 
        cutoff_layer_source, start_layer_target
    )
    
    try:
        # Use the composite forward function for evaluation
        # You'll need to modify evaluate_model to accept a custom forward function
        results = evaluate_model(
            args, 
            model=model_target,  # Use target model structure
            encoder=encoder_target, 
            device=device,
            custom_forward=composite_forward  # Pass custom forward
        )
        return results
    finally:
        cleanup()

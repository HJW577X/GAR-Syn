import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import METHOD_NAME
from .data import FeatureStore, SynergyDataset, prediction_frame
from .metrics import regression_metrics
from .model import GARSynNet


def set_seed(seed, cudnn_benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark


def configure_cuda_runtime(model_config, device):
    if device.type != "cuda":
        return
    allow_tf32 = bool(model_config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except AttributeError:
        pass


def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def move_inputs(inputs, labels, device):
    if inputs.device != device:
        inputs = inputs.to(device, non_blocking=True)
    labels = labels.reshape(-1, 1)
    if labels.device != device:
        labels = labels.to(device, non_blocking=True)
    return inputs, labels.float()


def make_loader(data_dir, fold, split, label, batch_size, shuffle, workers, drop_last=False):
    dataset = SynergyDataset(data_dir, fold, split, label)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


class GPUTensorLoader:
    def __init__(self, inputs, labels, batch_size, shuffle, drop_last=False):
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = labels.shape[0]
        self.device = labels.device

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, device=self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        stop = self.n
        if self.drop_last:
            stop = (self.n // self.batch_size) * self.batch_size
        for start in range(0, stop, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield self.inputs.index_select(0, idx), self.labels.index_select(0, idx)

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size


def make_gpu_loader(data_dir, fold, split, label, batch_size, shuffle, device, drop_last=False):
    dataset = SynergyDataset(data_dir, fold, split, label)
    inputs = dataset.inputs.to(device, non_blocking=True)
    labels = dataset.labels.to(device, non_blocking=True)
    return GPUTensorLoader(inputs, labels, batch_size, shuffle, drop_last=drop_last)


def garsyn_loss_batch(
    model,
    reverse_drug_map,
    inputs,
    labels,
    criterion,
    lambda_perm,
    lambda_gate_sparse,
    lambda_gate_balance,
    use_gate_regularization,
    batch_idx=0,
    perm_every=1,
    save_aux_during_train=True,
):
    need_perm = lambda_perm > 0 and perm_every is not None and perm_every > 0 and batch_idx % perm_every == 0
    need_aux = bool(use_gate_regularization and save_aux_during_train)

    if need_perm:
        pred, pred_swap, aux = model.forward_with_perm(reverse_drug_map, inputs, return_aux=need_aux)
    else:
        if need_aux:
            pred, aux = model(reverse_drug_map, inputs, return_aux=True)
        else:
            pred = model(reverse_drug_map, inputs, return_aux=False)
            aux = None
        pred_swap = None

    loss_syn = criterion(pred.float(), labels)
    loss_perm = criterion(pred.float(), pred_swap.float()) if need_perm else pred.new_tensor(0.0)
    loss_gate_sparse = pred.new_tensor(0.0)
    loss_gate_balance = pred.new_tensor(0.0)

    if need_aux and aux is not None and aux.get("gate"):
        gates = torch.cat([g.reshape(g.shape[0], -1) for g in aux["gate"]], dim=1)
        loss_gate_sparse = gates.mean()
        last_gate = aux["gate"][-1]
        token_score = last_gate.mean(dim=(2, 3))
        group_score = torch.stack(
            [
                token_score[:, 1:4].mean(dim=1),
                token_score[:, 4:7].mean(dim=1),
                token_score[:, 7:10].mean(dim=1),
            ],
            dim=1,
        )
        group_prob = group_score / (group_score.sum(dim=1, keepdim=True) + 1e-8)
        target = torch.full_like(group_prob, 1.0 / 3.0)
        loss_gate_balance = F.mse_loss(group_prob, target)

    loss = (
        loss_syn
        + lambda_perm * loss_perm
        + lambda_gate_sparse * loss_gate_sparse
        + lambda_gate_balance * loss_gate_balance
    )
    return loss, pred


def predict(model, device, reverse_drug_map, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Predict", leave=False, dynamic_ncols=True):
            inputs, labels = move_inputs(inputs, labels, device)
            pred = model(reverse_drug_map, inputs, return_aux=False)
            y_true.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
            y_pred.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def train_fold(args, model_config, label, fold):
    set_seed(args.seed, cudnn_benchmark=model_config.get("enable_cudnn_benchmark", True))
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    configure_cuda_runtime(model_config, device)
    features = FeatureStore(args.data_dir)
    model = GARSynNet(model_config, features).to(device)
    gpu_cache = model_config.get("gpu_cache", "none")
    graph_cache_mode = model_config.get("graph_cache_mode", "batch")
    if gpu_cache in {"features", "all"} and device.type == "cuda":
        # Dense drug/cell features are registered buffers and have already moved
        # with model.to(device). Graph objects are cached on GPU here, but graph
        # embeddings are not precomputed because the GNN encoder is trainable.
        model.build_idx_to_graph(features.reverse_drug_map)
        print(f"[GAR-Syn] cached dense feature buffers and drug graphs on GPU ({gpu_cache})")
    if graph_cache_mode == "full_batch" and device.type == "cuda":
        model.build_all_graph_batch(features.reverse_drug_map)
        print("[GAR-Syn] cached one full PyG drug-graph batch on GPU (graph_cache_mode=full_batch)")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["lr"],
        betas=(0.95, 0.999),
        amsgrad=True,
        weight_decay=model_config.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=5)
    criterion = torch.nn.MSELoss()
    amp_enabled = bool(model_config.get("use_amp", True) and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    if gpu_cache in {"indices", "features", "all"} and device.type == "cuda":
        print(f"[GAR-Syn] caching split inputs/labels on GPU ({gpu_cache})")
        train_loader = make_gpu_loader(
            args.data_dir,
            fold,
            "train",
            label,
            model_config["batch_size"],
            True,
            device,
            drop_last=True,
        )
        val_loader = make_gpu_loader(args.data_dir, fold, "val", label, model_config["batch_size"], False, device)
        test_loader = make_gpu_loader(args.data_dir, fold, "test", label, model_config["batch_size"], False, device)
    else:
        train_loader = make_loader(
            args.data_dir,
            fold,
            "train",
            label,
            model_config["batch_size"],
            True,
            model_config["num_workers"],
            drop_last=True,
        )
        val_loader = make_loader(args.data_dir, fold, "val", label, model_config["batch_size"], False, model_config["num_workers"])
        test_loader = make_loader(args.data_dir, fold, "test", label, model_config["batch_size"], False, model_config["num_workers"])

    save_dir = Path(args.results_dir) / METHOD_NAME / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"repeat{fold}_best.pth"
    best_state = None
    best_pearson = -float("inf")
    best_epoch = 0
    wait = 0
    started = time.time()

    print(f"[GAR-Syn] fold={fold} label={label} device={device}")
    for epoch in range(1, model_config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False, dynamic_ncols=True)
        for batch_idx, (inputs, labels_tensor) in enumerate(pbar):
            inputs, labels_tensor = move_inputs(inputs, labels_tensor, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                loss, _ = garsyn_loss_batch(
                    model,
                    features.reverse_drug_map,
                    inputs,
                    labels_tensor,
                    criterion,
                    model_config.get("lambda_perm", 0.1),
                    model_config.get("lambda_gate_sparse", 1e-4),
                    model_config.get("lambda_gate_balance", 1e-3),
                    model_config.get("use_gate_regularization", True),
                    batch_idx=batch_idx,
                    perm_every=model_config.get("perm_every", 4),
                    save_aux_during_train=model_config.get("save_aux_during_train", True),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            count += 1
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        y_val, p_val = predict(model, device, features.reverse_drug_map, val_loader)
        val_metrics = regression_metrics(y_val, p_val)
        scheduler.step(val_metrics["pearson"])
        train_loss = total_loss / max(count, 1)
        print(
            f"[GAR-Syn] fold={fold} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_pearson={val_metrics['pearson']:.4f}"
        )

        if val_metrics["pearson"] > best_pearson + model_config.get("min_delta", 0.001):
            best_pearson = val_metrics["pearson"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "best_pearson": best_pearson, "model": best_state}, checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= model_config.get("early_stop_patience", 20):
                print(f"[GAR-Syn] early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_test, p_test = predict(model, device, features.reverse_drug_map, test_loader)
    metrics = regression_metrics(y_test, p_test)
    metrics.update({"best_epoch": best_epoch, "seconds": time.time() - started})
    return p_test, metrics


def run_experiment(args, model_config, labels, folds):
    result_root = Path(args.results_dir) / METHOD_NAME
    pred_dir = result_root / "predict"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    frames = {fold: prediction_frame(args.data_dir, fold) for fold in folds}

    for label in labels:
        for fold in folds:
            pred, metrics = train_fold(args, model_config, label, fold)
            frames[fold][f"pred_{label}"] = pred
            frames[fold].to_csv(pred_dir / f"repeat{fold}_predict.csv", index=False)
            rows.append({"method": METHOD_NAME, "fold": fold, "label": label, **metrics})
            pd.DataFrame(rows).to_csv(result_root / "metrics_running.csv", index=False)

    with open(result_root / "manifest.json", "w") as f:
        json.dump({"method": METHOD_NAME, "labels": labels, "folds": folds, "model_config": model_config}, f, indent=2)
    pd.DataFrame(rows).to_csv(result_root / "metrics.csv", index=False)
    return rows

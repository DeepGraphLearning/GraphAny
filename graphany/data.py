import logging
import os
import os.path
import os.path as osp
import re
import ssl
import sys
import urllib

import dgl
import dgl.function as fn
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._utils import (
    _binary_search_perplexity as sklearn_binary_search_perplexity,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from graphany.utils import logger, timer


def get_entropy_normed_cond_gaussian_prob(X, entropy, metric="euclidean"):
    """
    Parameters
    ----------
    X:              The matrix for pairwise similarity
    entropy:     Perplexity of the conditional prob distribution
    Returns the entropy-normalized conditional gaussian probability based on distances.
    -------
    """

    # Compute pairwise distances
    perplexity = np.exp2(entropy)
    distances = pdist(X, metric=metric)
    distances = squareform(distances)

    # Compute the squared distances
    distances **= 2
    distances = distances.astype(np.float32)
    return sklearn_binary_search_perplexity(distances, perplexity, verbose=0)


def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    sampled_indices = [
        label_indices[torch.randperm(len(label_indices))[:k]]
        for label_indices in ref_node_idx
    ]
    return visible_nodes[torch.cat(sampled_indices)]


def get_data_split_masks(n_nodes, labels, num_train_nodes, label_idx=None, seed=42):
    label_idx = np.arange(n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=labels[test_and_valid_idx],
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def download_url(url: str, folder: str, log: bool = True, filename=None):
    r"""Modified from torch_geometric.data.download_url

    Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def load_heterophilous_dataset(url, raw_dir):
    # Wrap Heterophilous to DGL Graph Dataset format https://arxiv.org/pdf/2302.11640.pdf
    download_path = download_url(url, raw_dir)
    data = np.load(download_path)
    node_features = torch.tensor(data["node_features"])
    labels = torch.tensor(data["node_labels"])
    edges = torch.tensor(data["edges"])

    graph = dgl.graph(
        (edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int
    )
    num_classes = len(labels.unique())
    num_targets = 1 if num_classes == 2 else num_classes
    if num_targets == 1:
        labels = labels.float()
    train_masks = torch.tensor(data["train_masks"]).T
    val_masks = torch.tensor(data["val_masks"]).T
    test_masks = torch.tensor(data["test_masks"]).T

    return graph, labels, num_classes, node_features, train_masks, val_masks, test_masks


class CombinedDataset(pl.LightningDataModule):
    def __init__(self, train_ds_dict, eval_ds_dict, cfg):
        super().__init__()
        self.train_ds_dict = train_ds_dict
        self.eval_ds_dict = eval_ds_dict
        self.all_ds = list(self.train_ds_dict.values()) + list(
            self.eval_ds_dict.values()
        )
        self.cfg = cfg

    def to(self, device):
        for ds in self.all_ds:
            ds.to(device)

    def train_dataloader(self):
        sub_dataloaders = {
            name: ds.train_dataloader() for name, ds in self.train_ds_dict.items()
        }
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "min_size")

    def val_dataloader(self):
        sub_dataloaders = {
            name: ds.val_dataloader() for name, ds in self.eval_ds_dict.items()
        }
        # Use max_size instead of max_size_cycle to avoid repeated evaluation on small datasets
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "max_size")

    def test_dataloader(self):
        sub_dataloaders = {
            name: ds.test_dataloader() for name, ds in self.eval_ds_dict.items()
        }
        # Use max_size instead of max_size_cycle to avoid repeated evaluation on small datasets
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "max_size")


class GraphDataset(pl.LightningDataModule):
    def __init__(
            self,
            cfg,
            ds_name,
            cache_dir,
            train_batch_size=256,
            val_test_batch_size=256,
            n_hops=1,
            preprocess_device=torch.device("cpu"),
            permute_label=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = ds_name
        self.train_batch_size = train_batch_size
        self.permute_label = permute_label  # For checking label equivariance
        self.val_test_batch_size = val_test_batch_size
        self.preprocess_device = preprocess_device

        self.n_hops = n_hops

        self.data_source, ds_alias = cfg["_ds_meta_data"][ds_name].split(", ")
        self.gidtype = None
        self.dist = None
        self.unmasked_pred = None
        if self.data_source == "pyg":
            components = ds_alias.split(".")
            ds_init_args = {
                "_target_": f"torch_geometric.datasets.{ds_alias}",
                "root": f"{cfg.dirs.data_storage}{self.data_source}/{ds_alias}/",
            }
            if len(components) == 2:  # If sub-dataset
                ds_init_args["_target_"] = f"torch_geometric.datasets.{components[0]}"
                ds_init_args["name"] = components[1]
        elif self.data_source == "dgl":
            ds_init_args = {
                "_target_": f"dgl.data.{ds_alias}",
                "raw_dir": f"{cfg.dirs.data_storage}{self.data_source}/",
            }
        elif self.data_source == "ogb":
            ds_init_args = {
                "_target_": f"ogb.nodeproppred.DglNodePropPredDataset",
                "root": f"{cfg.dirs.data_storage}{self.data_source}/",
                "name": ds_alias,
            }
        elif self.data_source == "heterophilous":
            target = "graphany.data.load_heterophilous_dataset"
            url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{ds_alias}.npz"
            ds_init_args = {
                "_target_": target,
                "raw_dir": f"{cfg.dirs.data_storage}{self.data_source}/",
                "url": url,
            }
        else:
            raise NotImplementedError(f"Unsupported {self.data_source=}")
        self.data_init_args = OmegaConf.create(ds_init_args)
        # self.cache_f_name = osp.join(
        #     cache_dir, f'{self.name}_{n_hops}')
        if cfg.get("feat_chn"):
            all_channels = "+".join([cfg.feat_chn, cfg.pred_chn])
            all_hops = re.findall(r"\d+", all_channels)
            n_hops = max(max([int(_) for _ in all_hops]), n_hops)

        self.split_index = 0
        (
            self.g,
            self.label,
            self.feat,
            self.train_mask,
            self.val_mask,
            self.test_mask,
            self.num_class,
        ) = self.load_dataset(self.data_init_args)
        self.n_nodes, self.n_edges = self.g.num_nodes(), self.g.num_edges()
        self.cache_f_name = osp.join(
            cache_dir,
            f"{self.name}_{n_hops}hop_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split="
            f"{self.split_index}.pt",
        )

        self.dist_f_name = osp.join(
            cache_dir,
            f"{self.name}_{n_hops}hop_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split="
            f"{self.split_index}_{cfg.feat_chn}_entropy={cfg.entropy}_dist.pt",
        )

        self.gidtype = self.g.idtype
        self.train_indices = self.train_mask.nonzero().view(-1)

        (
            self.features,
            self.unmasked_pred,
            self.dist,
        ) = self.prepare_prop_features_logits_and_dist_features(
            self.g, self.feat, n_hops=cfg.n_hops
        )
        # Remove the graph, as GraphAny doesn't use it in training
        del self.g
        del self.feat
        torch.cuda.empty_cache()

    def to(self, device):  # Supports nested dictionary
        def to_device(input):
            if input is None:
                return None
            elif isinstance(input, dict):
                return {key: to_device(value) for key, value in input.items()}
            elif isinstance(input, list):
                return [to_device(item) for item in input]
            elif hasattr(input, "to"):
                return input.to(device)
            else:
                return (
                    input  # Return as is if it's not a tensor or any nested structure
                )

        # Apply to_device to all attributes that may contain tensors
        attrs = [
            "label",
            "feat",
            "train_mask",
            "val_mask",
            "test_mask",
            "train_indices",
            "unmasked_pred",
        ]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, to_device(getattr(self, attr)))

    def load_dataset(self, data_init_args):
        dataset = instantiate(data_init_args)

        if self.data_source == "ogb":
            split_idx = dataset.get_idx_split()
            train_indices, valid_indices, test_indices = (
                split_idx["train"],
                split_idx["valid"],
                split_idx["test"],
            )
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
            g, label = dataset[0]
            label = label.view(-1)

            def to_mask(indices):
                mask = torch.BoolTensor(g.number_of_nodes()).fill_(False)
                mask[indices] = 1
                return mask

            train_mask, val_mask, test_mask = map(
                to_mask, (train_indices, valid_indices, test_indices)
            )

            num_class = label.max().item() + 1

            feat = g.ndata["feat"]
        elif self.data_source == "heterophilous":
            g, label, num_class, feat, train_mask, val_mask, test_mask = dataset
        elif self.data_source == "dgl":
            g = dataset[0]
            num_class = dataset.num_classes

            # get node feature
            feat = g.ndata["feat"]

            # get data split
            train_mask = g.ndata["train_mask"]
            val_mask = g.ndata["val_mask"]
            test_mask = g.ndata["test_mask"]

            label = g.ndata["label"]
        elif self.data_source == "pyg":
            g = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]))
            n_nodes = dataset.x.shape[0]
            num_class = dataset.num_classes
            # get node feature
            feat = dataset.x
            label = dataset.y

            if (
                    hasattr(dataset, "train_mask")
                    and hasattr(dataset, "val_mask")
                    and hasattr(dataset, "test_mask")
            ):
                train_mask, val_mask, test_mask = (
                    dataset.train_mask,
                    dataset.val_mask,
                    dataset.test_mask,
                )
            else:
                if label.ndim > 1:
                    raise NotImplementedError(
                        "Multi-Label classification currently unsupported."
                    )
                logging.warning(
                    f"No dataset split found for {self.name}, splitting with semi-supervised settings!!"
                )
                train_mask, val_mask, test_mask = get_data_split_masks(
                    n_nodes, label, 20 * num_class, seed=self.cfg.seed
                )

                self.split_index = self.cfg.seed
        else:
            raise NotImplementedError(f"Unsupported {self.data_source=}")
        if train_mask.ndim == 1:
            pass  # only one train/val/test split
        elif train_mask.ndim == 2:
            # ! Multiple splits
            # Modified: Use the ${seed} split if not specified!
            split_index = self.data_init_args.get("split", self.cfg.seed)
            # Avoid invalid split index
            self.split_index = split_index = (split_index % train_mask.ndim)
            train_mask = train_mask[:, split_index].squeeze()
            val_mask = val_mask[:, split_index].squeeze()
            if test_mask.ndim == 2:
                test_mask = test_mask[:, split_index].squeeze()
        else:
            raise ValueError("train/val/test masks have more than 2 dimensions")
        print(
            f"{self.name} {g.num_nodes()} {g.num_edges()} {feat.shape[1]} {num_class} {len(train_mask.nonzero())}"
        )

        if self.cfg.add_self_loop:
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)
        if self.cfg.to_bidirected:
            g = dgl.to_bidirected(g)
        g = dgl.to_simple(g)  # Remove duplicate edges.
        return g, label, feat, train_mask, val_mask, test_mask, num_class

    def compute_linear_gnn_logits(
            self, features, n_per_label_examples, visible_nodes, bootstrap=False
    ):
        # Compute and save LinearGNN logits into a dict. Note the computation is on CPU as torch does not support
        # the gelss driver on GPU currently.
        preds = {}
        label, num_class, device = self.label, self.num_class, torch.device("cpu")
        label = label.to(device)
        visible_nodes = visible_nodes.to(device)
        for channel, F in features.items():
            F = F.to(device)
            if bootstrap:
                ref_nodes = sample_k_nodes_per_label(
                    label, visible_nodes, n_per_label_examples, num_class
                )
            else:
                ref_nodes = visible_nodes
            Y_L = torch.nn.functional.one_hot(label[ref_nodes], num_class).float()
            with timer(
                    f"Solving with CPU driver (N={len(ref_nodes)}, d={F.shape[1]}, k={num_class})",
                    logger.debug,
            ):
                W = torch.linalg.lstsq(
                    F[ref_nodes.cpu()].cpu(), Y_L.cpu(), driver="gelss"
                )[0]
            preds[channel] = F @ W

        return preds

    def compute_channel_logits(self, features, visible_nodes, sample, device):
        pred_logits = self.compute_linear_gnn_logits(
            {
                c: features[c]
                for c in set(self.cfg.feat_channels + self.cfg.pred_channels)
            },
            self.cfg.n_per_label_examples,
            visible_nodes,
            bootstrap=sample,
        )
        return {c: logits.to(device) for c, logits in pred_logits.items()}

    def prepare_prop_features_logits_and_dist_features(self, g, input_feats, n_hops):
        # Calculate Low-pass features containing AX, A^2X and High-pass features
        # (I-A)X, and (I-A)^2X
        if not os.path.exists(self.cache_f_name):
            g = g.to(self.preprocess_device)
            with timer(
                    f"Computing {self.name} message passing and normalized predictions to file {self.cache_f_name}",
                    logger.info,
            ):
                dim = input_feats.size(1)
                LP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(
                    self.preprocess_device
                )
                HP = torch.zeros(n_hops, g.number_of_nodes(), dim).to(
                    self.preprocess_device
                )

                g.ndata["LP"] = input_feats.to(self.preprocess_device)
                g.ndata["HP"] = input_feats.to(self.preprocess_device)
                for hop_idx in range(n_hops):
                    # D^-1 A filter
                    g.update_all(fn.copy_u("LP", "temp"), fn.mean("temp", "LP"))

                    # (I - D^-1A) filter
                    g.update_all(fn.copy_u("HP", "temp"), fn.mean("temp", "HP_out"))
                    g.ndata["HP"] = g.ndata["HP"] - g.ndata["HP_out"]

                    LP[hop_idx] = g.ndata["LP"].clone()
                    HP[hop_idx] = g.ndata["HP"].clone()
                lp_feat_dict = {f"L{l + 1}": x for l, x in enumerate(LP)}
                hp_feat_dict = {f"H{l + 1}": x for l, x in enumerate(HP)}

                features = {"X": input_feats, **lp_feat_dict, **hp_feat_dict}
                unmasked_pred = self.compute_channel_logits(
                    features,
                    self.train_indices,
                    sample=False,
                    device=self.preprocess_device,
                )
                torch.save((features, unmasked_pred), self.cache_f_name)
        else:
            features, unmasked_pred = torch.load(self.cache_f_name, map_location="cpu")
        if not os.path.exists(self.dist_f_name):
            with timer(
                    f"Computing {self.name} conditional gaussian distances "
                    f"and save to {self.dist_f_name}",
                    logger.info,
            ):
                # y_feat: n_nodes, n_channels, n_labels
                y_feat = np.stack(
                    [unmasked_pred[c].cpu().numpy() for c in self.cfg.feat_channels],
                    axis=1,
                )
                # Conditional gaussian probability
                bsz, n_channel, n_class = y_feat.shape
                dist_feat_dim = n_channel * (n_channel - 1)
                # Conditional gaussian probability
                cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
                for i in range(bsz):
                    cond_gaussian_prob[i, :, :] = get_entropy_normed_cond_gaussian_prob(
                        y_feat[i, :, :], self.cfg.entropy
                    )
                dist = np.zeros((bsz, dist_feat_dim), dtype=np.float32)

                # Compute pairwise distances between channels n_channels(n_channels-1)/2 total features
                pair_index = 0
                for c in range(n_channel):
                    for c_prime in range(n_channel):
                        if c != c_prime:  # Diagonal distances are useless
                            dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                            pair_index += 1

                dist = torch.from_numpy(dist)
                torch.save(dist, self.dist_f_name)
        else:
            dist = torch.load(self.dist_f_name, map_location="cpu")
        return features, unmasked_pred, dist

    def train_dataloader(self):
        return DataLoader(
            self.train_mask.nonzero().view(-1),
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_mask.nonzero().view(-1), batch_size=self.val_test_batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_mask.nonzero().view(-1), batch_size=self.val_test_batch_size
        )

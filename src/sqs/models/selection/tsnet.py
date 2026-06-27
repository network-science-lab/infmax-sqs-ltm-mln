"""
Vendored TsNet model architecture from infmax-trainer-icm-mln.

Source: https://github.com/network-science-lab/infmax-trainer-icm-mln
Copied to avoid dependency conflict.
"""

from abc import abstractmethod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx import Proxy
from torch.nn import BatchNorm1d, Dropout, Linear, Module
from torch_geometric.nn import GATConv, GINConv, Sequential


# Base


class BaseHeteroModule(Module):
    """Abstract base for heterogeneous GNN modules."""

    @abstractmethod
    def forward(
        self,
        x_dict: Proxy,
        z_dict: Proxy,
        edge_index_dict: Proxy,
    ) -> dict[str, Tensor]:
        """Compute predictions for all node types."""


# Aggregation strategies


class MaxAggregation(torch.nn.Module):
    """Element-wise maximum over layer embeddings."""

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: max-pooled embedding tensor.
        """
        return torch.amax(torch.stack(list(h.values())), dim=0)


class MinAggregation(torch.nn.Module):
    """Element-wise minimum over layer embeddings."""

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: min-pooled embedding tensor.
        """
        return torch.amin(torch.stack(list(h.values())), dim=0)


class AvgAggregation(torch.nn.Module):
    """Element-wise mean over layer embeddings."""

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: mean-pooled embedding tensor.
        """
        return torch.mean(torch.stack(list(h.values())), dim=0)


class SumAggregation(torch.nn.Module):
    """Element-wise sum over layer embeddings."""

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: summed embedding tensor.
        """
        return torch.sum(torch.stack(list(h.values())), dim=0)


class LayerwiseAggregation(torch.nn.Module):
    """Trainable soft-attention aggregation over layer embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """
        :param hidden_channels: dimension of the per-layer embeddings.
        """
        super().__init__()
        self.attn = torch.nn.Linear(hidden_channels, 1)

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: attention-weighted embedding tensor.
        """
        stacked = torch.stack(list(h.values()))
        attn_scores = F.softmax(self.attn(stacked), dim=0)
        return (attn_scores * stacked).sum(dim=0)


class AttentionAggregation(torch.nn.Module):
    """Bilinear attention aggregation over layer embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """
        :param hidden_channels: dimension of the per-layer embeddings.
        """
        super().__init__()
        self.V = torch.nn.Bilinear(hidden_channels, hidden_channels, 1, bias=False)
        self.q = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all trainable parameters."""
        self.V.reset_parameters()
        self.q[0].reset_parameters()

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        :param h: per-layer embedding dict.
        :returns: bilinear-attention-weighted embedding tensor.
        """
        b = torch.cat(
            [self.V(self.q(h[et]), h[et]).tanh() for et in h],
            dim=-1,
        )
        alpha = b.softmax(dim=-1)
        embeddings = torch.stack([h[et] for et in h], dim=-1)
        return (alpha.unsqueeze(dim=1) * embeddings).sum(dim=-1)


# Main model

_ACTOR = "actor"

_AGGREGATORS = {
    "MaxAggregation": MaxAggregation,
    "MinAggregation": MinAggregation,
    "AvgAggregation": AvgAggregation,
    "SumAggregation": SumAggregation,
    "LayerwiseAggregation": LayerwiseAggregation,
    "AttentionAggregation": AttentionAggregation,
}


class TopSpreadersNetwork(BaseHeteroModule):
    """
    Top Spreaders Network (TsNet).

    Architecture:
    1. Encode each MLN layer independently with shared GAT+GIN blocks.
    2. Aggregate per-layer embeddings.
    3. Predict spreading potentials with a linear head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        aggregation_type: Literal[
            "MaxAggregation",
            "MinAggregation",
            "AvgAggregation",
            "SumAggregation",
            "LayerwiseAggregation",
            "AttentionAggregation",
        ] = "LayerwiseAggregation",
    ) -> None:
        """
        :param input_dim: number of input node features.
        :param hidden_channels: intermediate embedding width.
        :param output_dim: number of predicted spreading-potential targets.
        :param aggregation_type: name of the cross-layer aggregation class.
        """
        super().__init__()

        self.layerwise_encoder = Sequential(
            "x_actors, x_edges",
            [
                (GATConv(input_dim, hidden_channels // 4, heads=4),
                 "x_actors, x_edges -> x_interim"),
                (BatchNorm1d(hidden_channels), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (self._gin(hidden_channels, hidden_channels),
                 "x_interim, x_edges -> x_interim"),
                (BatchNorm1d(hidden_channels), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (GATConv(hidden_channels, hidden_channels // 8, heads=4),
                 "x_interim, x_edges -> x_interim"),
                (BatchNorm1d(hidden_channels // 2), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (self._gin(hidden_channels // 2, hidden_channels // 4),
                 "x_interim, x_edges -> x_interim"),
                (BatchNorm1d(hidden_channels // 4), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
            ],
        )

        aggregator_cls = _AGGREGATORS.get(aggregation_type)
        if aggregator_cls is None:
            raise ValueError(
                f"Unknown aggregation_type {aggregation_type!r}. "
                f"Available: {sorted(_AGGREGATORS)}"
            )
        if aggregation_type in ("LayerwiseAggregation", "AttentionAggregation"):
            self.layerwise_aggregator = aggregator_cls(hidden_channels // 4)
        else:
            self.layerwise_aggregator = aggregator_cls()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels // 4, hidden_channels // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_channels // 2, output_dim),
            torch.nn.Sigmoid(),
        )

    @staticmethod
    def _gin(in_channels: int, out_channels: int) -> GINConv:
        return GINConv(
            nn=torch.nn.Sequential(
                Linear(in_channels, out_channels),
                torch.nn.LeakyReLU(inplace=True),
                Linear(out_channels, out_channels),
            ),
            train_eps=True,
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        z_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        """
        Predict spreading potentials for all actors.

        :param x_dict: node features ``{"actor": Tensor[n_actors, input_dim]}``.
        :param z_dict: padding mask ``{"actor": Tensor[n_actors, n_layers]}``.
        :param edge_index_dict: per-layer edge indices
            ``{("actor", "l_<i>", "actor"): Tensor[2, n_edges]}``.

        :returns: ``{"actor": Tensor[n_actors, output_dim]}`` with predicted scores.
        """
        z_mask = 1 - z_dict[_ACTOR]
        y_relations: dict[str, Tensor] = {}
        for layer_idx, (layer_name, layer_edges) in enumerate(edge_index_dict.items()):
            layer_x = x_dict[_ACTOR] * z_mask[:, layer_idx].unsqueeze(1).expand_as(
                x_dict[_ACTOR]
            )
            y_relations[layer_name] = self.layerwise_encoder(layer_x, layer_edges)
        y_aggregated = self.layerwise_aggregator(y_relations)
        return {_ACTOR: self.head(y_aggregated)}

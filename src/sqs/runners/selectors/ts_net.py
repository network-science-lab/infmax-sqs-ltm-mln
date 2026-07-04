"""Seed selector backed by the vendored TsNet GNN model.

Loads a pre-trained ``TopSpreadersNetwork`` directly from a checkpoint directory
(``config.json`` + ``best.ckpt``) without requiring the ``infmax-trainer-icm-mln``
package.  Network topology is derived on-the-fly from the ``MultilayerNetwork``
passed to ``actorwise``, so no external dataset path is needed.

Example YAML configuration::

    - name: "ss_ts_net"
      class: "StaticSeedingStrategy"
      params:
        selector: "ts-net"
        mode: "SS"
        selector_params:
          model_dir: "/path/to/infmax-trainer-icm-mln/model/<run_id>"
          device: "cpu"
"""

import ast
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.mln.mlnetwork_torch import MultilayerNetworkTorch
from network_diffusion.seeding.base_selector import BaseSeedSelector

from sqs.models.selection.tsnet import TopSpreadersNetwork

_logger = logging.getLogger(__name__)


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that returns a harmless stub for any class that cannot be imported.

    PyTorch-Lightning checkpoints embed full class references (e.g.
    ``src.wrapper.HeteroGNNWrapper``) in ``hyper_parameters``.  We only need the
    ``state_dict`` tensors, so unknown classes are replaced with a no-op stub
    instead of raising ``ModuleNotFoundError``.
    """

    def find_class(self, module: str, name: str) -> type:
        """Return the real class if importable, otherwise a harmless stub."""
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            class _Stub:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def __setstate__(self, state: Any) -> None:
                    if isinstance(state, dict):
                        self.__dict__.update(state)

            _Stub.__qualname__ = f"_Stub[{module}.{name}]"
            return _Stub


class _StubPickle:
    """pickle-module substitute for ``torch.load`` that tolerates unknown classes."""

    UnpicklingError = pickle.UnpicklingError

    class Unpickler(_StubUnpickler):
        pass


class TsNetSeedSelector(BaseSeedSelector):
    """
    Seed selector that ranks actors using a pre-trained TsNet GNN model.

    The model is loaded once on the first call to ``actorwise`` and reused for
    subsequent calls.  No external dataset path is required -- topology is derived
    directly from the ``MultilayerNetwork`` instance at inference time.

    :param model_dir: directory containing ``config.json`` and ``best.ckpt``
        produced by ``infmax-trainer-icm-mln`` training.
    :param device: PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        """
        Initialise the selector.

        :param model_dir: path to the directory with ``config.json`` and ``best.ckpt``.
        :param device: PyTorch device string.
        """
        self._model_dir = Path(model_dir)
        self._device = device
        self._model: TopSpreadersNetwork | None = None
        self._saved_config: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Return a short description of this selector."""
        return f"TsNetSeedSelector(model_dir={self._model_dir}, device={self._device})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> tuple[TopSpreadersNetwork, dict[str, Any]]:
        """
        Load ``TopSpreadersNetwork`` from disk on first call; return cached objects
        on subsequent calls.

        The PyTorch-Lightning checkpoint stores the model under the key
        ``"student."`` (the ``self.student`` attribute of the wrapper).  This
        prefix is stripped before loading the state-dict.

        :returns: tuple of (model in eval mode, full saved config dict).
        """
        if self._model is not None:
            assert self._saved_config is not None
            return self._model, self._saved_config

        config_path = self._model_dir / "config.json"
        ckpt_path = self._model_dir / "best.ckpt"

        with config_path.open(encoding="utf-8") as fh:
            saved = json.load(fh)

        model_params: dict[str, Any] = saved["model_config"]["model"]["parameters"]
        model = TopSpreadersNetwork(**model_params)

        # Lightning wraps the student model, state keys have "student." prefix.
        # Use _StubPickle to tolerate unknown class references in hyper_parameters.
        raw_ckpt = torch.load(
            str(ckpt_path),
            map_location=self._device,
            pickle_module=_StubPickle,
        )
        state_dict = {
            k.removeprefix("student."): v
            for k, v in raw_ckpt["state_dict"].items()
            if k.startswith("student.")
        }
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()

        self._model = model
        self._saved_config = saved
        return self._model, self._saved_config

    def _get_x_type(self, saved: dict[str, Any]) -> str:
        """
        Parse the ``features_type`` from the saved run config.

        :param saved: full config dict loaded from ``config.json``.

        :returns: feature type string (e.g. ``"zeros"`` or ``"ones"``).
        """
        raw = saved.get("run_config", {}).get("data", {}).get("train_data", "")
        try:
            train_data = ast.literal_eval(raw)
            return str(train_data[0].get("features_type", "zeros"))
        except (ValueError, SyntaxError, IndexError, TypeError):
            _logger.warning("Could not parse features_type from config; defaulting to 'zeros'.")
            return "zeros"

    @staticmethod
    def _build_inputs(
        mlt: MultilayerNetworkTorch,
        x_type: str,
        input_dim: int,
        device: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[Any, torch.Tensor]]:
        """
        Build the three input dicts required by ``TopSpreadersNetwork.forward``.

        Edge indices are extracted from the 3-D sparse adjacency tensor by
        filtering on the first (layer) index -- no dense conversion needed.

        :param mlt: tensor representation of the multilayer network.
        :param x_type: feature initialisation strategy (``"zeros"`` or ``"ones"``).
        :param input_dim: number of input node features expected by the model.
        :param device: PyTorch device string.

        :returns: tuple of ``(x_dict, z_dict, edge_index_dict)``.
        """
        n_actors = len(mlt.actors_map)

        if x_type == "zeros":
            features = torch.zeros(n_actors, input_dim, dtype=torch.float32)
        elif x_type == "ones":
            features = torch.ones(n_actors, input_dim, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported x_type {x_type!r}. "
                "Only 'zeros' and 'ones' are supported without tsds-utils."
            )

        x_dict: dict[str, torch.Tensor] = {"actor": features.to(device)}
        z_dict: dict[str, torch.Tensor] = {
            "actor": mlt.nodes_mask.T.float().to(device)
        }

        # Extract per-layer edge indices from the 3-D sparse adjacency tensor
        # without materialising a dense [n_layers x n_actors x n_actors] matrix.
        idx = mlt.adjacency_tensor.coalesce().indices()  # [3, n_edges_total]
        edge_index_dict: dict[Any, torch.Tensor] = {}
        for layer_pos, layer_name in enumerate(mlt.layers_order):
            mask = idx[0] == layer_pos
            edge_index_dict[("actor", f"l_{layer_pos}", "actor")] = (
                idx[1:, mask].to(device)
            )

        return x_dict, z_dict, edge_index_dict

    # Sort spec matching the original top_k_approach in infmax-trainer-icm-mln:
    # primary exposed desc, then simulation_length asc, peak_infected asc, peak_iteration desc.
    _SORT_SPEC: list[tuple[str, bool]] = [
        ("exposed", False),
        ("simulation_length", True),
        ("peak_infected", True),
        ("peak_iteration", False),
    ]

    def _lexicographic_sort(self, scores: torch.Tensor, y_names: list[str]) -> list[int]:
        """
        Lexicographic actor sort matching the original ``top_k_approach`` order.

        Applied via repeated stable argsort from least to most important criterion
        (reversed iteration).

        :param scores: raw model output shaped ``[n_actors, output_dim]``.
        :param y_names: output column names in the same order as ``scores`` columns.

        :returns: actor indices sorted from best to worst predicted spreader.
        """
        sorted_indices = torch.arange(scores.shape[0])
        for name, ascending in reversed(self._SORT_SPEC):
            if name not in y_names:
                continue
            col = scores[:, y_names.index(name)]
            if not ascending:
                col = -col
            order = col[sorted_indices].argsort(stable=True)
            sorted_indices = sorted_indices[order]
        return sorted_indices.tolist()

    def _run_inference(self, net: MultilayerNetwork) -> list[str]:
        """
        Run the model on ``net`` and return actor IDs sorted by predicted spreading
        influence (EXPOSED desc, SIMULATION_LENGTH asc, PEAK_INFECTED asc, PEAK_ITERATION desc).

        :param net: multilayer network to rank actors for.

        :returns: list of actor ID strings in descending predicted-influence order.
        """
        model, saved = self._load_model()
        input_dim: int = saved["model_config"]["model"]["parameters"]["input_dim"]
        x_type = self._get_x_type(saved)

        raw_y = saved.get("run_config", {}).get("data", {}).get("output_label_name", "[]")
        try:
            y_names: list[str] = ast.literal_eval(raw_y)
        except (ValueError, SyntaxError):
            _logger.warning("Could not parse output_label_name; falling back to index-0 sort.")
            y_names = []

        mlt = MultilayerNetworkTorch.from_mln(net, device=self._device)
        x_dict, z_dict, edge_index_dict = self._build_inputs(
            mlt, x_type, input_dim, self._device
        )

        with torch.no_grad():
            out = model(x_dict, z_dict, edge_index_dict)

        scores = out["actor"].cpu()  # [n_actors, output_dim]
        if y_names:
            sorted_indices = self._lexicographic_sort(scores, y_names)
        else:
            sorted_indices = scores[:, 0].argsort(descending=True).tolist()

        return [str(mlt.actors_map.inverse[idx]) for idx in sorted_indices]

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """
        Return all actors in ``net`` sorted by predicted spreading influence.

        Any actors present in ``net`` but missing from the model's output (e.g.
        due to actor-id type mismatches) are appended at the tail in arbitrary
        order.

        :param net: multilayer network whose actors are to be ranked.

        :returns: actors sorted from highest to lowest predicted spreading
            influence.
        """
        actor_lookup: dict[str, MLNetworkActor] = {
            str(a.actor_id): a for a in net.get_actors()
        }
        ranked_ids = self._run_inference(net)
        ranked: list[MLNetworkActor] = [
            actor_lookup[aid] for aid in ranked_ids if aid in actor_lookup
        ]
        ranked_set = {str(a.actor_id) for a in ranked}
        tail = [a for a in actor_lookup.values() if str(a.actor_id) not in ranked_set]
        return ranked + tail

    def _calculate_ranking_list(self, graph: Any) -> list[Any]:
        """Not used -- ``TsNetSeedSelector`` is actor-wise only."""
        raise NotImplementedError(
            "TsNetSeedSelector ranks at the actor level; use actorwise() directly."
        )

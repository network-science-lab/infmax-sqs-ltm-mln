"""Abstract base class for SS/SQ/SQ_R timing-mode seeding strategies."""

from abc import ABC, abstractmethod
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.selectors import SELECTOR_CLASSES, is_already_active


class BaseSeedingStrategy(ABC):
    """
    Abstract base for seeding strategies that share SS/SQ/SQ_R timing logic.

    Subclasses implement ``_candidates`` to control *which* actors are eligible
    and in what order.  ``next_batch`` applies the chosen timing mode on top of
    that candidate list and is not intended to be overridden.

    Timing modes:

    - ``SS``   -- single-shot: seed all candidates at epoch 0.
    - ``SQ``   -- sequential: ``batch_size`` candidates seeded every epoch.
    - ``SQ_R`` -- sequential with revival: seed only at epoch 0 or when
      spreading has stagnated (``stagnant=True``).

    ``batch_size`` is ignored for ``SS``.
    ``selector_params`` forwards extra keyword arguments to the selector
    constructor (e.g. ``TsNetSeedSelector`` needs ``model_dir``).
    """

    def __init__(
        self,
        selector: str,
        mode: str,
        batch_size: int = 1,
        selector_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the strategy.

        :param selector: key from ``SELECTOR_CLASSES`` identifying the ranking
            method.
        :param mode: timing mode -- ``"SS"``, ``"SQ"``, or ``"SQ_R"``.
        :param batch_size: number of actors to seed per epoch (``SQ`` and
            ``SQ_R`` only).
        :param selector_params: extra keyword arguments forwarded verbatim to
            the selector constructor.  ``None`` is equivalent to ``{}``.
        """
        if selector not in SELECTOR_CLASSES:
            raise ValueError(
                f"Unknown selector {selector!r}. Available: {sorted(SELECTOR_CLASSES)}"
            )
        if mode not in {"SS", "SQ", "SQ_R"}:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: SS, SQ, SQ_R")
        self._selector = SELECTOR_CLASSES[selector](**(selector_params or {}))
        self._mode = mode
        self._batch_size = batch_size

    def _build_subnet(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> MultilayerNetwork:
        """
        Build a subnet restricted to actors eligible for seeding right now.

        An actor is eligible when it is present in ``remaining`` (not yet
        seeded) and not already organically activated in ``net``.  Live actor
        objects from ``net.get_actors()`` are used so that the subnet's node
        attributes reflect the current network state.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: subnet of ``net`` restricted to eligible inactive actors.
        """
        remaining_ids = {a.actor_id for a in remaining}
        eligible = [
            a
            for a in net.get_actors()
            if a.actor_id in remaining_ids and not is_already_active(a, net)
        ]
        return net.subgraph(eligible)

    @abstractmethod
    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return actors eligible for seeding at this moment, in preference order.

        Implementations decide how to rank actors (static pre-computed ranking,
        live subnet re-ranking, etc.).  The caller (``next_batch``) applies
        the timing-mode slice on the result.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.

        :returns: eligible actors in descending preference order.
        """

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """
        Return actors to seed at this epoch according to the timing mode.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.
        :param epoch: current epoch index.
        :param stagnant: ``True`` when spreading has not produced new
            activations for ``patience`` consecutive epochs.

        :returns: actors to seed now; empty list means no seeding this epoch.
        """
        if self._mode == "SS":
            return self._candidates(net, remaining) if epoch == 0 else []

        if self._mode == "SQ":
            return self._candidates(net, remaining)[: self._batch_size]

        if self._mode == "SQ_R":
            if epoch > 0 and not stagnant:
                return []
            return self._candidates(net, remaining)[: self._batch_size]

        raise ValueError(f"Unknown mode: {self._mode!r}")

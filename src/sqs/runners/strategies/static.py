"""Static seeding strategy backed by a pre-computed actor ranking."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.selectors import is_already_active
from sqs.runners.strategies.base import BaseSeedingStrategy


class StaticSeedingStrategy(BaseSeedingStrategy):
    """
    Seed deployment strategy backed by a pre-computed actor ranking.

    The ranking is computed once on the first call to ``_candidates`` via the
    chosen ``BaseSeedSelector``.  Subsequent calls filter out actors that have
    already been seeded (absent from ``remaining``) or organically activated
    by LTM (detected via live network state).  Timing modes are inherited from
    ``BaseSeedingStrategy`` (SS, SQ, SQ_R).

    ``batch_size`` is ignored for ``SS``.
    ``selector_params`` forwards extra keyword arguments to the selector
    constructor, which is needed for selectors that require configuration
    (e.g. ``TsNetSeedSelector`` needs ``model_dir``, ``network_type``,
    ``network_name``).  ``None`` is equivalent to ``{}``.
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
        super().__init__(selector, mode, batch_size, selector_params)
        self._ranking: list[MLNetworkActor] | None = None

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return ranked actors eligible for seeding at this epoch.

        The ranking is computed lazily on the first call, when all actors are
        still inactive, so ``MLNetworkActor.__eq__`` matches ranking entries
        to ``remaining`` entries correctly.  Eligibility requires the actor to
        be present in ``remaining`` (not yet seeded) and not already active in
        the live network (not organically activated by LTM).

        Actors are returned from ``self._ranking``, not from ``remaining``.
        Because the ranking is built at epoch 0 when all actors have
        ``status=None``, ``MLNetworkActor.__eq__`` matches ranking entries
        to ``remaining`` entries by ``actor_id`` and ``_layers_states``,
        so the simulator's ``remaining.remove()`` call works correctly.

        :param net: live network used to check organic activation status.
        :param remaining: actors not yet seeded (passed in from ``next_batch``).

        :returns: subset of ``self._ranking`` eligible for seeding, in ranking
            order.
        """
        if self._ranking is None:
            self._ranking = self._selector.actorwise(net)

        remaining_ids = {a.actor_id for a in remaining}
        return [
            a
            for a in self._ranking  # type: ignore[union-attr]
            if a.actor_id in remaining_ids and not is_already_active(a, net)
        ]

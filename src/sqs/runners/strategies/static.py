"""Static seeding strategy backed by a pre-computed actor ranking."""

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.selectors import SELECTOR_CLASSES, is_already_active


class StaticSeedingStrategy:
    """
    Seed deployment strategy backed by a pre-computed actor ranking.

    The ranking is computed once at epoch 0 via the chosen ``BaseSeedSelector``.
    Subsequent calls to ``next_batch`` filter out actors that have already been
    seeded (absent from ``remaining``) or have been organically activated by LTM
    (detected via live network state).  The three timing modes are:

    - ``SS``   -- single-shot: all candidates seeded at epoch 0, nothing after.
    - ``SQ``   -- sequential: ``batch_size`` candidates seeded every epoch.
    - ``SQ_R`` -- sequential with revival: seed only at epoch 0 or when spreading
      has stagnated (``stagnant=True``).

    ``batch_size`` is ignored for ``SS``.
    """

    def __init__(self, selector: str, mode: str, batch_size: int = 1) -> None:
        """
        Initialise the strategy.

        :param selector: key from ``SELECTOR_CLASSES`` identifying the ranking method.
        :param mode: timing mode -- ``"SS"``, ``"SQ"``, or ``"SQ_R"``.
        :param batch_size: number of actors to seed per epoch (``SQ`` and ``SQ_R`` only).
        """
        if selector not in SELECTOR_CLASSES:
            raise ValueError(
                f"Unknown selector {selector!r}. Available: {sorted(SELECTOR_CLASSES)}"
            )
        if mode not in {"SS", "SQ", "SQ_R"}:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: SS, SQ, SQ_R")
        self._selector = SELECTOR_CLASSES[selector]()
        self._mode = mode
        self._batch_size = batch_size
        self._ranking: list[MLNetworkActor] | None = None

    def _candidates(
        self, remaining: list[MLNetworkActor], net: MultilayerNetwork
    ) -> list[MLNetworkActor]:
        """
        Return ranked actors eligible for seeding at this epoch.

        Eligibility requires the actor to be present in ``remaining`` (not yet
        seeded) and not already active in the live network (not organically
        activated by LTM).  Order follows the pre-computed ranking.

        Actors are returned from ``self._ranking``, not from ``remaining``.
        Because the ranking is built at epoch 0 when all actors have
        ``status=None``, ``MLNetworkActor.__eq__`` matches ranking entries
        to ``remaining`` entries by ``actor_id`` and ``_layers_states``,
        so the simulator's ``remaining.remove()`` call works correctly.

        :param remaining: actors not yet seeded (passed in from ``next_batch``).
        :param net: live network used to check organic activation status.

        :returns: subset of ``self._ranking`` eligible for seeding, in ranking order.
        """
        remaining_ids = {a.actor_id for a in remaining}
        return [
            a
            for a in self._ranking  # type: ignore[union-attr]
            if a.actor_id in remaining_ids and not is_already_active(a, net)
        ]

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """
        Return actors to seed at this epoch according to the timing mode.

        The ranking is computed lazily on the first call (epoch 0), when all
        actors are still inactive, so ``MLNetworkActor.__eq__`` matches ranking
        entries to ``remaining`` entries correctly.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds.
        :param epoch: current epoch index.
        :param stagnant: ``True`` when spreading has not produced new activations
            for ``patience`` consecutive epochs.

        :returns: actors to seed now; empty list means no seeding this epoch.
        """
        if self._ranking is None:
            self._ranking = self._selector.actorwise(net)

        if self._mode == "SS":
            return self._candidates(remaining, net) if epoch == 0 else []

        if self._mode == "SQ":
            return self._candidates(remaining, net)[: self._batch_size]

        if self._mode == "SQ_R":
            if epoch > 0 and not stagnant:
                return []
            return self._candidates(remaining, net)[: self._batch_size]

        raise ValueError(f"Unknown mode: {self._mode!r}")

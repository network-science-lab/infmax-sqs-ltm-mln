"""Static seeding strategy backed by a pre-computed actor ranking."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

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

    def get_or_compute_ranking(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """
        Return the actor ranking, computing it from ``net`` on the first call.

        Safe to call before the repetition loop to amortise expensive selector
        computations (e.g. ``mds``) across all repetitions.

        :param net: network used to compute the ranking when not yet cached.

        :returns: ordered list of actors from most to least preferred.
        """
        if self._ranking is None:
            self._ranking = self._selector.actorwise(net)
        return self._ranking

    def set_precomputed_ranking(self, ranking: list[MLNetworkActor]) -> None:
        """
        Inject a pre-computed ranking to skip the on-first-call computation.

        :param ranking: actor ranking produced by ``get_or_compute_ranking`` on
            a strategy instance with the same selector configuration.
        """
        self._ranking = ranking

    def _candidates(
        self, net: MultilayerNetwork, remaining: list[MLNetworkActor]
    ) -> list[MLNetworkActor]:
        """
        Return ranked actors eligible for seeding at this epoch.

        The ranking is computed lazily on the first call.  Eligibility is
        determined solely by membership in ``remaining``: the simulator prunes
        organically-activated actors from ``remaining`` after every LTM step,
        so no live network lookup is needed here.

        :param net: current state of the multilayer network (used only on the
            first call to build the ranking).
        :param remaining: actors not yet used as seeds.

        :returns: subset of ``self._ranking`` eligible for seeding, in ranking
            order.
        """
        if self._ranking is None:
            self._ranking = self._selector.actorwise(net)

        return self._stable_candidates(self._ranking, remaining)  # type: ignore[arg-type]

"""Seeding strategies for sequential spreading simulations."""

from typing import Protocol, runtime_checkable

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


@runtime_checkable
class SeedingStrategy(Protocol):
    """
    Protocol for seed deployment strategies.

    Implementations control both *which* actors to seed and *when* to seed them.
    The simulator calls ``next_batch`` at every epoch; returning an empty list
    means "no seeding this epoch".
    """

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """
        Return actors to seed at this epoch.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds, in their initial order.
        :param epoch: current epoch index (continuous across the whole simulation).
        :param stagnant: ``True`` when the spreading has not produced new
            activations for ``patience`` consecutive epochs.

        :returns: actors to seed now; empty list means no seeding this epoch.
        """
        ...


class MockSeedingStrategy:
    """
    Mock strategy that seeds one actor per epoch in the order they are provided.

    Ignores network state, epoch index, and stagnation entirely.  Intended as a
    placeholder so that the full pipeline can be exercised end-to-end before real
    strategies are implemented.
    """

    def next_batch(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
        epoch: int,
        stagnant: bool,
    ) -> list[MLNetworkActor]:
        """
        Return the first actor from ``remaining``, or an empty list if none remain.

        :param net: current network state (ignored).
        :param remaining: actors not yet seeded; may be empty.
        :param epoch: current epoch index (ignored).
        :param stagnant: whether spreading has stagnated (ignored).

        :returns: a one-element list with the first remaining actor, or ``[]``.
        """
        if not remaining:
            return []
        return [remaining[0]]

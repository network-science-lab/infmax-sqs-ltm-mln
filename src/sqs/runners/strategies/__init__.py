"""Seeding strategies for sequential spreading simulations."""

from typing import Protocol, runtime_checkable

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork

from sqs.runners.strategies.mock import MockSeedingStrategy
from sqs.runners.strategies.static import StaticSeedingStrategy

__all__ = [
    "MockSeedingStrategy",
    "SeedingStrategy",
    "StaticSeedingStrategy",
]


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

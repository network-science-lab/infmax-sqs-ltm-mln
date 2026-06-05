"""Seed schedule strategies for sequential seeding simulations."""

from typing import Protocol, runtime_checkable

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


@runtime_checkable
class SeedSchedule(Protocol):
    """
    Protocol for sequential seed selection strategies.

    Implementations decide which actor to seed next given the current network
    state and the list of actors not yet seeded.
    """

    def next_seed(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
    ) -> MLNetworkActor:
        """
        Select the next actor to seed.

        :param net: current state of the multilayer network.
        :param remaining: actors not yet used as seeds, in their initial order.

        :returns: the actor chosen as the next seed.
        """
        ...


class MockSeedSchedule:
    """
    Mock seed schedule that returns actors in the order they are provided.

    Ignores network state entirely.  Intended as a placeholder for PR 1 so
    that the full pipeline can be exercised end-to-end before real ranking
    strategies are implemented.
    """

    def next_seed(
        self,
        net: MultilayerNetwork,
        remaining: list[MLNetworkActor],
    ) -> MLNetworkActor:
        """
        Return the first actor from ``remaining``.

        :param net: current network state (ignored).
        :param remaining: actors not yet seeded; must be non-empty.

        :returns: the first actor in ``remaining``.
        """
        if not remaining:
            raise ValueError("No remaining actors to seed.")
        return remaining[0]

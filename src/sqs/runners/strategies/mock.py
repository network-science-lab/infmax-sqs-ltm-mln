"""Mock seeding strategy for pipeline smoke-testing."""

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


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
        return [next(iter(remaining))]

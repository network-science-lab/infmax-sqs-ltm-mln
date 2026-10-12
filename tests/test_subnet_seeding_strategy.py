"""Unit tests for SubnetSeedingStrategy."""

from unittest.mock import patch

import pytest
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.selectors import SELECTOR_CLASSES
from sqs.runners.strategies.subnet import SubnetSeedingStrategy


@pytest.fixture()
def net():
    """Return a fresh toy network with initial states applied."""
    network = get_toy_network_piotr()
    model = SQSMLTModel(protocol="OR", mi_value=0.3)
    model.determine_initial_states(network)
    return network


@pytest.fixture()
def actors(net):
    """Return all actors from the toy network."""
    return net.get_actors()


def _activate(net, actor):
    """Force-activate an actor in the network by setting status=1 on all its layers."""
    for layer in actor.layers:
        net.layers[layer].nodes[actor.actor_id]["status"] = MLTModel.ACTIVE_STATE


class TestUnknownParams:
    def test_unknown_selector_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown selector"):
            SubnetSeedingStrategy(selector="nonexistent", mode="SQ")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            SubnetSeedingStrategy(selector="deg-c", mode="INVALID")


class TestSSMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) > 0

    def test_returns_empty_after_epoch_0(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SS")
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_returns_all_candidates_at_epoch_0(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == len(actors)


class TestSQMode:
    def test_seeds_every_epoch(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        for epoch in range(3):
            batch = strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
            assert len(batch) == 1

    def test_respects_batch_size(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=3)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 3

    def test_batch_size_capped_by_remaining(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=100)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == len(actors)


class TestSQRMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 1

    def test_skips_when_not_stagnant(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_seeds_when_stagnant(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=True)
        assert len(batch) == 1


class TestActiveActorFiltering:
    def test_organically_active_actor_excluded_from_subnet(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        # Simulate full organic spread: the simulator prunes all actors from remaining.
        for actor in actors:
            _activate(net, actor)
        remaining_after_ltm: list = []  # simulator would have pruned all actors
        batch = strategy.next_batch(net, remaining_after_ltm, epoch=0, stagnant=False)
        assert batch == []

    def test_active_actor_not_in_batch(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        # Simulate the simulator pruning actors[0] after it organically activated.
        _activate(net, actors[0])
        remaining = list(actors[1:])
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        assert actors[0] not in batch

    def test_active_actor_reduces_batch_size(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        # Simulate the simulator pruning actors[0] after it organically activated.
        _activate(net, actors[0])
        remaining = list(actors[1:])
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        assert len(batch) == len(remaining)

    def test_seeded_actor_absent_from_batch(self, net, actors) -> None:
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        # Remove one actor from remaining, simulating a prior seeding event.
        remaining = list(actors)
        remaining.pop(0)
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        assert len(batch) == len(remaining)

    def test_batch_actors_are_from_remaining(self, net, actors) -> None:
        """Returned actors must be objects from ``remaining`` so remove() works."""
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        remaining = list(actors)
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        assert all(a in remaining for a in batch)


class TestRankingRecomputed:
    def test_actorwise_called_each_seeding_epoch(self, net, actors) -> None:
        """Without a budget (no simulator) the ranking is recomputed at every seeding call."""
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        with patch.object(strategy._selector, "actorwise", wraps=strategy._selector.actorwise) as mock_aw:
            for epoch in range(4):
                strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
        assert mock_aw.call_count == 4

    def test_actorwise_not_called_when_no_seeding_sqr(self, net, actors) -> None:
        """SQ_R must skip re-ranking on non-stagnant epochs after epoch 0."""
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        with patch.object(strategy._selector, "actorwise", wraps=strategy._selector.actorwise) as mock_aw:
            strategy.next_batch(net, list(actors), epoch=0, stagnant=False)  # seeds
            strategy.next_batch(net, list(actors), epoch=1, stagnant=False)  # skips
            strategy.next_batch(net, list(actors), epoch=2, stagnant=False)  # skips
            strategy.next_batch(net, list(actors), epoch=3, stagnant=True)   # seeds
        assert mock_aw.call_count == 2

    def test_actorwise_not_called_after_epoch_0_ss(self, net, actors) -> None:
        """SS must call actorwise exactly once (at epoch 0) and never again."""
        strategy = SubnetSeedingStrategy(selector="deg-c", mode="SS")
        with patch.object(strategy._selector, "actorwise", wraps=strategy._selector.actorwise) as mock_aw:
            for epoch in range(4):
                strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
        assert mock_aw.call_count == 1


_SELECTORS_NO_CHECKPOINT = [k for k in SELECTOR_CLASSES if k != "ts-net"]


class TestAllSelectors:
    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_selector_returns_nonempty_batch(self, net, actors, selector_key: str) -> None:
        strategy = SubnetSeedingStrategy(selector=selector_key, mode="SQ", batch_size=1)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 1

    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_selector_batch_actors_are_subset_of_remaining(
        self, net, actors, selector_key: str
    ) -> None:
        remaining = list(actors)
        strategy = SubnetSeedingStrategy(selector=selector_key, mode="SQ", batch_size=len(actors))
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        remaining_ids = {a.actor_id for a in remaining}
        assert all(a.actor_id in remaining_ids for a in batch)


@pytest.mark.parametrize(("rerank_percent", "max_rankings"), [(0, None), (10, 10), (50, 2)])
def test_rerank_percent_limits_rankings(rerank_percent: float, max_rankings: int | None) -> None:
    """The subnet is re-ranked once per ``rerank_percent`` % of the budget (0: every seed)."""
    from sqs.loaders.net_loader import load_network
    from sqs.models.propagation.mltm import SQSMLTModel
    from sqs.runners.simulator import SequentialSimulator

    strategy = SubnetSeedingStrategy("deg-c-d", "SQ", 1, rerank_percent=rerank_percent)
    with patch.object(strategy._selector, "actorwise", wraps=strategy._selector.actorwise) as ranking:
        log = SequentialSimulator(SQSMLTModel("AND", 0.5), load_network("lazega")).perform_propagation(
            30, strategy, max_epochs=None, patience=2, track_full_logs=False
        )
    seeds = sum(len(v) for v in log.seeding_events.values())
    assert ranking.call_count == seeds if max_rankings is None else 1 <= ranking.call_count <= max_rankings

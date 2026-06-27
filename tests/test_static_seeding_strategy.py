"""Unit tests for StaticSeedingStrategy."""

from unittest.mock import patch

import pytest
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.selectors import SELECTOR_CLASSES
from sqs.runners.strategies import StaticSeedingStrategy


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
            StaticSeedingStrategy(selector="nonexistent", mode="SS")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            StaticSeedingStrategy(selector="deg-c", mode="INVALID")


class TestSSMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) > 0

    def test_returns_empty_after_epoch_0(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SS")
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_returns_all_candidates_at_epoch_0(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        # SS returns all candidates at once; simulator will cap to n_seeds
        assert len(batch) == len(actors)


class TestSQMode:
    def test_seeds_every_epoch(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        for epoch in range(3):
            batch = strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
            assert len(batch) == 1

    def test_respects_batch_size(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=3)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 3

    def test_batch_size_capped_by_remaining(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=100)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == len(actors)


class TestSQRMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 1

    def test_skips_when_not_stagnant(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_seeds_when_stagnant(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=True)
        assert len(batch) == 1


class TestActiveActorFiltering:
    def test_organically_active_actor_skipped(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        # build ranking first
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)

        # activate every actor organically (simulating LTM)
        for actor in actors:
            _activate(net, actor)

        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_partially_active_reduces_batch(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)

        _activate(net, actors[0])

        batch = strategy.next_batch(net, list(actors[1:]), epoch=1, stagnant=False)
        assert actors[0] not in batch


class TestRankingComputedOnce:
    def test_actorwise_called_once(self, net, actors) -> None:
        strategy = StaticSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        with patch.object(strategy._selector, "actorwise", wraps=strategy._selector.actorwise) as mock_aw:
            for epoch in range(5):
                strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
        mock_aw.assert_called_once()


_SELECTORS_NO_CHECKPOINT = [k for k in SELECTOR_CLASSES if k != "ts-net"]


class TestAllSelectors:
    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_selector_returns_nonempty_batch(self, net, actors, selector_key: str) -> None:
        strategy = StaticSeedingStrategy(selector=selector_key, mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) > 0

    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_selector_batch_actors_are_subset_of_remaining(
        self, net, actors, selector_key: str
    ) -> None:
        remaining = list(actors)
        strategy = StaticSeedingStrategy(selector=selector_key, mode="SS")
        batch = strategy.next_batch(net, remaining, epoch=0, stagnant=False)
        remaining_ids = {a.actor_id for a in remaining}
        assert all(a.actor_id in remaining_ids for a in batch)

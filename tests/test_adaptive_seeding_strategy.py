"""Unit tests for AdaptiveSeedingStrategy."""

from unittest.mock import patch

import pytest
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.selectors import SELECTOR_CLASSES
from sqs.runners.strategies.adaptive import AdaptiveSeedingStrategy


# ---------------------------------------------------------------------------
# Toy-network topology reference (toy_network_piotr):
#   l1 edges: (1,4),(4,2),(4,3),(4,5),(2,3),(3,5),(3,6),(7,9),(9,8),(8,10)
#   l2 edges: (1,2),(2,7),(2,11),(7,9),(11,5),(11,10),(4,5),(4,6),(6,10),(10,8),(9,8)
#   l3 edges: (1,4),(4,3),(4,5),(2,6),(2,9),(6,5),(6,9),(9,7),(3,5),(5,11),(11,10)
#
# When actor 4 is activated, its neighbours across all layers are: {1,2,3,5,6}.
# Actors {7,8,9,10,11} have no direct edge to actor 4 in any layer.
# ---------------------------------------------------------------------------


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


def _actor_by_id(actors, actor_id):
    """Return the actor object with the given ID."""
    return next(a for a in actors if a.actor_id == actor_id)


class TestUnknownParams:
    def test_unknown_selector_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown selector"):
            AdaptiveSeedingStrategy(selector="nonexistent", mode="SQ")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            AdaptiveSeedingStrategy(selector="deg-c", mode="INVALID")


class TestSSMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SS")
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) > 0

    def test_returns_empty_after_epoch_0(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SS")
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []


class TestSQMode:
    def test_seeds_every_epoch(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        for epoch in range(3):
            batch = strategy.next_batch(net, list(actors), epoch=epoch, stagnant=False)
            assert len(batch) == 1

    def test_respects_batch_size(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=3)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 3


class TestSQRMode:
    def test_seeds_at_epoch_0(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 1

    def test_skips_when_not_stagnant(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=False)
        assert batch == []

    def test_seeds_when_stagnant(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ_R", batch_size=1)
        strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        batch = strategy.next_batch(net, list(actors), epoch=1, stagnant=True)
        assert len(batch) == 1


class TestFallback:
    def test_uses_selector_when_no_active_nodes(self, net, actors) -> None:
        """At epoch 0 there are no active nodes; selector actorwise must be called."""
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        with patch.object(
            strategy._selector, "actorwise", wraps=strategy._selector.actorwise
        ) as mock_aw:
            strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        mock_aw.assert_called_once()

    def test_returns_nonempty_batch_when_no_active_nodes(self, net, actors) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=2)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 2

    def test_empty_remaining_returns_empty(self, net) -> None:
        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        batch = strategy.next_batch(net, [], epoch=1, stagnant=False)
        assert batch == []


class TestFrontierFilter:
    """When actor 4 is activated its neighbours {1,2,3,5,6} form the frontier."""

    def test_frontier_contains_only_neighbours_of_active_node(self, net, actors) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(remaining))
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)

        batch_ids = {a.actor_id for a in batch}
        # Actors {7,8,9,10,11} share no edge with actor 4 in any layer.
        non_neighbours = {7, 8, 9, 10, 11}
        assert batch_ids.isdisjoint(non_neighbours), (
            f"Non-neighbours {non_neighbours & batch_ids} appeared in frontier batch"
        )

    def test_frontier_includes_all_direct_neighbours(self, net, actors) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(remaining))
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)

        batch_ids = {a.actor_id for a in batch}
        expected_frontier = {1, 2, 3, 5, 6}
        assert expected_frontier.issubset(batch_ids)

    def test_active_actor_not_in_batch(self, net, actors) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = list(actors)

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(actors))
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)
        assert actor4 not in batch

    def test_selector_not_called_when_frontier_exists(self, net, actors) -> None:
        """When frontier candidates exist the selector fallback must be skipped."""
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=1)
        with patch.object(
            strategy._selector, "actorwise", wraps=strategy._selector.actorwise
        ) as mock_aw:
            strategy.next_batch(net, remaining, epoch=1, stagnant=False)
        mock_aw.assert_not_called()


class TestActivationPotentialRanking:
    """
    Actor 1 has actor 4 as a neighbour in both l1 and l3.
    When actor 4 is the only active node:
      score(1) = 2/3 ~ 0.667   (2 active neighbours out of 3 total)
      score(5) = 3/8 = 0.375   (4 appears in l1, l2, l3 for actor 5: 3 times / total 8)
    Actor 1 must rank above actor 5.
    """

    def test_actor_with_higher_ratio_ranks_first(self, net, actors) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(remaining))
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)

        batch_ids = [a.actor_id for a in batch]
        idx_1 = batch_ids.index(1)
        idx_5 = batch_ids.index(5)
        assert idx_1 < idx_5, (
            f"Actor 1 (score ~0.667) should rank above actor 5 (score 0.375), "
            f"but positions were {idx_1} and {idx_5}"
        )

    def test_batch_actors_are_from_remaining(self, net, actors) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector="deg-c", mode="SQ", batch_size=len(remaining))
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)
        assert all(a in remaining for a in batch)


_SELECTORS_NO_CHECKPOINT = [k for k in SELECTOR_CLASSES if k != "ts-net"]


class TestAllSelectors:
    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_fallback_returns_nonempty_batch(self, net, actors, selector_key: str) -> None:
        """Selector fallback (no active nodes) must work for every selector."""
        strategy = AdaptiveSeedingStrategy(selector=selector_key, mode="SQ", batch_size=1)
        batch = strategy.next_batch(net, list(actors), epoch=0, stagnant=False)
        assert len(batch) == 1

    @pytest.mark.parametrize("selector_key", _SELECTORS_NO_CHECKPOINT)
    def test_frontier_batch_actors_in_remaining(self, net, actors, selector_key: str) -> None:
        actor4 = _actor_by_id(actors, 4)
        _activate(net, actor4)
        remaining = [a for a in actors if a.actor_id != 4]

        strategy = AdaptiveSeedingStrategy(selector=selector_key, mode="SQ", batch_size=3)
        batch = strategy.next_batch(net, remaining, epoch=1, stagnant=False)
        assert all(a in remaining for a in batch)

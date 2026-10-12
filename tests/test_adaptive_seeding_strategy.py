"""Unit tests for AdaptiveSeedingStrategy (learnt activation rule + push)."""

import random
from fractions import Fraction

import pytest
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.loaders.net_loader import load_network
from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.simulator import SequentialSimulator
from sqs.runners.strategies import AdaptiveSeedingStrategy, StaticSeedingStrategy
from sqs.runners.strategies.adaptive import MU_GRID
from sqs.runners.strategies.base import ActorPool


@pytest.fixture()
def net():
    """Toy network with all actors inactive."""
    network = get_toy_network_piotr()
    model = SQSMLTModel(protocol="OR", mi_value=0.3)
    model.update_network(network, model.determine_initial_states(network))
    return network


def _activate(net, actor_ids) -> None:
    """Force-activate actors on all their layers."""
    actors = {a.actor_id: a for a in net.get_actors()}
    SQSMLTModel("OR", 0.3).update_network(
        net,
        [NUBuff(aid, layer, MLTModel.ACTIVE_STATE) for aid in actor_ids for layer in actors[aid].layers],
    )


def _true_delta(protocol: str, net) -> Fraction:
    """delta of OR (any layer: the smallest fraction) or AND (all layers)."""
    return Fraction(1, max(len(a.layers) for a in net.get_actors())) if protocol == "OR" else Fraction(1)


class TestModes:
    def test_ss_seeds_only_at_epoch_0(self, net) -> None:
        strategy = AdaptiveSeedingStrategy("deg-c", "SS")
        assert len(strategy.next_batch(net, ActorPool(net.get_actors()), epoch=0, stagnant=False)) > 0
        assert strategy.next_batch(net, ActorPool(net.get_actors()), epoch=1, stagnant=False) == []

    def test_sq_seeds_every_epoch(self, net) -> None:
        strategy = AdaptiveSeedingStrategy("deg-c", "SQ")
        for epoch in range(3):
            assert len(strategy.next_batch(net, ActorPool(net.get_actors()), epoch, stagnant=False)) == 1

    def test_sq_r_waits_for_stagnation(self, net) -> None:
        strategy = AdaptiveSeedingStrategy("deg-c", "SQ_R")
        assert len(strategy.next_batch(net, ActorPool(net.get_actors()), epoch=0, stagnant=False)) == 1
        assert strategy.next_batch(net, ActorPool(net.get_actors()), epoch=1, stagnant=False) == []
        assert len(strategy.next_batch(net, ActorPool(net.get_actors()), epoch=1, stagnant=True)) == 1

    @pytest.mark.parametrize("selector", ["deg-c", "deg-c-d"])
    def test_ss_order_is_push_then_ranking(self, net, selector: str) -> None:
        """SS: push descending, ties in the selector (Static) ranking order."""
        _activate(net, [4])
        actors = ActorPool(a for a in net.get_actors() if a.actor_id != 4)
        strategy = AdaptiveSeedingStrategy(selector, "SS", rerank_percent=100)
        strategy.budget = 5
        order = [a.actor_id for a in strategy.next_batch(net, actors, epoch=0, stagnant=False)]
        ranking = StaticSeedingStrategy(selector, "SS").get_or_compute_ranking(net)
        expected = sorted(
            (a.actor_id for a in ranking if a.actor_id in actors.ids and a.actor_id not in strategy._self_active),
            key=lambda aid: -strategy._push[aid],
        )
        assert order == expected


@pytest.mark.parametrize(("rerank_percent", "max_rankings"), [(0, None), (10, 1 + 10), (100, 1)])
def test_rerank_percent_limits_rankings(rerank_percent: float, max_rankings: int | None) -> None:
    """0: a subnet ranking before every seed; p: one per p % of the budget; 100: Static only."""
    network = load_network("lazega")  # budget 30: 10 % -> every 3 seeds
    strategy = AdaptiveSeedingStrategy("deg-c-d", "SQ", rerank_percent=rerank_percent)
    calls = {"rankings": 0, "seeds": 0}
    actorwise, candidates = strategy._selector.actorwise, strategy._candidates

    def counted_actorwise(net):
        calls["rankings"] += 1
        return actorwise(net)

    def counted_candidates(net, remaining):
        calls["seeds"] += 1
        return candidates(net, remaining)

    strategy._selector.actorwise, strategy._candidates = counted_actorwise, counted_candidates
    SequentialSimulator(SQSMLTModel("AND", 0.5), network).perform_propagation(
        30, strategy, max_epochs=None, patience=2, track_full_logs=False
    )
    if max_rankings is None:
        assert calls["rankings"] == 1 + calls["seeds"]  # initial ranking + one per seed
    else:
        assert 1 <= calls["rankings"] <= max_rankings


@pytest.mark.parametrize("net_name", ["toy", "lazega"])
@pytest.mark.parametrize("protocol", ["OR", "AND"])
@pytest.mark.parametrize("mu", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("mode", ["SQ", "SQ_R"])
def test_true_rule_stays_feasible(net_name: str, protocol: str, mu: float, mode: str) -> None:
    """After a whole run the true (mu, delta) is still consistent with every observation."""
    network = get_toy_network_piotr() if net_name == "toy" else load_network(net_name)
    strategy = AdaptiveSeedingStrategy("deg-c", mode)
    n_seeds = max(2, round(0.2 * network.get_actors_num()))
    random.seed(0)
    sim = SequentialSimulator(SQSMLTModel(protocol, mu), network.copy())
    sim.perform_propagation(n_seeds, strategy, max_epochs=None, patience=2)
    delta = _true_delta(protocol, network)
    assert strategy._lo[mu] < delta <= strategy._hi[mu]


@pytest.mark.parametrize("protocol", ["OR", "AND"])
@pytest.mark.parametrize("mu", [0.2, 0.3, 0.5])
def test_top_candidate_has_max_brute_force_push(net, protocol: str, mu: float) -> None:
    """With the estimate set to the true rule, the top candidate pushes the most actors."""
    active = {4} if protocol == "OR" else {1, 5}
    _activate(net, active)
    strategy = AdaptiveSeedingStrategy("deg-c", "SQ")
    pool = ActorPool(a for a in net.get_actors() if a.actor_id not in active)
    delta = _true_delta(protocol, net)
    for m in MU_GRID:  # only the true rule stays feasible
        strategy._lo[m], strategy._hi[m] = (delta - Fraction(1, 100), delta) if m == mu else (1, 1)
    strategy._start(net, pool)

    model = SQSMLTModel(protocol, mu)

    def newly_active(extra: list) -> set:
        trial = net.copy()
        for layer, graph in net.layers.items():
            for node, status in graph.nodes(data="status"):
                trial.layers[layer].nodes[node]["status"] = status
        _activate(trial, extra)
        active_before = {
            a.actor_id for a in trial.get_actors() if set(a.states.values()) == {MLTModel.ACTIVE_STATE}
        }
        after = {b.node_name for b in MLTModel.network_evaluation_step(model, trial) if b.new_state == MLTModel.ACTIVE_STATE}
        return after - active_before

    on_their_own = newly_active([])
    push = {
        aid: len(newly_active([aid]) - on_their_own - {aid})
        for aid in pool.ids
        if aid not in on_their_own
    }
    top = strategy._candidates(net, pool)[0].actor_id
    assert top not in on_their_own
    assert push[top] == max(push.values())
    best = [aid for aid, p in push.items() if p == push[top]]
    assert strategy._rank[top] == min(strategy._rank[aid] for aid in best)


def test_runs_end_to_end_on_lazega() -> None:
    network = load_network("lazega")
    for mode in ["SQ", "SQ_R"]:
        log = SequentialSimulator(SQSMLTModel("AND", 0.5), network.copy()).perform_propagation(
            15, AdaptiveSeedingStrategy("deg-c-d", mode), max_epochs=None, patience=2
        )
        assert sum(len(v) for v in log.seeding_events.values()) <= 15

"""The simulator (changes-only LTM steps) must give the same results as the classic full-step loop."""

import random

import pytest
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.loaders.net_loader import load_network
from sqs.misc.result_handler import extract_simulation_result
from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.simulator import SequentialSimulator
from sqs.runners.strategies import StaticSeedingStrategy


def classic_run(net, model, strategy, n_seeds: int, patience: int) -> tuple[dict, dict]:
    """
    Reference loop: seed -> full MLTModel step on every actor, stagnation by snapshot comparison.

    :returns: seeding events and cumulative active counts per epoch.
    """
    model.update_network(net, model.determine_initial_states(net))
    remaining = list(net.get_actors())
    seeding_events, active_counts = {}, {}
    planted, counter, stagnant, old_states, epoch = 0, 0, False, [], 0
    while True:
        seeds = []
        if planted < n_seeds:
            seeds = strategy.next_batch(net, remaining, epoch, stagnant)[: n_seeds - planted]
        if seeds:
            for actor in seeds:
                remaining.remove(actor)
            model.update_network(
                net,
                [NUBuff(a.actor_id, layer, MLTModel.ACTIVE_STATE) for a in seeds for layer in a.layers],
            )
            seeding_events[epoch] = [str(a.actor_id) for a in seeds]
            planted += len(seeds)
        new_states = MLTModel.network_evaluation_step(model, net)
        model.update_network(net, new_states)
        active = {b.node_name for b in new_states if b.new_state == MLTModel.ACTIVE_STATE}
        active_counts[epoch] = len(active)
        remaining = [a for a in remaining if a.actor_id not in active]
        if not remaining:
            break
        counter = 0 if seeds or set(new_states) != set(old_states) else counter + 1
        stagnant = counter >= patience
        if not stagnant:
            old_states = new_states
        if planted >= n_seeds and stagnant:
            break
        epoch += 1
    return seeding_events, active_counts


@pytest.mark.parametrize("net_name", ["toy", "lazega"])
@pytest.mark.parametrize("protocol", ["OR", "AND"])
def test_same_results_as_classic_loop(net_name: str, protocol: str) -> None:
    net = get_toy_network_piotr() if net_name == "toy" else load_network(net_name)
    total = net.get_actors_num()
    ranking = StaticSeedingStrategy("deg-c", "SS").get_or_compute_ranking(net)
    for mi in [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]:
        for budget in [0.05, 0.15, 0.3]:
            n_seeds = max(1, round(budget * total))
            for mode in ["SS", "SQ", "SQ_R"]:
                runs = []
                for classic in (True, False):
                    random.seed(0)
                    strategy = StaticSeedingStrategy("deg-c", mode)
                    strategy.set_precomputed_ranking(ranking)
                    model = SQSMLTModel(protocol, mi)
                    if classic:
                        runs.append(classic_run(net.copy(), model, strategy, n_seeds, patience=2))
                    else:
                        log = SequentialSimulator(model, net.copy()).perform_propagation(
                            n_seeds, strategy, max_epochs=None, patience=2, track_full_logs=False
                        )
                        runs.append((log.seeding_events, log.active_counts))
                case = (mi, budget, mode)
                assert runs[0] == runs[1], case
                assert extract_simulation_result(_log(*runs[0]), total, n_seeds) == (
                    extract_simulation_result(_log(*runs[1]), total, n_seeds)
                ), case


def _log(seeding_events: dict, active_counts: dict):
    """Wrap a timeline in the minimal object ``extract_simulation_result`` reads."""
    return type("Log", (), {"seeding_events": seeding_events, "active_counts": active_counts})


def test_activations_match_full_step() -> None:
    """``activations`` on all inactive actors equals the newly active ones of a full MLTModel step."""
    net = load_network("lazega")
    for protocol in ["OR", "AND"]:
        for mi in [0.2, 0.4, 0.6]:
            model = SQSMLTModel(protocol, mi)
            model.update_network(net, model.determine_initial_states(net))
            random.seed(1)
            active = set(random.sample([a.actor_id for a in net.get_actors()], 20))
            model.update_network(
                net,
                [
                    NUBuff(a.actor_id, layer, MLTModel.ACTIVE_STATE)
                    for a in net.get_actors()
                    if a.actor_id in active
                    for layer in a.layers
                ],
            )
            inactive = [a for a in net.get_actors() if a.actor_id not in active]
            full = {
                b.node_name
                for b in MLTModel.network_evaluation_step(model, net)
                if b.new_state == MLTModel.ACTIVE_STATE
            }
            assert {a.actor_id for a in model.activations(net, inactive)} == full - active

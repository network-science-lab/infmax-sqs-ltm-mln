"""Multilayer Linear Threshold Model variant for sequential seeding."""

from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork as MLNetwork
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBuff
from network_diffusion.seeding.mocking_selector import MockingActorSelector


class SQSMLTModel(MLTModel):
    """
    MLTModel adapted for sequential seeding experiments.

    Overrides ``determine_initial_states`` to unconditionally mark every actor as
    INACTIVE, bypassing the budget/selector mechanism inherited from ``MLTModel``.
    Seeds are injected one at a time by ``SequentialSimulator`` during the run loop.
    """

    def __init__(self, protocol: str, mi_value: float) -> None:
        """
        Initialise the model.

        :param protocol: activation protocol; either ``"AND"`` or ``"OR"``.
        :param mi_value: activation threshold in [0, 1]; a node becomes active when
            the fraction of active neighbours in a layer exceeds this value.
        """
        super().__init__(
            seeding_budget=(100, 0),
            seed_selector=MockingActorSelector(preselected_actors=[]),
            protocol=protocol,
            mi_value=mi_value,
        )
        self._mi_value = mi_value
        self._needs: dict[int, int] = {}  # degree -> active neighbours needed

    def determine_initial_states(self, net: MLNetwork) -> list[NUBuff]:
        """
        Return an all-INACTIVE assignment for every node in the network.

        :param net: multilayer network to initialise.

        :returns: list of ``NetworkUpdateBuffer`` entries setting every node to the
            INACTIVE state across all layers.
        """
        initial_states: list[NUBuff] = []
        for actor in net.get_actors():
            for layer in actor.layers:
                initial_states.append(
                    NUBuff(
                        node_name=actor.actor_id,
                        layer_name=layer,
                        new_state=self.INACTIVE_STATE,
                    )
                )
        return initial_states

    def activations(
        self,
        net: MLNetwork,
        candidates: list[MLNetworkActor],
        active_neighbours: dict[tuple[str, Any], int] | None = None,
    ) -> list[MLNetworkActor]:
        """
        Return the inactive ``candidates`` that activate in one synchronous LTM step.

        Same rule as ``MLTModel.network_evaluation_step``: a layer gives a positive input when the
        impulse (1 / degree per active neighbour, summed as floats) exceeds ``mi``; the protocol
        combines layer inputs. The network is not modified.

        :param net: network in its current state.
        :param candidates: inactive actors to evaluate.
        :param active_neighbours: active-neighbour count per (layer, actor id), if maintained by
            the caller; otherwise neighbours are counted in ``net``.

        :returns: actors that become active.
        """
        return [
            actor
            for actor in candidates
            if self.protocol(
                {
                    layer: self._layer_input(net, layer, actor.actor_id, active_neighbours)
                    for layer in actor.layers
                }
            )
        ]

    def _layer_input(
        self,
        net: MLNetwork,
        layer: str,
        actor_id: Any,
        active_neighbours: dict[tuple[str, Any], int] | None,
    ) -> str:
        """Return the input of one node: ACTIVE when enough of its neighbours are active."""
        graph = net.layers[layer]
        if active_neighbours is None:
            active = sum(
                graph.nodes[nb]["status"] == self.ACTIVE_STATE for nb in graph.adj[actor_id]
            )
        else:
            active = active_neighbours.get((layer, actor_id), 0)
        passes = active >= self._need(graph.degree(actor_id))
        return self.ACTIVE_STATE if passes else self.INACTIVE_STATE

    def _need(self, degree: int) -> int:
        """Return the fewest active neighbours whose summed impulse exceeds ``mi``."""
        if degree not in self._needs:
            impulse, need = 0.0, degree + 1  # degree + 1: never
            for count in range(1, degree + 1):
                impulse += 1 / degree
                if impulse > self._mi_value:
                    need = count
                    break
            self._needs[degree] = need
        return self._needs[degree]

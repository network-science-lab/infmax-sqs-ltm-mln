"""Multilayer Linear Threshold Model variant for sequential seeding."""

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

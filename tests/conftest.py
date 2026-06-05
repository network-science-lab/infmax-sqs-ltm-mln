"""Shared pytest fixtures."""

import pytest
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

from sqs.models.mltm import SQSMLTModel
from sqs.runners.seed_schedule import MockSeedSchedule
from sqs.runners.simulator import SequentialSimulator


@pytest.fixture()
def toy_net():
    """Return a fresh copy of the Piotr toy multilayer network for each test."""
    return get_toy_network_piotr()


@pytest.fixture()
def or_model():
    """Return an OR-protocol SQSMLTModel with mi=0.15."""
    return SQSMLTModel(protocol="OR", mi_value=0.15)


@pytest.fixture()
def mock_schedule():
    """Return a MockSeedSchedule instance."""
    return MockSeedSchedule()


@pytest.fixture()
def simulator(or_model, toy_net):
    """Return a SequentialSimulator bound to the toy network copy."""
    return SequentialSimulator(or_model, toy_net.copy())

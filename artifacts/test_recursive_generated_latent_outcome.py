import torch

from artifacts.evaluate_recursive_generated_latent_outcome import _led_to
from d4mj.config import Config


def test_led_to_excludes_the_action_being_evaluated():
    config = Config(device="cpu")
    actions = torch.arange(20) % config.n_actions
    incoming = _led_to(actions, 4, 9, config)
    assert torch.equal(incoming, actions[3:8])
    assert actions[8] not in incoming[-1:]

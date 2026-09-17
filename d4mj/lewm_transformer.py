"""The pinned base-LeWM predictor package, imported rather than reimplemented.

A comparison backend only. Mamba remains the architecture under study; this exists so a
faithful upstream predictor can be measured on the same task, which is what makes our own
adaptation's mistakes visible.

Nothing here is a copy of the source. `ARPredictor`, `Embedder` and `MLP` are instantiated
from the vendored file itself, loaded under a private module name so it can never collide
with anything on `sys.path`, and only after its bytes are checked against `SOURCES.lock`'s
commit. Reimplementing them would defeat the point of the control.
"""

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .state import WindowPredictiveState

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "third_party/sources/lucas-maes__le-wm"
# The audited bytes of the classes this backend instantiates, pinned so a vendor bump is a
# loud failure rather than a silently different control.
PINNED = {
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "config/train/model/lewm.yaml": "7be97eaa2c83f809b9ea7a6da5a7d3f3c70502c27537383f54db32335e21e42b",
}
PRIVATE_NAME = "_d4mj_pinned_lewm_module"


def _digest(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            out.update(block)
    return out.hexdigest()


def source_digests() -> dict[str, str]:
    """Measured digests of every pinned file, for the source manifest and for tests."""
    return {name: _digest(SOURCE / name) for name in PINNED}


def source_module():
    """The vendored module, verified and loaded once under a private name."""
    if PRIVATE_NAME in sys.modules:
        return sys.modules[PRIVATE_NAME]
    for name, expected in PINNED.items():
        actual = _digest(SOURCE / name)
        if actual != expected:
            raise ValueError(f"lewm_transformer: pinned {name} differs from the audited bytes")
    path = SOURCE / "module.py"
    spec = importlib.util.spec_from_file_location(PRIVATE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_NAME] = module
    spec.loader.exec_module(module)
    return module


def build_package(*, width=192, depth=6, heads=16, head_dim=64, mlp_dim=2048, context=3,
                  n_actions=17, dropout=0.1, embedding_dropout=0.0,
                  action_smoothed_dim=10, action_mlp_scale=4):
    """The three pinned modules, with the constructor arguments from `lewm.yaml`.

    Returned separately rather than wrapped: the world owns how they are called, and the
    oracle test needs to compare these exact objects against its own instantiation.
    """
    import torch

    source = source_module()
    predictor = source.ARPredictor(num_frames=context, depth=depth, heads=heads,
                                   mlp_dim=mlp_dim, input_dim=width, hidden_dim=width,
                                   output_dim=width, dim_head=head_dim, dropout=dropout,
                                   emb_dropout=embedding_dropout)
    action_encoder = source.Embedder(input_dim=n_actions, smoothed_dim=action_smoothed_dim,
                                     emb_dim=width, mlp_scale=action_mlp_scale)
    projector = source.MLP(input_dim=width, hidden_dim=mlp_dim, output_dim=width,
                           norm_fn=torch.nn.BatchNorm1d)
    return predictor, action_encoder, projector


class LeWMTransformerWorld(nn.Module):
    """The pinned predictor package as a world, with the source's finite-window semantics.

    The source keeps no carry: `JEPA.rollout` recomputes the last `context` pairs and resets
    positions to `0:len` on every call. So this world buffers inputs rather than a recurrent
    memory, and every transition is a fresh bounded forward. A persistent KV cache would not
    be equivalent -- when the window slides the positions change, and the retained keys carry
    indirect dependencies on the evicted token.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.dynamics
        self.predictor, self.action_encoder, self.predictor_projector = build_package(
            width=d.width, depth=d.depth, heads=d.heads, head_dim=d.head_dim,
            mlp_dim=d.mlp_dim, context=d.context, n_actions=d.n_actions,
            dropout=d.dropout, embedding_dropout=d.embedding_dropout,
            action_smoothed_dim=d.action_smoothed_dim, action_mlp_scale=d.action_mlp_scale)
        # Kept only so existing tooling that asks a world for agent features still works.
        # It is not part of the source predictor, never feeds prediction, and never trains.
        # The source predictor's hidden width is the latent width, not Mamba's 256, so the
        # shim reads latent + native h and still emits the 256-wide feature tooling expects.
        self.agent_readout = nn.Sequential(
            nn.Linear(config.encoder.latent_dim + d.width, d.readout_width),
            nn.LayerNorm(d.readout_width, eps=1e-6), nn.GELU())
        self.agent_readout.requires_grad_(False)

    @property
    def context(self) -> int:
        return self.config.dynamics.context

    def _latents(self, z):
        if z.ndim != 4 or tuple(z.shape[2:]) != (1, self.config.encoder.latent_dim) or z.shape[1] < 1:
            raise ValueError("world latent must be B,T,1,latent_dim with T>=1")

    def _actions(self, a, shape):
        if a.dtype != torch.long or tuple(a.shape) != tuple(shape):
            raise ValueError("outgoing actions must be int64 B,T matching completed pairs")
        if bool(((a < 0) | (a >= self.config.dynamics.n_actions)).any()):
            raise ValueError("outgoing action out of range")

    def validate_state(self, state):
        if not isinstance(state, WindowPredictiveState):
            raise ValueError("transformer world requires a WindowPredictiveState")
        self._latents(state.latent)
        keep = self.context - 1
        if state.past_latents.ndim != 3 or state.past_latents.shape[1] > keep:
            raise ValueError(f"window state buffers at most {keep} completed pairs")
        if state.past_actions.shape != state.past_latents.shape[:2]:
            raise ValueError("buffered latents and actions disagree")
        if state.past_actions.numel():
            self._actions(state.past_actions, state.past_actions.shape)
        if state.history.shape != (state.latent.shape[0], 1, self.config.dynamics.width):
            raise ValueError("history must be B,1,width")

    def start(self, z):
        self._latents(z)
        if z.shape[1] != 1:
            raise ValueError("start takes exactly one initial frame")
        batch, device, dtype = z.shape[0], z.device, z.dtype
        return WindowPredictiveState(
            z.clone(), z.new_zeros(batch, 0, self.config.encoder.latent_dim),
            torch.zeros(batch, 0, dtype=torch.long, device=device),
            z.new_zeros(batch, 1, self.config.dynamics.width), 0)

    def readout(self, z, history):
        return self.agent_readout(torch.cat((z[:, :, 0], history), -1))

    def features(self, state):
        return self.readout(state.latent, state.history)

    def _window(self, latents, actions):
        """One source call on a bounded window, positions reset to 0:len as `rollout` does."""
        if latents.shape[1] > self.context:
            raise ValueError("source window exceeds its position table")
        one_hot = nn.functional.one_hot(actions, self.config.dynamics.n_actions).to(latents.dtype)
        hidden = self.predictor(latents, self.action_encoder(one_hot))
        flat = self.predictor_projector(hidden.flatten(0, 1))
        return flat.reshape(hidden.shape[0], hidden.shape[1], -1), hidden

    def teacher(self, z, actions, *, state=None, backend=None):
        """Aligned next-latent prediction over every completed pair.

        At the training length the whole window is one parallel call. Longer eval scans roll
        the window instead of extrapolating the three learned positions.
        """
        self._latents(z)
        pairs = z.shape[1] - 1
        self._actions(actions, (z.shape[0], pairs))
        if state is not None:
            self.validate_state(state)
            if not torch.equal(state.latent, z[:, :1]):
                raise ValueError("continuing teacher scan must begin at the state's current latent")
        if pairs == 0:
            return TransformerTeacherOutput(z[:, :0], z.new_zeros(z.shape[0], 0, self.config.dynamics.width),
                                            state if state is not None else self.start(z[:, :1]))
        inputs = z[:, :-1, 0]
        # A continuation resumes from the buffered pairs; dropping them would restart the
        # window and make chunked evaluation disagree with an uninterrupted scan.
        prior_latents = state.past_latents if state is not None else inputs[:, :0]
        prior_actions = state.past_actions if state is not None else actions[:, :0]
        offset = prior_latents.shape[1]
        full_inputs = torch.cat((prior_latents, inputs), 1)
        full_actions = torch.cat((prior_actions, actions), 1)
        if offset + pairs <= self.context:
            # What upstream does: one parallel causal pass over the whole window. Splitting
            # it into per-step windows would be a different objective, not an optimization --
            # the prediction BatchNorm would see several smaller batches and update several
            # times, and dropout would be drawn once per call.
            step_predicted, step_hidden = self._window(full_inputs, full_actions)
            predicted, hidden = step_predicted[:, offset:], step_hidden[:, offset:]
        else:
            # Past the position table the source slides its window and resets positions,
            # so a long evaluation scan must roll rather than extrapolate.
            rolled_predicted, rolled_hidden = [], []
            for end in range(offset + 1, offset + pairs + 1):
                start = max(0, end - self.context)
                step_predicted, step_hidden = self._window(full_inputs[:, start:end],
                                                           full_actions[:, start:end])
                rolled_predicted.append(step_predicted[:, -1:])
                rolled_hidden.append(step_hidden[:, -1:])
            predicted, hidden = torch.cat(rolled_predicted, 1), torch.cat(rolled_hidden, 1)
        keep = self.context - 1
        final = WindowPredictiveState(z[:, -1:],
                                      full_inputs[:, -keep:].clone() if keep else full_inputs[:, :0],
                                      full_actions[:, -keep:].clone() if keep else full_actions[:, :0],
                                      hidden[:, -1:].clone(), (0 if state is None else state.step) + pairs)
        return TransformerTeacherOutput(predicted.unsqueeze(2), hidden, final)

    def _transition(self, state, action):
        self.validate_state(state)
        self._actions(action, (state.latent.shape[0], 1))
        latents = torch.cat((state.past_latents, state.latent[:, :, 0]), 1)
        actions = torch.cat((state.past_actions, action), 1)
        predicted, hidden = self._window(latents, actions)
        keep = self.context - 1
        return (predicted[:, -1:].unsqueeze(2), hidden[:, -1:],
                latents[:, -keep:] if keep else latents[:, :0],
                actions[:, -keep:] if keep else actions[:, :0])

    def _streaming_mode(self):
        """One step at a time is evaluation. Training mode would let this update the
        prediction BatchNorm from a single row and keep dropout live, so a rollout would
        silently modify the model and stop being reproducible."""
        if any(isinstance(m, nn.BatchNorm1d) and m.training
               for m in self.predictor_projector.modules()):
            raise RuntimeError("predictor_normalization: streaming requires fixed BN statistics (eval mode)")
        if self.predictor.training or self.action_encoder.training:
            raise RuntimeError("predictor_dropout: streaming requires eval mode")

    def advance(self, state, action):
        self._streaming_mode()
        latent, history, past_latents, past_actions = self._transition(state, action)
        result = WindowPredictiveState(latent, past_latents.clone(), past_actions.clone(),
                                       history.clone(), state.step + 1)
        return result, self.features(result)

    def observe_latent(self, state, action, z_next):
        """The same transition; only the accepted successor is replaced by truth.

        Observed and generated branches therefore share exactly the same predictor output.
        """
        self._streaming_mode()
        self._latents(z_next)
        if z_next.shape != state.latent.shape:
            raise ValueError("observed successor must match one current latent")
        _, history, past_latents, past_actions = self._transition(state, action)
        result = WindowPredictiveState(z_next.clone(), past_latents.clone(), past_actions.clone(),
                                       history.clone(), state.step + 1)
        return result, self.features(result)


@dataclass(frozen=True)
class TransformerTeacherOutput:
    predicted: "torch.Tensor"
    features: "torch.Tensor"
    state: "WindowPredictiveState"

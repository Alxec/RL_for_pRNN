"""Actor-critic model definitions used by the RL training code.

``ACModelSR`` is the primary model family: it consumes a spatial
representation (usually the hidden state of a trained pRNN) and can optionally
combine it with a visual embedding. The Theta/Rollout variants are retained for
future work on Theta-pRNNs and are not part of the current active workflow.
"""

# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import torch_ac


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RecACModel(nn.Module, torch_ac.RecurrentACModel):
    """Recurrent image-based actor-critic model.

    Images are encoded by a convolutional network and passed through the
    supplied recurrent cell before actor and critic heads. This model has not
    yet been used in the active experiments, but is retained for future
    recurrence-based RL work.
    """

    def __init__(self, obs_space, action_space, cell, memory_size=300, with_obs=False, rgb=True, with_HD=False):
        super().__init__()

        # Decide which components are enabled
        self.memorysize = memory_size
        self.with_obs = with_obs
        self.rgb = rgb
        self.with_HD = with_HD

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        self.memory_rnn = cell(self.image_embedding_size, self.memorysize)

        # Define embedding size
        if self.with_obs:
            self.embedding_size = self.image_embedding_size + self.memorysize
        else:
            self.embedding_size = self.memorysize
        if self.with_HD:
            self.embedding_size += 1

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs, memory, noise, **kwargs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        if self.rgb:
            x /= 255
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        memory, _ = self.memory_rnn(x, noise, memory)

        if self.with_obs:
            embedding = torch.cat((x, memory), dim=1)
        else:
            embedding = memory

        if self.with_HD:
            embedding = torch.cat((embedding, obs.direction.unsqueeze(dim=1)), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
    

class ACModel(nn.Module, torch_ac.ACModel):
    """Feed-forward actor-critic model for visual observations.

    A convolutional image embedding, optionally augmented with one-hot head
    direction, is consumed by separate actor and critic heads.
    """

    def __init__(self, obs_space, action_space, with_HD=True, rgb=True):
        super().__init__()
        self.with_HD = with_HD
        self.rgb = rgb
        self.act_dim = action_space.n
        self.define_model(obs_space)

        # Initialize parameters correctly
        self.apply(init_params)

    def define_model(self, obs_space):
        # Define image embedding
        self.CV(obs_space)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    @property
    def embedding_size(self):
        if self.with_HD:
            return self.image_embedding_size + 4
        else:
            return self.image_embedding_size
    
    def CV(self, obs_space):
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        if n<7 or m<7:
            last_kernel_size = (1,1)
            self.image_embedding_size = ((n-1)//2-1)*((m-1)//2-1)*64
        else:
            last_kernel_size = (2,2)
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, last_kernel_size),
            nn.ReLU()
        )

    def forward(self, obs, **kwargs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        if self.rgb:
            x /= 255
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.with_HD:
            onehot_HD = torch.nn.functional.one_hot(obs.direction.long(), num_classes=4).float()
            embedding = torch.cat((x, onehot_HD), dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class ACModelSR(ACModel):
    """Primary actor-critic model operating on a spatial representation.

    The spatial representation is usually a hidden state from a trained pRNN,
    so the policy operates in the world model's latent space. It can also
    concatenate a visual embedding and one-hot head direction in parallel.
    """

    def __init__(self, obs_space, action_space, SR_size=-1, with_CV=True, 
                 rgb=True, with_HD=True):
        self.with_CV = with_CV
        self.SR_single = SR_size # if SRs are not used, the arg should be -1
        super(ACModelSR, self).__init__(obs_space, action_space, 
                                        with_HD=with_HD, rgb=rgb)

    @property
    def SR_size(self):
        if self.with_HD:
            return self.SR_single + 4
        else:
            return self.SR_single

    @property
    def embedding_size(self):
        return self.image_embedding_size + self.SR_size
    
    def CV(self, obs_space):
        if self.with_CV:
            super().CV(obs_space)
        else:
            self.image_embedding_size = 0

    def forward(self, obs, SR, **kwargs):
        if self.with_CV:
            x = obs.image.transpose(1, 3).transpose(2, 3)
            if self.rgb:
                x /= 255
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)


        onehot_HD = torch.nn.functional.one_hot(obs.direction.long(), num_classes=4).float()

        if self.with_HD:
            if self.with_CV:
                embedding = torch.cat((x, SR, onehot_HD), dim=1)
            else:
                embedding = torch.cat((SR, onehot_HD), dim=1)
        else:
            if self.with_CV:
                embedding = torch.cat((x, SR), dim=1)
            else:
                embedding = SR

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class PredictiveVisualEncoder(nn.Module):
    """Expose an encoder saved with a pRNN as an RL visual backbone.

    This adapter deliberately follows the corresponding Miniworld Shell:
    VAE uses ``encode`` then ``reparameterize``; the contrastive encoder sees
    a one-frame sequence; pRNN-AE uses its CNN directly.  Consequently the
    policy consumes the same latent ``z`` that the predictive model uses,
    rather than a separately implemented MiniGrid-style convolutional trunk.
    """

    def __init__(self, encoder: nn.Module, *, encoder_type: str):
        super().__init__()
        if encoder_type not in {"autoencoder", "vae", "contrastive"}:
            raise ValueError(f"Unsupported predictive visual encoder type: {encoder_type!r}.")
        self.encoder = encoder
        self.encoder_type = encoder_type

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "vae":
            mu, log_var = self.encoder.encode(images)
            features = self.encoder.reparameterize(mu, log_var)
        elif self.encoder_type == "contrastive":
            features = self.encoder.encode(images.unsqueeze(1)).squeeze(1)
        else:
            features = self.encoder(images)
        return features.flatten(start_dim=1)


class MiniworldCNNEncoder(nn.Module):
    """Trainable visual encoder matching pRNN's 64x64 CNN encoders.

    This is the deterministic encoder used when an RL visual baseline is not
    borrowing a trained pRNN encoder.  Its convolution, flattening, latent
    projection, activation, and Xavier-normal initialisation match the
    autoencoder/contrastive Miniworld encoders in ``pRNN``.  Keep its
    configuration aligned with ``ConfigsRNN/config.yaml:encoder``.
    """

    def __init__(self, in_channels, latent_dim, net_config):
        super().__init__()
        output_channels, kernel_sizes, strides, paddings, _ = net_config
        modules = []
        for out_channels, kernel_size, stride, padding in zip(
                output_channels, kernel_sizes, strides, paddings
        ):
            modules.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(),
            ))
            in_channels = out_channels
        modules.extend((
            nn.Flatten(),
            nn.Linear(output_channels[-1] * 16 * 16, latent_dim),
            nn.ReLU(),
        ))
        self.encoder = nn.Sequential(*modules)
        self.latent_dim = int(latent_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.encoder.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


class SquashedDiagGaussian:
    """Independent Gaussian policy transformed exactly into a Box space."""

    def __init__(
            self,
            mean: torch.Tensor,
            log_std: torch.Tensor,
            low: torch.Tensor,
            high: torch.Tensor,
    ):
        self.mean = mean
        self.std = log_std.exp().expand_as(mean)
        self.base_dist = Normal(mean, self.std)
        self.low = low
        self.high = high
        self.scale = (high - low) / 2
        self.bias = (high + low) / 2

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.scale + self.bias

    def sample(self) -> torch.Tensor:
        return self._squash(self.base_dist.sample())

    def rsample(self) -> torch.Tensor:
        return self._squash(self.base_dist.rsample())

    def mode(self) -> torch.Tensor:
        return self._squash(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # Undo the affine/tanh transforms.  The clamp only protects the
        # inverse transform at an exact Box boundary.
        normalized = ((action - self.bias) / self.scale).clamp(-0.999999, 0.999999)
        raw_action = torch.atanh(normalized)
        correction = torch.log(self.scale) + torch.log(1 - normalized.square() + 1e-6)
        return (self.base_dist.log_prob(raw_action) - correction).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # A transformed Gaussian has no simple closed-form entropy.  This
        # reparameterized Monte-Carlo estimate remains differentiable.
        action = self.rsample()
        return -self.log_prob(action)


class ContinuousACModel(nn.Module, torch_ac.ACModel):
    """Visual actor-critic with a bounded continuous action distribution."""

    def __init__(
            self,
            obs_space,
            action_space,
            with_HD=False,
            rgb=True,
            visual_encoder: nn.Module | None = None,
            freeze_visual_encoder=False,
            numHDs=4,
    ):
        super().__init__()
        if not hasattr(action_space, "shape"):
            raise TypeError("ContinuousACModel requires a Gymnasium Box action space.")
        self.with_HD = with_HD
        self.rgb = rgb
        self.numHDs = int(numHDs)
        self.act_dim = int(np.prod(action_space.shape))
        self.visual_encoder = visual_encoder
        self.freeze_visual_encoder = bool(freeze_visual_encoder)
        self.register_buffer("action_low", torch.as_tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_space.high, dtype=torch.float32))
        self.define_model(obs_space)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))
        # Do not recurse through a supplied pRNN encoder: it is pretrained
        # state, not a newly initialized policy layer.
        self.actor.apply(init_params)
        self.critic.apply(init_params)
        if self.visual_encoder is not None and self.freeze_visual_encoder:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad_(False)
            self.visual_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.visual_encoder is not None and self.freeze_visual_encoder:
            self.visual_encoder.eval()
        return self

    @property
    def embedding_size(self):
        return self.image_embedding_size + (self.numHDs if self.with_HD else 0)

    def _image_tensor(self, obs):
        image = obs.image.transpose(1, 3).transpose(2, 3)
        return image / 255 if self.rgb else image

    def CV(self, obs_space):
        if self.visual_encoder is None:
            raise ValueError(
                "Continuous Miniworld visual policies require a pRNN visual encoder. "
                "Load a trained Miniworld pRNN and set inputs.visual_encoder=predictive_net."
            )

        height, width, channels = obs_space["image"]
        sample = torch.zeros((1, channels, height, width), dtype=torch.float32)
        with torch.no_grad():
            self.image_embedding_size = int(self.visual_encoder(sample).shape[-1])

    def _visual_embedding(self, obs):
        image = self._image_tensor(obs)
        return self.visual_encoder(image)

    def _onehot_hd(self, obs):
        hd = obs.HD if hasattr(obs, "HD") else obs.direction
        return F.one_hot(hd.long(), num_classes=self.numHDs).float()

    def define_model(self, obs_space):
        self.CV(obs_space)
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _distribution(self, embedding):
        return SquashedDiagGaussian(
            self.actor(embedding), self.log_std, self.action_low, self.action_high
        )

    def forward(self, obs, **kwargs):
        embedding = self._visual_embedding(obs)
        if self.with_HD:
            onehot_HD = self._onehot_hd(obs)
            embedding = torch.cat((embedding, onehot_HD), dim=1)
        return self._distribution(embedding), self.critic(embedding).squeeze(1)


class ContinuousACModelSR(ContinuousACModel):
    """Continuous-action policy with pRNN SR and optional visual features."""

    def __init__(
            self,
            obs_space,
            action_space,
            SR_size=-1,
            with_CV=True,
            rgb=True,
            with_HD=False,
            visual_encoder: nn.Module | None = None,
            freeze_visual_encoder=False,
            numHDs=4,
    ):
        self.with_CV = with_CV
        self.SR_single = SR_size
        super().__init__(
            obs_space,
            action_space,
            with_HD=with_HD,
            rgb=rgb,
            visual_encoder=visual_encoder,
            freeze_visual_encoder=freeze_visual_encoder,
            numHDs=numHDs,
        )

    @property
    def SR_size(self):
        return self.SR_single + (self.numHDs if self.with_HD else 0)

    @property
    def embedding_size(self):
        return self.image_embedding_size + self.SR_size

    def CV(self, obs_space):
        if self.with_CV:
            super().CV(obs_space)
        else:
            self.image_embedding_size = 0

    def forward(self, obs, SR, **kwargs):
        if self.with_CV:
            visual = self._visual_embedding(obs)
        if self.with_HD:
            onehot_HD = self._onehot_hd(obs)
            embedding = torch.cat((visual, SR, onehot_HD), dim=1) if self.with_CV else torch.cat((SR, onehot_HD), dim=1)
        else:
            embedding = torch.cat((visual, SR), dim=1) if self.with_CV else SR
        return self._distribution(embedding), self.critic(embedding).squeeze(1)


class ACModelTheta(ACModelSR):
    """Theta/Rollout actor-critic model for sequential pRNN outputs.

    This deferred model family handles a sequence of spatial representations,
    head directions, actions, and values from a Theta-pRNN (also called a
    Rollout pRNN). Do not treat it as part of the active baseline workflow.
    """

    def __init__(self, obs_space, action_space, SR_size=-1, with_CV=True, rgb=True,
                 k=1, V='single'):
        assert V in ['single', 'double', 'multi']
        self.V = V
        self.seq_length = k+1

        if V == 'single':
            self.V_size = 1
        else:
            self.V_size = self.seq_length

        super(ACModelTheta, self).__init__(obs_space, action_space, SR_size, with_CV, rgb)
        

    def define_model(self, obs_space):
        # Define image embedding
        self.CV(obs_space)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim * self.seq_length)
        )

        # Define critic's model
        if self.V == 'single':
            self.V_size = 1
        else:
            self.V_size = self.seq_length
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.V_size)
        )

    @property
    def SR_size(self):
        return self.SR_single * self.seq_length

    @property
    def embedding_size(self):
        return self.image_embedding_size + self.SR_size + \
            (4 + self.act_dim + self.V_size) * self.seq_length

    def forward(self, obs, SR, HDs, acts, values=None, **kwargs):
        SR = SR.reshape(-1, self.SR_size)
        if values==None:
            values = torch.zeros(HDs.shape)

        if self.with_CV:
            x = obs.image.transpose(1, 3).transpose(2, 3)
            if self.rgb:
                x /= 255
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)

        HDs = torch.nn.functional.one_hot(HDs.long(), num_classes=4).float()

        if self.with_CV:
            embedding = torch.cat((x, SR, HDs, acts, values), dim=1)
        else:
            embedding = torch.cat((SR, HDs, acts, values), dim=1)
        

        x = self.actor(embedding).reshape(-1, self.seq_length, self.act_dim)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value


class ACModelThetaShared(ACModelTheta):
    """Theta/Rollout variant with shared per-step features across a sequence.

    This experimental class currently uses no visual input and is retained for
    future Theta-pRNN/Rollout work.
    """

    def __init__(self, obs_space, action_space, SR_size=-1, k=1, V='single'):
        super(ACModelThetaShared, self).__init__(obs_space, action_space, SR_size,
                                                 with_CV=False, rgb=False, k=k, V=V) # No visual input (yet)
        

    def define_model(self, obs_space):
        # Define actor's model
        self.actor1 = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
        )
        self.actor2 = nn.Linear(64 * self.seq_length, self.act_dim * self.seq_length)

        # Define critic's model
        if self.V == 'single':
            self.V_size = 1
            self.critic = nn.Sequential(
                                        nn.Linear(self.embedding_size, 64),
                                        nn.Tanh(),
                                        nn.Linear(64, 1)
                                        )
        else:
            self.V_size = self.seq_length
            self.critic1 = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
            )
            self.critic2 = nn.Linear(64 * self.seq_length, self.V_size)

    @property
    def SR_size(self):
        return self.SR_single

    @property
    def embedding_size(self):
        return self.SR_size + 4 + self.act_dim + self.V_size

    def forward(self, obs, SR, HDs, acts, values=None, **kwargs):
        if values==None:
            values = torch.zeros(HDs.shape)

        HDs = torch.nn.functional.one_hot(HDs.long(), num_classes=4).float()
        acts = torch.nn.functional.one_hot(acts.long(), num_classes=self.act_dim).float()

        embedding = torch.cat((SR,
                               HDs,
                               acts,
                               values[:,:,None]), dim=-1)
        

        x = self.actor1(embedding).reshape(-1, 64 * self.seq_length)
        x = self.actor2(x).reshape(-1, self.seq_length, self.act_dim)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        if self.V == 'single':
            x = self.critic(embedding)
            value = x.mean(1)
        else:
            x = self.critic1(embedding).reshape(-1, 64 * self.seq_length)
            x = self.critic2(x)
            value = x.squeeze(1)

        return dist, value


class ACModelThetaSingle(ACModelTheta):
    """Theta/Rollout variant that emits a single action/value prediction.

    This experimental class currently uses no visual input and is retained for
    future Theta-pRNN/Rollout work.
    """

    def __init__(self, obs_space, action_space, SR_size=-1, k=1, V='single'):
        super(ACModelThetaSingle, self).__init__(obs_space, action_space, SR_size,
                                                 with_CV=False, rgb=False, k=k, V=V)
        

    def define_model(self, obs_space):
        # # Define actor's model
        # self.actor1 = nn.Sequential(
        #     nn.Linear(self.embedding_size, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 8),
        #     nn.Tanh()
        # )
        # self.actor2 = nn.Linear(8, self.act_dim)

        # self.critic1 = nn.Sequential(
        #     nn.Linear(self.embedding_size, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 8),
        #     nn.Tanh()
        # )

        # self.critic2 = nn.Linear(8, 1)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    @property
    def SR_size(self):
        return self.SR_single

    @property
    def embedding_size(self):
        return self.SR_size + 3

    def forward(self, obs, SR, HDs, acts, values=None, **kwargs):
        if values==None:
            values = torch.zeros(HDs.shape)
        HDs = torch.nn.functional.one_hot(HDs.long(), num_classes=4).float()
        embedding = torch.cat((SR,
                               HDs[:,:,None],
                               acts[:,:,None],
                               values[:,:,None]), dim=-1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)[:,0,:]
        value = x.squeeze(1)
        

        # x = self.actor1(embedding)
        # x = self.actor2(x)
        # dist = Categorical(logits=F.log_softmax(x, dim=-1))

        # x = self.critic1(embedding)
        # x = self.critic2(x[:,0,:])
        # value = x.squeeze(1)

        return dist, value

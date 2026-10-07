import gymnasium as gym
import torch
import numpy as np
from ratinabox.utils import get_distances_between

import RLutils
from .other import device
from RLutils.model import ACModel
from RLutils.spatial_strategies import PredictiveNetworkPastSR, PredictiveNetworkSR


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obss = RLutils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(RLutils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(RLutils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])


def move_prnn(prnn, device):
    """Move a pRNN and any Shell encoder (Miniworld) to ``device``."""
    prnn.pRNN.to(device)
    encoder = getattr(prnn.env_shell, "encoder", None)
    if isinstance(encoder, torch.nn.Module):
        encoder.to(device)


class ActorCriticAgent:
    """Roll out a trained actor-critic to collect pRNN evaluation trajectories.

    This repeats ``PredictivePPOAlgo._collect_single_step``: the actor gets the
    SR computed by the algorithm's own spatial strategy, with the same SR and
    HD timing as in training.  It serves both MiniGrid and continuous Miniworld.
    For Miniworld, ``position_agent`` (a ``MiniworldRandomAgent``) supplies the
    RatInABox position bins used by the pRNN's spatial decoding.
    """

    def __init__(self, action_space, acmodel, prnn, device, sr_strategy=None,
                 past_SR=True, position_agent=None):
        self.action_space = action_space
        self.acmodel = acmodel
        self.prnn = prnn
        self.device = device
        self.past_SR = bool(past_SR)
        if sr_strategy is None:
            # Unmasked pRNN SR, for callers without a PPO algorithm
            sr_strategy = (
                PredictiveNetworkPastSR(prnn, device) if self.past_SR
                else PredictiveNetworkSR(prnn, device)
            )
        self.sr_strategy = sr_strategy
        self.position_agent = position_agent
        self.continuous = isinstance(action_space, gym.spaces.Box)
        self.name = 'ActorCritic Agent'

    def _env_action(self, action):
        action = action.detach().cpu()
        return action.squeeze(0).numpy() if self.continuous else int(action.item())

    def _riab_position_bins(self, env, positions):
        """Bin Miniworld positions exactly as ``MiniworldRandomAgent`` does.

        Positions are converted to RatInABox coordinates, snapped to the
        nearest cell of the random agent's RatInABox environment, and returned
        as integer cell indices for the pRNN's spatial decoding.
        """
        positions = np.asarray(positions, dtype=float)
        riab_positions = np.stack(
            (positions[:, 0], env.env.size[1] - positions[:, 1]), axis=-1
        ) / 10
        riab_env = self.position_agent.Environment
        dx = riab_env.dx
        coord = riab_env.flattened_discrete_coords
        dist = get_distances_between(riab_positions, coord)
        return ((coord[dist.argmin(axis=1)] - dx/2) / dx).astype(int)

    @staticmethod
    def _prnn_state(hd):
        return None if hd is None else {"agent_dir": np.float32(hd)}

    def getObservations(self, env, tsteps, reset=True, includeRender=False,
                        discretize=False, **kwargs):
        if not reset:
            raise NotImplementedError("On-policy pRNN evaluation starts from an environment reset.")
        move_prnn(self.prnn, self.device)
        _, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)
        render = False

        obs = [None for t in range(tsteps+1)]
        act = [None for t in range(tsteps)]

        obs[0] = env.reset()
        self.prnn.reset_state(device=self.device)
        SR = self.sr_strategy.initialize_SR(obs=obs[0])

        state = {'agent_pos': np.resize(env.get_agent_pos(),(1,2)), 
                 'agent_dir': env.get_agent_dir(),
                }
        
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
        
        for aa in range(tsteps):
            preprocessed_obs = preprocess_obss([obs[aa]], device=self.device)
            with torch.no_grad():
                dist, _ = self.acmodel(preprocessed_obs, SR=SR)
            act[aa] = self._env_action(dist.sample())
            past_hd = env.get_agent_dir()

            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'], 
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())

            hd = past_hd if self.past_SR else env.get_agent_dir()
            SR = self.sr_strategy.compute_SR(
                action=act[aa], past_obs=obs[aa], new_obs=obs[aa+1],
                state=self._prnn_state(hd),
            )

            if includeRender:
                render[aa+1] = env.render(mode=None)
        
        self.prnn.reset_state(device=self.device)
        # pRNN analyses run on the CPU
        move_prnn(self.prnn, "cpu")

        if self.continuous:
            act = np.stack(act)
            if discretize:
                if self.position_agent is None:
                    raise ValueError("Discretised Miniworld positions require position_agent.")
                state['pos_continuous'] = state['agent_pos'].copy()
                state['agent_pos'] = self._riab_position_bins(env, state['agent_pos'])
        else:
            act = np.array(act).reshape(-1)
        return obs, act, state, render

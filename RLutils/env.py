import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import (
    DictObservationSpaceWrapper,
    DirectionObsWrapper,
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    OneHotPartialObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    ReseedWrapper,
    SymbolicObsWrapper,
    ViewSizeWrapper,
)
from functools import partial

import prnn.environments.Lroom
from prnn.utils.CANNNet import CANNnet
from prnn.utils.Shell import (
    FaramaMinigridShell,
    MiniworldContrastiveShell,
    MiniworldShell,
    MiniworldVAEShell,
)
from prnn.utils.env import RGBImgPartialObsWrapper_HD_Farama

wrappers = {
    "ReseedWrapper": ReseedWrapper,
    "ImgObsWrapper": ImgObsWrapper,
    "OneHotPartialObsWrapper": OneHotPartialObsWrapper,
    "RGBImgObsWrapper": RGBImgObsWrapper,
    "RGBImgPartialObsWrapper": RGBImgPartialObsWrapper,
    "RGBImgPartialObsWrapper_HD": RGBImgPartialObsWrapper_HD_Farama,
    "FullyObsWrapper": FullyObsWrapper,
    "DictObservationSpaceWrapper": DictObservationSpaceWrapper,
    "FlatObsWrapper": FlatObsWrapper,
    "ViewSizeWrapper": ViewSizeWrapper,
    "DirectionObsWrapper": DirectionObsWrapper,
    "SymbolicObsWrapper": SymbolicObsWrapper         
}

#replace lambda function so i can pickle pNet
def episode_video_trigger(episode, vid_n_episodes):
    return episode % vid_n_episodes == 0


MINIWORLD_SHELL_TYPES = {
    "autoencoder": MiniworldShell,
    "vae": MiniworldVAEShell,
    "contrastive": MiniworldContrastiveShell,
}


def miniworld_shell_type(shell):
    """Return the registered RL Shell type for a loaded Miniworld pRNN."""
    # Check encoder-specific subclasses before their MiniworldShell base class.
    for shell_type in ("vae", "contrastive", "autoencoder"):
        if isinstance(shell, MINIWORLD_SHELL_TYPES[shell_type]):
            return shell_type
    supported = ", ".join(sorted(MINIWORLD_SHELL_TYPES))
    raise TypeError(
        f"Unsupported Miniworld Shell {type(shell).__name__}; supported types: {supported}."
    )


def make_miniworld_shell(env, act_enc, env_key, HDbins, shell_type="autoencoder", encoder=None):
    """Wrap a Miniworld Gym environment in the requested pRNN Shell.

    This RL-side registry keeps pRNN free of RL-specific construction logic.
    Add future Shell families here without changing the pRNN package.
    """
    try:
        shell_class = MINIWORLD_SHELL_TYPES[shell_type]
    except KeyError as error:
        supported = ", ".join(sorted(MINIWORLD_SHELL_TYPES))
        raise ValueError(
            f"Unknown Miniworld Shell type {shell_type!r}; supported types: {supported}."
        ) from error

    if shell_class is MiniworldShell:
        return shell_class(env, act_enc, env_key, HDbins=HDbins)
    if encoder is None:
        raise ValueError(
            f"Miniworld Shell type {shell_type!r} requires its trained encoder."
        )
    if shell_class is MiniworldVAEShell:
        return shell_class(env, act_enc, env_key, vae=encoder, HDbins=HDbins)
    return shell_class(env, act_enc, env_key, encoder=encoder, HDbins=HDbins)

def make_minigrid_env(
             env_key,
             input_type,
             spatial_config,
             seed=0,
             vid_folder='',
             vid_n_episodes=0,
             wrapper=None,
             render_mode='rgb_array',
             act_enc=None,
             **kwargs
             ):
    """Build the established discrete Farama MiniGrid RL environment."""

    env = gym.make(env_key, render_mode=render_mode)

    if input_type == 'Visual_FO':
        # Not RGB one here because we want RL agent to have as much info as possible
        env = FullyObsWrapper(env)

    elif spatial_config.predictive_net or 'PO' in input_type:
        # The same RGB wrapper is used for comparability whenever partial observation is needed
        env = RGBImgPartialObsWrapper_HD_Farama(env, tile_size=1)
    
    else:
        # For the cases without any visual input
        env = HDObsWrapper(env)

    if wrapper:
        env = wrappers[wrapper](env, **kwargs)

    #Below I replaced the lambda function. This allows me to pickle pNet without errors
    if vid_n_episodes:
        #env = RecordVideo(env, video_folder=vid_folder, episode_trigger=lambda x: x%vid_n_episodes == 0)
        trigger_func = partial(episode_video_trigger, vid_n_episodes=vid_n_episodes)
        env = RecordVideo(env, video_folder=vid_folder, episode_trigger=trigger_func)
    
    env.reset(seed=seed)
    env = FaramaMinigridShell(env, act_enc, env_key)

    return env


def make_miniworld_env(
        env_key,
        input_type,
        seed=0,
        vid_folder='',
        vid_n_episodes=0,
        render_mode='rgb_array',
        act_enc='ContSpeedOnehotHDMiniworld',
        continuous_actions=True,
        hd_bins=12,
        shell_type='autoencoder',
        shell_encoder=None,
        with_HD=False,
        **kwargs,
):
    """Create a Miniworld Shell for continuous-action visual RL.

    Miniworld's ordinary observation is the agent camera (partial
    observation).  ``Visual_FO`` instead uses a top-down RGB observation;
    rendering uses the same viewpoint, so recorded videos match the policy
    input mode.
    """
    import prnn.environments.Miniworld  # Register project Miniworld tasks.

    view = 'top' if input_type == 'Visual_FO' else 'agent'
    env = gym.make(
        env_key,
        continuous=continuous_actions,
        render_mode=render_mode,
        view=view,
        obs_width=64,
        obs_height=64,
        window_width=64,
        window_height=64,
        **kwargs,
    )
    if input_type == 'Visual_FO':
        env = MiniworldFullyObservableWrapper(env)
    if with_HD:
        env = MiniworldHeadDirectionObsWrapper(env, hd_bins=hd_bins)

    if vid_n_episodes:
        # Insert this below RecordVideo so it changes only rendered frames,
        # never the observation delivered to the actor-critic.
        env = MiniworldVideoInfoOverlayWrapper(
            env,
            hd_bins=hd_bins,
            ac_receives_hd=with_HD,
        )
        trigger_func = partial(episode_video_trigger, vid_n_episodes=vid_n_episodes)
        env = RecordVideo(env, video_folder=vid_folder, episode_trigger=trigger_func)

    env.reset(seed=seed)
    return make_miniworld_shell(
        env,
        act_enc,
        env_key,
        HDbins=hd_bins,
        shell_type=shell_type,
        encoder=shell_encoder,
    )


class ResetWrapper(Wrapper):
    """
    Wrapper to return a single dictionary of observation, not a tuple with empty second element.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]
    

class HDObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    Including direction information (HD)
    """

    def __init__(self, env):
        super().__init__(env)
        HD_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "HD": HD_space}
        )

            
    def observation(self, obs):

        return {
            'mission': obs['mission'],
            'HD': obs['direction']
        }


class MiniworldFullyObservableWrapper(ObservationWrapper):
    """Replace Miniworld's agent-view observation with its top-down RGB view."""

    def __init__(self, env):
        super().__init__(env)
        # Miniworld uses the same framebuffer dimensions for observations and
        # top-down rendering when no framebuffer is supplied.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, _obs):
        return self.env.unwrapped.render_top_view()


class MiniworldHeadDirectionObsWrapper(ObservationWrapper):
    """Attach Miniworld's discretised continuous HD to image observations.

    The binning deliberately matches ``ContSpeed*HDMiniworld`` in pRNN's
    action encodings.  MiniGrid's four cardinal directions are unrelated to
    this configurable continuous-world representation.
    """

    def __init__(self, env, hd_bins):
        super().__init__(env)
        self.hd_bins = int(hd_bins)
        if self.hd_bins < 1:
            raise ValueError("Miniworld hd_bins must be at least one.")
        self.observation_space = spaces.Dict({
            "image": env.observation_space,
            "HD": spaces.Discrete(self.hd_bins),
        })

    def observation(self, obs):
        hd = float(self.env.unwrapped.agent.dir) % (2 * np.pi)
        hd_bin = min(int(hd / (2 * np.pi) * self.hd_bins), self.hd_bins - 1)
        return {"image": obs, "HD": hd_bin}


class MiniworldVideoInfoOverlayWrapper(Wrapper):
    """Add agent-state metadata beside rendered Miniworld video frames.

    This is deliberately a rendering-only wrapper.  It is installed below
    Gymnasium's :class:`RecordVideo`, after any visual/HD observation wrappers,
    so it does not alter the image or HD bin received by the actor-critic.
    """

    display_scale = 3
    panel_width = 260

    def __init__(self, env, *, hd_bins, ac_receives_hd):
        super().__init__(env)
        self.hd_bins = int(hd_bins)
        self.ac_receives_hd = bool(ac_receives_hd)
        if self.hd_bins < 1:
            raise ValueError("hd_bins must be at least one.")

    def info_lines(self):
        """Return the values shown in the side panel for the current frame."""
        agent = getattr(self.env.unwrapped, "agent", None)
        position = getattr(agent, "pos", None)
        direction = getattr(agent, "dir", None)

        if position is None:
            position_text = "Agent position: unavailable"
        else:
            position = np.asarray(position, dtype=np.float64).reshape(-1)
            if position.size >= 2:
                position_text = f"Agent position: ({position[0]:.2f}, {position[1]:.2f})"
            else:
                position_text = "Agent position: unavailable"

        if direction is None:
            direction_text = "Agent direction: unavailable"
            hd_text = "AC HD bin: unavailable" if self.ac_receives_hd else "AC HD bin: not provided"
        else:
            direction = float(direction) % (2 * np.pi)
            direction_text = (
                f"Agent direction: {direction:.3f} rad ({np.degrees(direction):.1f} deg)"
            )
            if self.ac_receives_hd:
                hd_bin = min(int(direction / (2 * np.pi) * self.hd_bins), self.hd_bins - 1)
                hd_text = f"AC HD bin: {hd_bin} / {self.hd_bins - 1}"
            else:
                hd_text = "AC HD bin: not provided"

        return (position_text, direction_text, hd_text)

    @staticmethod
    def _font():
        try:
            return ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            return ImageFont.load_default()

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None

        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[-1] < 3:
            raise ValueError("Miniworld video rendering must produce an RGB frame.")
        frame_array = np.asarray(frame_array[..., :3], dtype=np.uint8)

        image = Image.fromarray(frame_array)
        resampling = getattr(Image, "Resampling", Image).NEAREST
        image = image.resize(
            (image.width * self.display_scale, image.height * self.display_scale),
            resample=resampling,
        )
        canvas = Image.new(
            "RGB",
            (image.width + self.panel_width, max(image.height, 112)),
            color=(24, 28, 34),
        )
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = self._font()
        x = image.width + 14
        draw.text((x, 14), "Miniworld agent", fill=(240, 240, 240), font=font)
        for line_index, line in enumerate(self.info_lines(), start=1):
            draw.text((x, 14 + line_index * 25), line, fill=(196, 220, 255), font=font)
        return np.asarray(canvas)

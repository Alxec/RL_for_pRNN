import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import *
from functools import partial

from prnn.utils.CANNNet import CANNnet
from prnn.utils.Shell import FaramaMinigridShell

wrappers = {
    "ReseedWrapper": ReseedWrapper,
    "ActionBonus": ActionBonus,
    "StateBonus": StateBonus,
    "ImgObsWrapper": ImgObsWrapper,
    "OneHotPartialObsWrapper": OneHotPartialObsWrapper,
    "RGBImgObsWrapper": RGBImgObsWrapper,
    "RGBImgPartialObsWrapper": RGBImgPartialObsWrapper,
    "RGBImgPartialObsWrapper_HD": RGBImgPartialObsWrapper_HD,
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

def make_env(
             env_key,
             input_type,
             seed=0,
             vid_folder='',
             vid_n_episodes=0,
             wrapper=None,
             render_mode='rgb_array',
             act_enc=None,
             **kwargs
             ):
    env = gym.make(env_key, render_mode=render_mode)

    if input_type == 'Visual_FO':
        # Not RGB one here because we want RL agent to have as much info as possible
        env = FullyObsWrapper(env)

    elif 'pRNN' in input_type or 'PO' in input_type:
        # The same RGB wrapper is used for comparability whenever partial observation is needed
        env = RGBImgPartialObsWrapper_HD(env, tile_size=1)
    
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

    # if 'pRNN' in input_type or 'CANN' in input_type or 'Intrinsic' in input_type:
    #     env = FaramaMinigridShell(env, act_enc, env_key)
    # else:
    #     env = ResetWrapper(env)
    #     env.reset(seed=seed)

    return env


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
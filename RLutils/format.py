import os
import json
import numpy
import torch
# import torch_ac
from torch_ac.utils import DictList
import gymnasium as gym


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            # return torch_ac.DictList({
            return DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Miniworld image observations with a configurable, discretised HD.
    # This must precede MiniGrid's symbolic ``HD`` case below.
    elif (
        isinstance(obs_space, gym.spaces.Dict)
        and "image" in obs_space.spaces
        and "HD" in obs_space.spaces
    ):
        obs_space = {"image": obs_space.spaces["image"].shape, "HD": 1}

        def preprocess_obss(obss, device=None):
            return DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "HD": preprocess_int([obs["HD"] for obs in obss], device=device),
            })

    # Check if it is a MiniGrid symbolic observation space with HD
    elif isinstance(obs_space, gym.spaces.Dict) and "HD" in obs_space.spaces.keys():
        obs_space = {"direction": 1}

        def preprocess_obss(obss, device=None):
            # return torch_ac.DictList({
            return DictList({
                "direction": preprocess_int([obs["HD"] for obs in obss], device=device),
            })

    # Check if it is a MiniGrid observation space with images
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape,
                     "direction": 1}

        def preprocess_obss(obss, device=None):
            # return torch_ac.DictList({
            return DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "direction": preprocess_int([obs["direction"] for obs in obss], device=device),
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_int(integer, device=None):
    integer = numpy.array(integer)
    return torch.tensor(integer, device=device, dtype=torch.uint8)

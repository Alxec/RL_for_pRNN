"""
Spatial representation strategies for RL algorithms.

This module provides various spatial representation strategies that can be used
to encode spatial information for RL agents, including predictive networks,
place cells, and continuous attractor neural networks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import torch
import numpy as np

from RLutils.pc import FakePlaceCells
from prnn.utils.CANNNet import CANNnet

logger = logging.getLogger(__name__)


# ============================================================================
# Spatial Representation Strategies
# ============================================================================

class SpatialRepresentationStrategy(ABC):
    """Abstract base class for spatial representation strategies."""
    
    @abstractmethod
    def compute_SR(self, action: np.ndarray, obs: Dict) -> torch.Tensor:
        """
        Compute spatial representation for current state.
        
        Args:
            action: Action taken
            obs: Current observation
        
        Returns:
            Spatial representation tensor [1, SR_dim]
        """
        pass
    
    @abstractmethod
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        """
        Initialize spatial representation at start of episode.
        
        Args:
            obs: Initial observation
        
        Returns:
            Initial SR tensor
        """
        pass
    
    def get_SR_size(self, initial_SR: torch.Tensor) -> int:
        """Get size of spatial representation."""
        return initial_SR.shape[-1]


class NoSpatialRepresentation(SpatialRepresentationStrategy):
    """No spatial representation - returns empty tensors."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def compute_SR(self, **kwargs) -> torch.Tensor:
        return torch.tensor([], device=self.device).unsqueeze(dim=0)
    
    def initialize_SR(self, **kwargs) -> torch.Tensor:
        return torch.tensor([], device=self.device).unsqueeze(dim=0)


class PredictiveNetworkSR(SpatialRepresentationStrategy):
    """Spatial representation using pRNN that uses a(t-1) action."""
    
    def __init__(self, predictive_net: Any, device: torch.device, 
                 mask_indices: Optional[List[int]] = None):
        self.pN = predictive_net
        self.device = device
        self.mask_indices = mask_indices
        self.pN.pRNN.to(device)
    
    def compute_SR(
            self,
            action: np.ndarray,
            new_obs: Dict,
            **kwargs
        ) -> torch.Tensor:
        """Compute SR using standard predictive network."""
        obs_list = [new_obs, new_obs]
        
        obs_pN, act_pN = self.pN.env_shell.env2pred(obs_list, action)
        obs_pN = obs_pN.to(self.device)
        act_pN = act_pN.to(self.device)
        
        with torch.no_grad():
            SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        
        return self._apply_mask(SR)
    
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        """Initialize SR for predictive network."""
        obs_pN, act_pN = self.pN.env_shell.env2pred([obs, obs], np.array([0]))
        act_pN = torch.zeros_like(act_pN)
        obs_pN = obs_pN.to(self.device)
        act_pN = act_pN.to(self.device)
        
        with torch.no_grad():
            SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        return SR
    
    def _apply_mask(self, SR: torch.Tensor) -> torch.Tensor:
        """Apply masking to spatial representation if configured."""
        if self.mask_indices is not None:
            SR[0, self.mask_indices] = 0
        return SR
    
    def last_SR(self, SR: torch.Tensor, **kwargs) -> torch.Tensor:
        return SR


class PredictiveNetworkPastSR(PredictiveNetworkSR):
    """Spatial representation using pRNN that uses a(t) action."""
    
    def compute_SR(
            self,
            action: np.ndarray,
            past_obs: Dict,
            **kwargs
        ) -> torch.Tensor:
        """Compute SR using standard predictive network."""
        obs_list = [past_obs, past_obs]
        
        obs_pN, act_pN = self.pN.env_shell.env2pred(obs_list, action)
        obs_pN = obs_pN.to(self.device)
        act_pN = act_pN.to(self.device)
        
        with torch.no_grad():
            SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        
        return self._apply_mask(SR)
    
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        return torch.zeros((1, self.pN.hidden_size), device=self.device)
    
    def last_SR(self, det_action: np.ndarray, obs: Dict, **kwargs) -> torch.Tensor:
        """Compute last SR after episode ends."""
        return self.compute_SR(det_action, obs)


class ThetaCyclePredictiveNetworkSR(SpatialRepresentationStrategy):
    """Spatial representation using theta cycle predictive network."""
    
    def __init__(self, predictive_net: Any, device: torch.device, 
                 mask_indices: Optional[List[int]] = None):
        self.pN = predictive_net
        self.device = device
        self.mask_indices = mask_indices
        self.theta_k = self.pN.pRNN.k + 1
        self.pN.pRNN.to(device)
    
    def compute_SR(self, action: np.ndarray, obs: Dict) -> torch.Tensor:
        """Compute SR using theta cycle predictive network."""
        obs_list = [obs] * (self.theta_k + 1)
        act_repeated = action.repeat(self.theta_k)
        
        obs_pN, act_pN = self.pN.env_shell.env2pred(obs_list, act_repeated)
        obs_pN = obs_pN.to(self.device)
        act_pN = act_pN.to(self.device)
        
        with torch.no_grad():
            SR = self.pN.predict(obs_pN, act_pN)[2][0]
        
        return self._apply_mask(SR)
    
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        """Initialize SR for theta cycle network."""
        return torch.zeros((1, self.pN.hidden_size), device=self.device)
    
    def _apply_mask(self, SR: torch.Tensor) -> torch.Tensor:
        """Apply masking to spatial representation if configured."""
        if self.mask_indices is not None:
            SR[0, self.mask_indices] = 0
        return SR


class PlaceCellsSR(SpatialRepresentationStrategy):
    """Spatial representation using place cells."""
    
    def __init__(self, place_cells: Any, device: torch.device, 
                 mask_indices: Optional[List[int]] = None,
                 pastSR: bool = False):
        self.PC = place_cells
        self.device = device
        self.mask_indices = mask_indices
        self.pastSR = pastSR
    
    def compute_SR(self, action: np.ndarray,
                   past_obs: Dict,
                   new_obs: Dict) -> torch.Tensor:
        """Compute SR using place cells."""
        if self.pastSR:
            obs = past_obs
        else:
            obs = new_obs
        SR = torch.tensor(
            self.PC.activation(obs.get('agent_pos')), 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(dim=0)
        
        return self._apply_mask(SR)
    
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        """Initialize SR for place cells."""
        return torch.zeros((1, self.PC.size), device=self.device)
    
    def _apply_mask(self, SR: torch.Tensor) -> torch.Tensor:
        """Apply masking to spatial representation if configured."""
        if self.mask_indices is not None:
            SR[0, self.mask_indices] = 0
        return SR


class CANNSR(SpatialRepresentationStrategy):
    """Spatial representation using CANN (Continuous Attractor Neural Network)."""
    
    def __init__(self, cann: Any, device: torch.device):
        self.CANN = cann
        self.device = device
    
    def compute_SR(self, action: np.ndarray, obs: Dict) -> torch.Tensor:
        """Compute SR using CANN."""
        # TODO: Implement based on CANN interface
        # For now, return empty tensor as placeholder
        return torch.tensor([], device=self.device).unsqueeze(dim=0)
    
    def initialize_SR(self, obs: Dict) -> torch.Tensor:
        """Initialize SR for CANN."""
        return torch.zeros((1, self.CANN.hidden_size), device=self.device)


def create_spatial_representation_strategy(
    config,
    env,
    predictiveNet,
    device: torch.device
) -> SpatialRepresentationStrategy:
    """
    Factory function to create appropriate spatial representation strategy.
    
    Args:
        config: Spatial configuration
        device: Torch device
    
    Returns:
        Appropriate SpatialRepresentationStrategy instance
    """
    if config.predictive_net is not None:
        if config.train:
            assert config.predictive_net.seqdur > 0, "Set an appropriate seqdur"
        # Check for theta cycle mode
        if 'thcyc' in str(predictiveNet.pRNN):
            logger.info("Using ThetaCyclePredictiveNetworkSR")
            return ThetaCyclePredictiveNetworkSR(
                predictiveNet, 
                device, 
                config.mask_indices
            )
        elif config.past_SR:
            logger.info("Using PastSR PredictiveNetworkSR")
            return PredictiveNetworkPastSR(
                predictiveNet, 
                device,
                mask_indices=config.mask_indices
            )
        else:
            logger.info("Using PredictiveNetworkSR")
            return PredictiveNetworkSR(
                predictiveNet, 
                device, 
                config.mask_indices
            )
    elif config.place_cells is not None:
        logger.info("Using PlaceCellsSR")
        PC = FakePlaceCells(
            env,
            config.cells,
            config.place_cells.pc_sd,
        )
        return PlaceCellsSR(
            PC, 
            device, 
            config.mask_indices,
            config.past_SR
        )
    elif config.CANN is not None:
        logger.info("Using CANNSR")
        CANN = CANNnet(env,
                        hidden_size = config.cells,
                        mapsize = [env.width, env.height])
        return CANNSR(CANN, device)
    else:
        logger.info("Using NoSpatialRepresentation")
        return NoSpatialRepresentation(device)

import torch
import numpy as np

import RLutils
from .other import device
from RLutils.model import ACModel


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


class ActorCriticAgent:

    def __init__(self, action_space, acmodel, prnn, device, pastSR=True):
        self.action_space = action_space
        self.acmodel = acmodel
        self.prnn = prnn
        self.device = device
        self.pastSR = pastSR
        self.name = 'ActorCritic Agent'

    def next_SR(self, obs, act):

        obs = [obs, obs]
    
        obs_pN, act_pN = self.prnn.env_shell.env2pred(obs, act)
        obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
        with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
            SR = self.prnn.predict_single(obs_pN[:,:-1,:], act_pN).squeeze(dim=0)

        return SR
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False, **kwargs):
        
        self.prnn.pRNN.to(self.device)
        render = False

        obs = [None for t in range(tsteps+1)]
        act = [None for t in range(tsteps)]

        if reset:
            obs[0] = env.reset()
        else:
            o = env.env.gen_obs()
            obs[0] = env.env.observation(o)
        
        state = {'agent_pos': np.resize(env.get_agent_pos(),(1,2)), 
                 'agent_dir': env.get_agent_dir(),
                }
        
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
        
        if self.pastSR:
            SR = torch.zeros((1,self.prnn.hidden_size), device=self.device) 
            state['SRs'] = SR.cpu().numpy()
        else:
            raise NotImplementedError
        
        for aa in range(tsteps):
            #obs_tensor = torch.tensor(obs[aa]['image'], device=self.device)
            _, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)
            preprocessed_obs = preprocess_obss([obs[aa]], device=self.device)
            with torch.no_grad():
                dist, _ = self.acmodel(preprocessed_obs, SR=SR)
                action = dist.sample()
                act[aa] = action.cpu().numpy()
            
            obs[aa+1] = env.step(act[aa])[0]
            state['agent_pos'] = np.append(state['agent_pos'], 
                                           np.resize(env.get_agent_pos(),(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.get_agent_dir())

            SR = self.next_SR(obs[aa], act[aa])
            state['SRs'] = np.append(state['SRs'], SR.cpu().numpy())

            if includeRender:
                render[aa+1] = env.render(mode=None)
        
        self.prnn.pRNN.to("cpu")
        
        act = np.array(act).reshape(-1)
        return obs, act, state, render
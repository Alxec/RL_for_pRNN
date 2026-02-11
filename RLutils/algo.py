# Based on the PPO algo from torch-ac library (https://github.com/lcswillems/torch-ac)


import torch
import math
import numpy as np

from scipy.stats import entropy
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from scipy.spatial.distance import cosine
from scipy.linalg import toeplitz

def compare_trajs(traj1, traj2):
    delta = (traj1 == traj2).cumprod()
    return delta.sum()/len(delta)


class PredictivePPOAlgo:
    """The base class for RL algorithms."""

    def __init__(self, env, acmodel, predictiveNet=None, device=None, num_frames=None, discount=0.99, lr=0.001,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, place_cells=None,
                 cann=None, train_pN=False, noise_mu=0, noise_std=0.03, prnn_seqdur=0, intrinsic=False, k_int=1,
                 pastSR=False, curious_agent=False, k_curious=1, exploration=False, mask_indices=None):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        env : environment
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time - NO?
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle - NO?
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        print('Store parameters')
        self.env = env
        self.acmodel = acmodel
        self.pN = predictiveNet
        self.device = device
        self.num_frames = num_frames or 128
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.intrinsic = intrinsic
        self.k_int = k_int
        self.PC = place_cells
        self.CANN = cann
        self.train_pN = train_pN
        self.noise_mu = noise_mu
        self.noise_std = noise_std
        self.prnn_seqdur = prnn_seqdur
        self.pastSR = pastSR
        self.curious_agent = curious_agent
        self.k_curious = k_curious
        self.exploration = exploration
        self.mask_indices = mask_indices
        assert pastSR ^ ('Next' in str(env.encodeAction))

        if hasattr(self.env, 'loc_mask'):
            self.loc_mask = self.env.loc_mask
        elif 'Shell' in str(type(env)):
            self.loc_mask = [x==None or x.can_overlap() for x in env.env.grid.grid]
        else:
            self.loc_mask = [x==None or x.can_overlap() for x in env.grid.grid]

        if self.pN and 'thcyc' in str(self.pN.pRNN):
            self.theta = True
            self.k = self.pN.pRNN.k + 1
        else:
            self.theta = False

        # Control parameters
        print('Control parameters')
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames % self.recurrence == 0

        # Configure models
        print('Configure acmodel')
        self.acmodel.to(self.device)
        self.acmodel.train()
        if self.pN:
        # TODO: should be elsewhere if saving the net
            self.pN.pRNN.to(self.device)

        self.obs = self.env.reset()
        self.loc = self.agent_pos()
        self.mask = 1
        print('Reset done')

        # Initialize spatial representations (if used)
        self.init_SR()

        # Initialize experiences
        self.init_exp()

        # Initialize log values
        self.init_log()

        # Initialize intrinsic rewards
        if self.intrinsic:
            self.ref = torch.zeros((1,self.SR.shape[-1]), device=self.device)
            self.nrefs = 0
            self.int_rewards = torch.zeros(self.num_frames, device=self.device)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        print('All done')

    def collect_experiences(self, return_joint_distribution=False):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames, ...).
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        #Joint prob between states and actions. Used in on-policy analysis
        joint_probabilities = np.zeros((getattr(self.env, "numHDs"),
                                        self.env.width,
                                        self.env.height,
                                        getattr(self.acmodel, "act_dim")), dtype=np.float32)

        #The lists below are only relevant if pRNN is being trained
        self.done_indices = [0]
        self.last_observations = []

        for i in range(self.num_frames):
            # Do one agent-environment interaction

            action, dist, value, memory, det_action = self.next_experience()

            obs, reward, terminated, truncated, _ = self.env.step(det_action)
            if self.exploration:
                reward, terminated, truncated = 0, False, False
            loc = self.agent_pos()
            done = terminated or truncated

            # Update spatial representation
            if self.pastSR:
                # using the obs obtained at the previous step
                SR = self.next_SR(det_action, self.obs)
            else:
                # using the obs obtained at the current step
                SR = self.next_SR(det_action, obs)

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.locs[i] = self.loc
            self.loc = loc
            # SR at step i is the one used to get act[i] (from step i-1 for pastSR)
            self.SRs[i] = self.SR
            self.SR = SR # self.SR recorded at step i is the one used to get act[i+1]
            # if self.acmodel.recurrent: # Not using it now, disabled for efficiency
            #     self.memories[i] = self.memory
            #     self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - done
            self.actions[i] = action
            self.values[i] = value
            # if self.reshape_reward is not None: # Not using it now, disabled for efficiency
            #     self.rewards[i] = self.reshape_reward(obs, action, reward, done)
            # else:
            #     self.rewards[i] = reward
            self.rewards[i] = reward
            self.log_probs[i] = dist.log_prob(action)

            #add counts to joint probs
            try:
                hd = self.obss[i]["direction"]
            except KeyError:
                hd = self.obss[i]["HD"]
            x, y = self.locs[i]
            act_probs = dist.probs.detach().cpu().numpy().squeeze()
            joint_probabilities[hd, x, y, :] += act_probs

            # Update log values

            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            if self.prnn_seqdur > 0 and (i+1)%self.prnn_seqdur==0:
                done = True

            if done:
                if self.intrinsic and reward > 1e-5:
                    if self.pastSR:
                        _,_,_,_,det_action = self.next_experience()
                        SR = self.next_SR(det_action, self.obs)
                    self.update_ref(SR)
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return)
                self.log_reshaped_return.append(self.log_episode_reshaped_return)
                self.log_num_frames.append(self.log_episode_num_frames)
                if self.pN:
                    self.pN.reset_state(device=self.device)
                self.init_SR()
                self.last_observations.append(self.obs)
                self.obs = self.env.reset()
                self.log_episode_return = 0
                self.log_episode_reshaped_return = 0
                self.log_episode_num_frames = 0
                self.done_indices.append(i+1)
        
        #make sure last obs is included in done indices. 
        #these is when each trial ends, for prnn training
        if self.done_indices[-1]!=i+1:
            self.done_indices.append(i+1)
            self.last_observations.append(self.obs)

        #Calculate curious rewards
        if self.curious_agent:
            with torch.no_grad():
                actions_preformatted = self.actions.cpu().numpy()
                obs_formatted, act_formatted = self.pN.env_shell.env2pred(self.obss + [self.obs], actions_preformatted)
                obs_formatted, act_formatted = obs_formatted.to(self.device), act_formatted.to(self.device)
                obs_pred, obs_next, _ = self.pN.predict(obs_formatted, act_formatted)
                obs_pred, obs_next = obs_pred.squeeze(0), obs_next.squeeze(0)
                MSEs = ((obs_pred - obs_next) ** 2).mean(dim=1)  # [2048]
                self.curious_rewards = MSEs

        # Calculate intrinsic rewards
        if self.intrinsic:
            _,_,_,_,det_action = self.next_experience()
            if self.pastSR:
                SR = self.next_SR(det_action, self.obs)
            else:
                SR = self.next_SR(det_action, obs)
            # Add SR from the last state
            SRs = torch.cat((self.SRs, SR), dim=0).cpu()
            # Calculate cosine similarity between SRs and reference SR
            # if self.pastSR:
            errors = torch.tensor([cosine(SR, self.ref.squeeze().cpu()) for SR in SRs[1:]], device=self.device)
            errors = torch.cat((errors[0][None], errors), dim=0)
            # else:
            #     errors = torch.tensor([cosine(SR, self.ref.squeeze().cpu()) for SR in SRs], device=self.device)
            # Calculate intrinsic rewards
            self.int_rewards = errors[:-1] - errors[1:]

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            # if self.acmodel.recurrent:
            #     noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
            #     # TODO: remove SR if not using PCs with recurrence
            #     _, next_value, _ = self.acmodel(preprocessed_obs,
            #                                     (self.memory * self.mask)[None],
            #                                     noise=noise,
            #                                     SR=self.SR)
            # else:
                _, next_value = self.acmodel(preprocessed_obs, SR=self.SR)

        for i in reversed(range(self.num_frames)):
            next_mask = self.masks[i+1] if i < self.num_frames - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames - 1 else 0

            reward_term = self.rewards[i] + self.k_int * self.int_rewards[i] + self.k_curious * self.curious_rewards[i]
            delta = reward_term + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        exps = DictList()
        exps.obs = self.obss
        exps.SR = self.SRs
        # if self.acmodel.recurrent:
        #     exps.memory = self.memories
        #     exps.mask = self.masks
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage # approximates current and discounted future returns
        exps.log_prob = self.log_probs
        exps.done_indices = self.done_indices
        exps.last_observations = self.last_observations

        # Calculate locations entropy
        for loc in self.locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)
        
        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros([self.env.env.grid.width, self.env.env.grid.height])

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Reset pN state
        if self.pN:
            self.pN.reset_state(device=self.device)

        # Log some values

        logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames,
            "num_episodes": self.log_done_counter,
            "intrinsic_rewards": self.int_rewards.tolist(),
            "curious_rewards": self.curious_rewards.tolist(),
            "values": self.values.tolist(),
            "advantages": self.advantages.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "joint_dist": joint_probabilities
        }

        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        return exps, logs


    def update_parameters(self, exps, update_params=True):
        # Collect experiences
        # TODO: deal with analyses, predNet saving, option to backprop through pRNN

        #below has to be done so that exps can be batched
        done_indices = exps.done_indices.copy()
        last_observations = exps.last_observations.copy()
        del exps['done_indices']
        del exps['last_observations']        

        for _ in range(self.epochs): # TODO: should it be just one epoch?
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    # if self.acmodel.recurrent:
                    #     noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
                    #     # NOTE: it is still not analogous to pRNN because obs is from current step, and act is not provided
                    #     dist, value, memory = self.acmodel(sb.obs, (memory.T * sb.mask).T[None], noise=noise, SR=sb.SR)
                    # else:
                    dist, value = self.acmodel(sb.obs, SR=sb.SR)

                    policy_entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * policy_entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += (policy_entropy.item() / torch.log(torch.tensor(2.0))) #convert nats to bits
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    # if self.acmodel.recurrent and i < self.recurrence - 1:
                    #     exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                if update_params:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else:
                    grad_norm = 0.0

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Update pN

        if self.train_pN:
            #The below is bugged! discuss with Alex

            #obs, act = self.pN.env_shell.env2pred(exps.obs + [self.obs], exps.action) # including the last observation
            #_,_,_ = self.pN.trainStep(obs, act, mask=exps.masks+[self.mask])

            #Fixed code:
            self.pN.pRNN.to(self.device)
            for idx in range(1, len(done_indices)):
                start_episode = done_indices[idx-1]
                end_episode = done_indices[idx]
                last_obs = last_observations[idx-1]
                self._train_pN(exps, start_episode, end_episode, last_obs)

        # Log some values

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms)
        }

        return logs


    def _train_pN(self, exps, startIdx, endIdx, last_obs):
        images_tensor, hd_tensor = exps.obs.image[startIdx:endIdx], exps.obs.direction[startIdx:endIdx]
        obs_for_pN = [{'image': images_tensor[i].cpu().numpy(), 'direction': hd_tensor[i].item()} 
                    for i in range(len(images_tensor))]
        act_for_pN = exps.action[startIdx:endIdx].cpu().numpy()
        
        obs, act = self.pN.env_shell.env2pred(obs_for_pN + [last_obs], act_for_pN) 

        obs = obs.to(self.device)
        act = act.to(self.device)
        _,_,_ = self.pN.trainStep(obs, act)
        self.pN.numTrainingEpochs +=1


    def randomAgent_collect_exp_and_update(self, agent):
        assert self.train_pN, "The only reason to have random actions in algo is to train the pRNN geinus..."
        self.pN.pRNN.to(self.device)
        numtrials = math.ceil(self.num_frames / self.prnn_seqdur)

        log_curr_seqdurs = []
        for bb in range(numtrials):
            curr_seqdur = min(self.prnn_seqdur, self.num_frames - (bb)*self.prnn_seqdur) 
            log_curr_seqdurs.append(curr_seqdur)
            #The above is needed if self.prnn_seqdur is not a perfect divisor of num trials.
            # It implies that the last trial might have <seqdur steps

            obs,act,state,_ = self.pN.collectObservationSequence(self.env,
                                                                 agent, curr_seqdur)
            
            #Train
            obs, act = obs.to(self.device), act.to(self.device)
            _,_,_ = self.pN.trainStep(obs, act)
            self.pN.numTrainingEpochs += 1

            #Collect location info
            locs_array = state['agent_pos'][:-1,:]
            loc_list_current = [tuple(thisloc) for thisloc in locs_array]

            startidx = bb*self.prnn_seqdur
            endidx = min(self.num_frames, (bb+1)*self.prnn_seqdur)
            self.locs[startidx:endidx] = loc_list_current

        for loc in self.locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(self.loc_visits, base=2)

        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        self.loc_visits = np.zeros([self.env.width, self.env.height])

        policy_entropy = entropy(agent.default_action_probability, base=2)

        return {"num_frames": self.num_frames,
                "num_frames_per_episode": log_curr_seqdurs,
                "num_episodes": numtrials,
                "entropy": policy_entropy,
                "loc_entropy": loc_entropy,
                "loc_entropy_5": loc_entropy_5}


    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        #If batch size is 257 and numtrials is 2048, there will still be 8 batches
        #first 7 batches will have 257 elements in them
        #The last batch will have 2048 - (257*7) = 249 elements in it
        #The above comment assumes recurrence is 1

        indexes = np.arange(0, self.num_frames, self.recurrence)
        indexes = np.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
    

    def next_experience(self):
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        memory = None

        with torch.no_grad():
            # if self.acmodel.recurrent:
            #     noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
            #     # TODO: remove SR if not using PCs with recurrence
            #     dist, value, memory = self.acmodel(preprocessed_obs,
            #                                         (self.memory * self.mask)[None],
            #                                         noise=noise,
            #                                         SR=self.SR)
            # else:
            dist, value = self.acmodel(preprocessed_obs, SR=self.SR)
        action = dist.sample() # choose action based on SR from step t-1
        det_action = action.cpu().numpy()

        return action, dist, value, memory, det_action
    

    def next_SR(self, act, obs):
        if self.theta:
            obs = [obs] * (self.k + 1)
            act = act.repeat(self.k)
        
            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
                SR = self.pN.predict(obs_pN, act_pN)[2][0]

        elif self.pN:
            obs = [obs, obs]
        
            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
                SR = self.pN.predict_single(obs_pN[:,:-1,:], act_pN).squeeze(dim=0)

        elif self.PC: # not calculating it on the same step for now
            SR = torch.tensor(self.PC.activation(self.env.env.agent_pos), dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        # Not using the CANNs anymore, so not solving the preprocessed_obs problem
        # elif self.CANN: # not calculating it on the same step for now
        #     with torch.no_grad():
        #         _,_,SR = self.CANN.predict(preprocessed_obs,None,self.env.get_agent_pos())
        #     SR = SR[0].to(self.device)
        
        else:
            SR = torch.tensor([], device=self.device).unsqueeze(dim=0)

        if self.mask_indices is not None:
            SR[0,self.mask_indices] = 0
        return SR
    

    def init_SR(self):
        SR = torch.tensor([], device=self.device).unsqueeze(dim=0)
        if self.pN:
            if self.pastSR:
                SR = torch.zeros((1,self.pN.hidden_size), device=self.device)
            else:
                obs_pN, act_pN = self.pN.env_shell.env2pred([self.obs,self.obs], np.array([0]))
                act_pN = torch.zeros_like(act_pN)
                obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
                with torch.no_grad():
                    SR = self.pN.predict_single(obs_pN[:,:-1,:], act_pN).squeeze(dim=0)

        if self.CANN:
            SR = torch.zeros((1,self.CANN.hidden_size), device=self.device)
        elif self.PC:
            SR = torch.zeros((1,self.PC.size), device=self.device)
        
        self.SR = SR

    def init_exp(self):
        self.obss = [None] * (self.num_frames)
        self.locs = [None] * (self.num_frames)
        # if self.acmodel.recurrent:
        #     self.memory = torch.zeros((1,self.acmodel.memorysize), device=self.device)
        #     self.memories = torch.zeros(self.num_frames, self.acmodel.memorysize, device=self.device)
        self.mask = 1
        self.masks = torch.zeros(self.num_frames, device=self.device)
        print('Masks done')
        self.actions = torch.zeros(self.num_frames, device=self.device, dtype=torch.int)
        self.values = torch.zeros(self.num_frames, device=self.device)
        self.SRs = torch.zeros((self.num_frames, self.SR.shape[1]), device=self.device)
        print('Values done')
        self.rewards = torch.zeros(self.num_frames, device=self.device)
        self.advantages = torch.zeros(self.num_frames, device=self.device)
        print('Advantages done')
        self.log_probs = torch.zeros(self.num_frames, device=self.device)
        self.int_rewards = torch.zeros(self.num_frames, device=self.device)
        self.curious_rewards = torch.zeros(self.num_frames, device=self.device)
        self.loc_visits = np.zeros([self.env.width, self.env.height])
        self.loc_history = [np.zeros(np.sum(self.loc_mask))] * 5

    def init_log(self):
        print('Initialize log values')
        self.log_episode_return = 0
        self.log_episode_reshaped_return = 0
        self.log_episode_num_frames = 0

        self.log_done_counter = 0
        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []
    

    def update_ref(self, activations):
        self.nrefs += 1
        self.ref = self.ref + (activations - self.ref) / self.nrefs


    def agent_pos(self):
        if hasattr(self.env, 'get_agent_pos'):
            loc = self.env.get_agent_pos()
        else:
            loc = self.env.agent_pos
        return loc
    

class thetaPPOalgo(PredictivePPOAlgo):
    def __init__(self, env, acmodel, predictiveNet=None, device=None, num_frames=None, discount=0.99, lr=0.001,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, place_cells=None,
                 cann=None, train_pN=False, noise_mu=0, noise_std=0.03, intrinsic=False, k_int=1,
                 pastSR=False, eval_type='rewards', value_type='single'):
        self.theta_k = acmodel.seq_length
        self.hd = torch.zeros((1, self.theta_k), device=device, dtype=torch.int)
        self.action = torch.ones((1, self.theta_k), device=device, dtype=torch.int)*3 # 3 is the stop action
        self.eval = torch.zeros((1, self.theta_k), device=device, dtype=torch.float)
        assert eval_type in ['rewards', 'errors']
        self.eval_type = eval_type
        self.value_type = value_type
        super().__init__(env, acmodel, predictiveNet, device, num_frames, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs, batch_size, preprocess_obss,
                         place_cells, cann, train_pN, noise_mu, noise_std, intrinsic, k_int, pastSR)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames, ...).
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames):
            # Do one agent-environment interaction

            self.action, dist, value, det_action = self.next_experience()

            obs, reward, terminated, truncated, _ = self.env.step(det_action[0,0])
            loc = self.agent_pos()
            done = terminated or truncated

            # Update spatial representation
            if self.pastSR:
                SR, hd = self.next_SR(det_action, self.obs)
                self.SRs[i] = self.SR # SR at step i is from step i-1 (empty at step 0)
                self.HDs[i] = self.hd
                self.hd = hd
            else:
                SR, hd = self.next_SR(det_action, obs)
                self.hd = hd
                self.SRs[i] = SR # SR at step i is from step i
                self.HDs[i] = self.hd

            # Update intrinsic rewards
            if self.intrinsic:
                self.evals[i] = self.eval
                if any(self.ref[0]):
                    SRs = torch.cat((self.SR[:,0], SR[0]), dim=0).cpu()
                    # Calculate cosine similarity between SRs and reference SR
                    errors = torch.tensor([cosine(SR, self.ref.squeeze().cpu()) for SR in SRs],
                                          device=self.device,
                                          dtype=torch.float)
                    # Calculate intrinsic rewards
                    int_rewards = errors[:-1] - errors[1:]

                    if self.eval_type == 'errors':
                        self.eval = errors[1:]
                    else:
                        self.eval = int_rewards
                    
                    self.eval = self.eval.unsqueeze(dim=0)
            
            self.SR = SR

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.locs[i] = self.loc
            self.loc = loc
            self.masks[i] = self.mask
            self.mask = 1 - done
            self.actions[i] = self.action
            self.values[i] = value
            self.rewards[i] = reward
            self.log_probs[i] = dist.log_prob(self.action)

            # Update log values

            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            if done:
                if self.intrinsic and reward > 1e-5:
                    if self.pastSR:
                        _,_,_,det_action = self.next_experience()
                        SR,_ = self.next_SR(det_action, self.obs)
                    self.update_ref(SR[0,0])
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return)
                self.log_reshaped_return.append(self.log_episode_reshaped_return)
                self.log_num_frames.append(self.log_episode_num_frames)
                self.pN.reset_state(device=self.device)
                self.init_SR()
                self.obs = self.env.reset()
                self.action = torch.ones((1, self.theta_k),
                                          device=self.device,
                                          dtype=torch.int)*3
                self.eval = torch.zeros((1, self.theta_k),
                                        device=self.device,
                                        dtype=torch.float)
                self.hd = torch.zeros((1, self.theta_k),
                                      device=self.device,
                                      dtype=torch.int)
                self.log_episode_return = 0
                self.log_episode_reshaped_return = 0
                self.log_episode_num_frames = 0

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs,
                                         SR=self.SR,
                                         HDs=self.hd,
                                         acts=self.action,
                                         values=self.eval)

        for i in reversed(range(self.num_frames)):
            next_mask = self.masks[i+1] if i < self.num_frames - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames - 1 else 0

            delta = self.rewards[i] + self.k_int * self.int_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        exps = DictList()
        exps.obs = self.obss
        exps.SR = self.SRs
        exps.HD = self.HDs
        exps.eval = self.evals
        exps.action = self.actions
        exps.in_action = torch.cat((torch.ones((1, self.theta_k), device=self.device, dtype=torch.int),
                                    self.actions[:-1]), dim=0)
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage # approximates current and discounted future returns
        exps.log_prob = self.log_probs

        # Calculate locations entropy
        for loc in self.locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(self.loc_visits)
        
        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0))
        self.loc_visits = np.zeros([self.env.env.grid.width, self.env.env.grid.height])

        # Calculate projected similarity
        act = self.actions.cpu().numpy()
        stop = act.shape[0]-act.shape[1]
        idx = np.flip(toeplitz(np.arange(act.shape[1]),
                               np.arange(act.shape[0])),
                      0)[:,act.shape[1]-1:].T
        sim = ((act[:,0][idx]==act[:stop+1]).cumprod(1)[:,1:].sum(1)/5).mean()

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Reset pN state
        if self.pN:
            self.pN.reset_state(device=self.device)

        # Log some values

        logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames,
            "num_episodes": self.log_done_counter,
            "intrinsic_rewards": self.int_rewards.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            'proj_sim': sim
        }

        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        return exps, logs


    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Create a sub-batch of experience

                sb = exps[inds]

                # Compute loss

                dist, value = self.acmodel(sb.obs,
                                           sb.SR,
                                           sb.HD,
                                           sb.in_action,
                                           sb.eval
                                           )

                entropy = dist.entropy().mean()

                ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                if self.value_type == 'single':
                    surr1 = (ratio.prod(dim=1) ** (1 / self.theta_k)) * sb.advantage
                    clamped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    surr2 = (clamped.prod(dim=1) ** (1 / self.theta_k)) * sb.advantage
                else:
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                surr1 = (value - sb.returnn).pow(2)
                surr2 = (value_clipped - sb.returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                batch_entropy += entropy.item()
                batch_value += value.mean().item()
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_loss += loss

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Update pN

        if self.train_pN:
            obs, act = self.pN.env_shell.env2pred(exps.obs + [self.obs], exps.action) # including the last observation
            _,_,_ = self.pN.trainStep(obs, act, mask=exps.masks+[self.mask])

        # Log some values

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms)
        }

        return logs
    

    def next_experience(self):
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)

        with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs,
                                       self.SR,
                                       self.hd,
                                       self.action,
                                       self.eval
                                       )
        action = dist.sample() # choose action based on SR from step t-1
        det_action = action.cpu().numpy()

        return action, dist, value, det_action
    

    def next_SR(self, act, obs):
        obs_pN, act_pN, hd = self.pN.env_shell.env2pred([obs]*(self.theta_k+1), act[0], hd_from='act')
        obs_pN, act_pN, hd = obs_pN.to(self.device), act_pN.to(self.device), hd.to(self.device)[None,:-1]
        with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
            _,_,SR = self.pN.predict(obs_pN, act_pN)
        if self.pastSR:
            SR = SR.permute(1,0,2)
        else:
            SR = SR.permute(1,0,2)[1][None]
        return SR, hd
    

    def init_SR(self):
        if self.pastSR:
            SR = torch.zeros((1,self.theta_k,self.pN.hidden_size), device=self.device)
        else:
            obs_pN, act_pN = self.pN.env_shell.env2pred([self.obs]*(self.theta_k+1), np.zeros(self.theta_k))
            act_pN = torch.zeros_like(act_pN)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                _,_,SR = self.pN.predict(obs_pN, act_pN)
            if self.pastSR:
                SR = SR.permute(1,0,2)
            else:
                SR = SR.permute(1,0,2)[1][None]
        
        self.SR = SR

    def init_exp(self):
        self.obss = [None] * (self.num_frames)
        self.locs = [None] * (self.num_frames)
        self.masks = torch.zeros(self.num_frames, device=self.device)
        print('Masks done')
        self.actions = torch.zeros((self.num_frames, self.theta_k), device=self.device, dtype=torch.int)
        self.HDs = torch.zeros((self.num_frames, self.theta_k), device=self.device, dtype=torch.int)
        self.values = torch.zeros(self.num_frames, device=self.device) #TODO: adapt for multiple
        self.SRs = torch.zeros((self.num_frames, *self.SR.shape[1:]), device=self.device)
        self.evals = torch.zeros((self.num_frames, self.theta_k), device=self.device)
        print('Values done')
        self.rewards = torch.zeros(self.num_frames, device=self.device)
        self.advantages = torch.zeros(self.num_frames, device=self.device)
        print('Advantages done')
        self.log_probs = torch.zeros(self.num_frames, self.theta_k, device=self.device)
        self.int_rewards = torch.zeros(self.num_frames, device=self.device)
        self.loc_visits = np.zeros([self.env.width, self.env.height])
        self.loc_history = [np.zeros(np.sum(self.loc_mask))] * 5


class SingleThetaPPOalgo(PredictivePPOAlgo):
    def __init__(self, env, acmodel, predictiveNet=None, device=None, num_frames=None, discount=0.99, lr=0.001,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, place_cells=None,
                 cann=None, train_pN=False, noise_mu=0, noise_std=0.03, intrinsic=False, k_int=1,
                 pastSR=False, eval_type='rewards', value_type='single'):
        self.theta_k = acmodel.seq_length
        self.hd = torch.zeros((1, self.theta_k), device=device, dtype=torch.int)
        self.action = torch.ones((1, self.theta_k), device=device, dtype=torch.int)*3 # 3 is the stop action
        self.eval = torch.zeros((1, self.theta_k), device=device, dtype=torch.float)
        assert eval_type in ['rewards', 'errors']
        self.eval_type = eval_type
        self.value_type = value_type
        super().__init__(env, acmodel, predictiveNet, device, num_frames, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, clip_eps, epochs, batch_size, preprocess_obss,
                         place_cells, cann, train_pN, noise_mu, noise_std, intrinsic, k_int, pastSR)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames, ...).
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames):
            # Do one agent-environment interaction

            self.action, dist, value, det_action = self.next_experience()

            obs, reward, terminated, truncated, _ = self.env.step(det_action[0,0])
            loc = self.agent_pos()
            done = terminated or truncated

            # Update spatial representation
            if self.pastSR:
                SR, hd = self.next_SR(det_action, self.obs)
                self.SRs[i] = self.SR # SR at step i is from step i-1 (empty at step 0)
                self.HDs[i] = self.hd
                self.hd = hd
            else:
                SR, hd = self.next_SR(det_action, obs)
                self.hd = hd
                self.SRs[i] = SR # SR at step i is from step i
                self.HDs[i] = self.hd

            # Update intrinsic rewards
            if self.intrinsic and any(self.ref):
                self.evals[i] = self.eval
                SRs = torch.cat((self.SR[0][None], SR), dim=0).cpu()
                # Calculate cosine similarity between SRs and reference SR
                errors = torch.tensor([cosine(SR, self.ref.squeeze().cpu()) for SR in SRs], device=self.device)
                # Calculate intrinsic rewards
                int_rewards = errors[:-1] - errors[1:]

                if self.eval_type == 'errors':
                    self.eval = errors[1:]
                else:
                    self.eval = int_rewards
            
            self.SR = SR

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.locs[i] = self.loc
            self.loc = loc
            self.masks[i] = self.mask
            self.mask = 1 - done
            self.actions[i] = self.action
            self.values[i] = value
            self.rewards[i] = reward
            self.log_probs[i] = dist.log_prob(self.action)

            # Update log values

            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            if done:
                if self.intrinsic and reward > 1e-5:
                    if self.pastSR:
                        _,_,_,det_action = self.next_experience()
                        SR,_ = self.next_SR(det_action, self.obs)
                    self.update_ref(SR[0])
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return)
                self.log_reshaped_return.append(self.log_episode_reshaped_return)
                self.log_num_frames.append(self.log_episode_num_frames)
                self.pN.reset_state(device=self.device)
                self.init_SR()
                self.obs = self.env.reset()
                self.action = torch.ones((1, self.theta_k),
                                          device=self.device,
                                          dtype=torch.int)*3
                self.eval = torch.zeros((1, self.theta_k),
                                        device=self.device,
                                        dtype=torch.float)
                self.hd = torch.zeros((1, self.theta_k),
                                      device=self.device,
                                      dtype=torch.int)
                self.log_episode_return = 0
                self.log_episode_reshaped_return = 0
                self.log_episode_num_frames = 0

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs,
                                         SR=self.SR,
                                         HDs=self.hd,
                                         acts=self.action,
                                         values=self.eval)

        for i in reversed(range(self.num_frames)):
            next_mask = self.masks[i+1] if i < self.num_frames - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames - 1 else 0

            delta = self.rewards[i] + self.k_int * self.int_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        exps = DictList()
        exps.obs = self.obss
        exps.SR = self.SRs
        exps.HD = self.HDs
        exps.eval = self.evals
        exps.action = self.actions
        exps.in_action = torch.cat((torch.ones((1, self.theta_k), device=self.device, dtype=torch.int),
                                    self.actions[:-1]), dim=0)
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage # approximates current and discounted future returns
        exps.log_prob = self.log_probs

        # Calculate locations entropy
        for loc in self.locs:
            self.loc_visits[loc] += 1
        self.loc_visits = self.loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(self.loc_visits)
        
        self.loc_history.pop(0)
        self.loc_history.append(self.loc_visits)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0))
        self.loc_visits = np.zeros([self.env.env.grid.width, self.env.env.grid.height])

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Reset pN state
        if self.pN:
            self.pN.reset_state(device=self.device)

        # Log some values

        logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames,
            "num_episodes": self.log_done_counter,
            "intrinsic_rewards": self.int_rewards.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5
        }

        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        return exps, logs


    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Create a sub-batch of experience

                sb = exps[inds]

                # Compute loss

                dist, value = self.acmodel(sb.obs,
                                           sb.SR,
                                           sb.HD,
                                           sb.in_action,
                                           sb.eval
                                           )

                entropy = dist.entropy()[:,0].mean()

                ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                if self.value_type == 'single':
                    surr1 = ratio[:,0] * sb.advantage
                    clamped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    surr2 = clamped[:,0] * sb.advantage

                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                surr1 = (value - sb.returnn).pow(2)
                surr2 = (value_clipped - sb.returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                batch_entropy += entropy.item()
                batch_value += value.mean().item()
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_loss += loss

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Update pN

        if self.train_pN:
            obs, act = self.pN.env_shell.env2pred(exps.obs + [self.obs], exps.action) # including the last observation
            _,_,_ = self.pN.trainStep(obs, act, mask=exps.masks+[self.mask])

        # Log some values

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "grad_norm": np.mean(log_grad_norms)
        }

        return logs
    

    def next_experience(self):
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)

        with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs,
                                       self.SR,
                                       self.hd,
                                       self.action,
                                       self.eval
                                       )
        action = dist.sample() # choose action based on SR from step t-1
        det_action = action.cpu().numpy()

        return action, dist, value, det_action
    

    def next_SR(self, act, obs):
        obs_pN, act_pN, hd = self.pN.env_shell.env2pred([obs]*(self.theta_k+1), act[0], hd_from='act')
        obs_pN, act_pN, hd = obs_pN.to(self.device), act_pN.to(self.device), hd.to(self.device)[None,:-1]
        with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
            _,_,SR = self.pN.predict(obs_pN, act_pN)
        SR = SR.permute(1,0,2)
        return SR, hd
    

    def init_SR(self):
        if self.pastSR:
            SR = torch.zeros((1,self.theta_k,self.pN.hidden_size), device=self.device)
        else:
            obs_pN, act_pN = self.pN.env_shell.env2pred([self.obs]*(self.theta_k+1), np.zeros(self.theta_k))
            act_pN = torch.zeros_like(act_pN)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                _,_,SR = self.pN.predict(obs_pN, act_pN)
            SR = SR.permute(1,0,2)
        
        self.SR = SR

    def init_exp(self):
        self.obss = [None] * (self.num_frames)
        self.locs = [None] * (self.num_frames)
        self.masks = torch.zeros(self.num_frames, device=self.device)
        print('Masks done')
        self.actions = torch.zeros((self.num_frames, self.theta_k), device=self.device, dtype=torch.int)
        self.HDs = torch.zeros((self.num_frames, self.theta_k), device=self.device, dtype=torch.int)
        self.values = torch.zeros(self.num_frames, device=self.device) #TODO: adapt for multiple
        self.SRs = torch.zeros((self.num_frames, *self.SR.shape[1:]), device=self.device)
        self.evals = torch.zeros((self.num_frames, self.theta_k), device=self.device)
        print('Values done')
        self.rewards = torch.zeros(self.num_frames, device=self.device)
        self.advantages = torch.zeros(self.num_frames, device=self.device)
        print('Advantages done')
        self.log_probs = torch.zeros(self.num_frames, self.theta_k, device=self.device)
        self.int_rewards = torch.zeros(self.num_frames, device=self.device)
        self.loc_visits = np.zeros([self.env.width, self.env.height])
        self.loc_history = [np.zeros(np.sum(self.loc_mask))] * 5
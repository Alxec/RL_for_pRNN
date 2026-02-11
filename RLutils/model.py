# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ACModelTheta(ACModelSR):
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



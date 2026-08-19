import numpy as np
import torch
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from RLutils.model import ACModelSR
from RLutils.format import get_obss_preprocessor
from RLutils.other import device
from prnn.utils.Architectures import pRNN_th

SCALES = {'viridis': plotly.colors.sequential.Viridis,
          'default': plotly.colors.sequential.Plasma,}


def policy_spatial_activity(prnn, activity: torch.Tensor) -> torch.Tensor:
    """Select the one pRNN state per timestep consumed by the RL policy.

    A standard pRNN emits one state at every environmental timestep. A
    :class:`pRNN_th` rollout model instead emits one state for each theta step.
    PPO's active ``ACModelSR`` consumes a single response to the recorded
    observation-action input. For rollout pRNNs, analysis therefore uses the
    initial theta response. ``continuousTheta`` governs state propagation
    between pRNN inputs and does not change this per-input analysis state.
    """
    if not isinstance(prnn.pRNN, pRNN_th):
        return activity

    return activity[0:1]

def mutual_info_policy(joint_dist):
    """
    Compute I(S;A) from the un-normalized joint_probs array
    S is state, and A is action
    
    Input is (hd,x,y,a)
    """
    _,_,_,A = joint_dist.shape
    new_joint = joint_dist.copy().reshape(-1, A)
    mask = new_joint.sum(axis=1) > 0
    new_joint = new_joint[mask]

    p_sa = new_joint / new_joint.sum() # normalise
    p_s  = p_sa.sum(axis=1)
    p_a  = p_sa.sum(axis=0)

    H_s = entropy(p_s, base=2)
    H_a = entropy(p_a, base=2)
    H_sa = entropy(p_sa.flatten(), base=2)
    mi = H_s + H_a - H_sa

    return mi

def plot_heatmaps(feature, title='', zmin=None, zmax=None, HDs=True, scale='default', show_fig=True):
        """
        Plot the heatmaps of a feature.
        """
        if not isinstance(zmin, (int, float)):
            zmin = np.nanmin(feature)
        if not isinstance(zmax, (int, float)):
            zmax = np.nanmax(feature)

        if HDs:
            fig = make_subplots(rows=4, cols=2,
                            specs=[[{"colspan": 2, "rowspan":2}, None],
                                [None, None],
                                [{}, {}],
                                [{}, {}]])

            fig.add_trace(go.Heatmap(z=np.nanmean(feature, axis=0).T,
                                        colorscale=SCALES[scale]),
                                        row=1, col=1)
            fig.add_trace(go.Heatmap(z=feature[0].T,
                                        colorscale=SCALES[scale]),
                            row=3, col=1)
            fig.add_trace(go.Heatmap(z=feature[1].T,
                                        colorscale=SCALES[scale]),
                            row=3, col=2)
            fig.add_trace(go.Heatmap(z=feature[2].T,
                                        colorscale=SCALES[scale]),
                            row=4, col=1)
            fig.update_traces(showscale=False)
            fig.add_trace(go.Heatmap(z=feature[3].T,
                                        colorscale=SCALES[scale]),
                            row=4, col=2)
            fig.update_layout(height=800, width=600,
                              title_text=title,
                              title_x=0.5,)
        else:
            fig = go.Figure(data=go.Heatmap(
                            z=np.nanmean(feature, axis=0).T,
                            colorscale=SCALES[scale]))
            fig.update_layout(height=400, width=500,
                              title_text=title,
                              title_x=0.5,)
        fig.update_traces(zmin=zmin, zmax=zmax)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False,
                         autorange="reversed")
        fig.update_layout(font_family="Courier New")
        if HDs:
            fig.add_annotation(x=-0.1, y=0.37,
                            xref="paper", yref="paper",
                        text='→',
                        font={'size':24, 'family':'Courier'},
                        showarrow=False)
            fig.add_annotation(x=0.5, y=0.37,
                            xref="paper", yref="paper",
                        text='↓',
                        font={'size':24, 'family':'Courier'},
                        showarrow=False)
            fig.add_annotation(x=-0.1, y=0.08,
                            xref="paper", yref="paper",
                        text='←',
                        font={'size':24, 'family':'Courier'},
                        showarrow=False)
            fig.add_annotation(x=0.5, y=0.08,
                            xref="paper", yref="paper",
                        text='↑',
                        font={'size':24, 'family':'Courier'},
                        showarrow=False)

        if show_fig:
            fig.show()
        return fig

class EnvironmentFeaturesAnalysis:
    """
    Class for analyzing the features of the environment learned or used by RL agent.
    """
    def __init__(self, env, agent, rl_model = None, prnn_model = None, timesteps = 10000, PC = None):
        self.env = env
        self.agent = agent # agent to collect observations
        self.rl_model = rl_model
        self.prnn = prnn_model
        if self.prnn is not None:
            self.prnn.pRNN.to(device)
        self.timesteps = timesteps
        self.PC = PC
        _, self.preprocess_obss = get_obss_preprocessor(self.env.observation_space)

        self.data = self.collect_data()
        if rl_model:
            self.act_probs, self.values = self.get_values_actions()

    def collect_data(self):
        """
        Collect data from the environment (and pRNN).
        """
        data = {}

        if self.prnn:
            print('Collecting pRNN observations...')
            prnn_obs, prnn_act, data['state'], _, data['obs'] = \
                self.env.collectObservationSequence(self.agent, self.timesteps, save_env=True)
            with torch.no_grad():
                _, _, activity = self.prnn.predict(prnn_obs.to(device), prnn_act.to(device))
            data['h'] = policy_spatial_activity(self.prnn, activity)

        elif self.PC:
            print('Collecting PC observations...')
            data['obs'], _, data['state'], _ = self.agent.getObservations(self.env, self.timesteps)
            data['h'] = torch.tensor([self.PC.activation(pos) for pos in data['state']['agent_pos']],
                                     dtype=torch.float32, device=device).unsqueeze(0)
        else:
            print('Collecting environment observations...')
            data['obs'], _, data['state'], _ = self.agent.getObservations(self.env, self.timesteps)

        print('Collected {} steps for analysis'.format(len(data['obs'])-1))

        return data
    
    def get_values_actions(self):
        """
        Get the values and actions from the RL model.
        """
        probs = []
        values = []

        for t in range(self.timesteps):
            preprocessed_obs = self.preprocess_obss([self.data['obs'][t+1]], device=device)
            with torch.no_grad():
                if self.prnn or self.PC:
                    dist, value = self.rl_model(preprocessed_obs, SR=self.data['h'][:,t])
                else:
                    dist, value = self.rl_model(preprocessed_obs)

            prob = dist.probs
            probs.append(prob.to('cpu').numpy().squeeze())
            values.append(value.to('cpu').numpy())

        return np.array(probs), np.array(values)
    
    def values_map(self, zmin=None, zmax=None, HDs=True, scale='default', show_fig=True):
        """
        Plot the heatmaps of values.
        """
        instances_map = np.zeros((4, self.env.width-2, self.env.height-2))
        values_map = np.zeros((4, self.env.width-2, self.env.height-2))
        for t in range(self.timesteps):
            values_map[self.data['state']['agent_dir'][t+1],
                       self.data['state']['agent_pos'][t+1, 0]-1,
                       self.data['state']['agent_pos'][t+1, 1]-1] += self.values[t]
            
            instances_map[self.data['state']['agent_dir'][t+1],
                          self.data['state']['agent_pos'][t+1, 0]-1,
                          self.data['state']['agent_pos'][t+1, 1]-1] += 1
        
        values_map /= instances_map

        return plot_heatmaps(values_map, 'Values', zmin, zmax, HDs, scale, show_fig)

    def policy_map(self, show_fig=True):
        """
        Plot the heatmap of values.
        """
        instances_map = np.zeros((4, self.env.width-2, self.env.height-2, 4))
        probs_map = np.zeros((4, self.env.width-2, self.env.height-2, 4))
        for t in range(self.timesteps):
            probs_map[self.data['state']['agent_dir'][t+1],
                       self.data['state']['agent_pos'][t+1, 0]-1,
                       self.data['state']['agent_pos'][t+1, 1]-1] += self.act_probs[t]
            
            instances_map[self.data['state']['agent_dir'][t+1],
                          self.data['state']['agent_pos'][t+1, 0]-1,
                          self.data['state']['agent_pos'][t+1, 1]-1] += 1
        
        probs_map[0,:,:,2] += probs_map[1,:,:,0] + probs_map[3,:,:,1]
        probs_map[1,:,:,2] += probs_map[2,:,:,0] + probs_map[0,:,:,1]
        probs_map[2,:,:,2] += probs_map[1,:,:,1] + probs_map[3,:,:,0]
        probs_map[3,:,:,2] += probs_map[0,:,:,0] + probs_map[2,:,:,1]
        
        instances_map[0,:,:,2] += instances_map[1,:,:,0] + instances_map[3,:,:,1]
        instances_map[1,:,:,2] += instances_map[2,:,:,0] + instances_map[0,:,:,1]
        instances_map[2,:,:,2] += instances_map[1,:,:,1] + instances_map[3,:,:,0]
        instances_map[3,:,:,2] += instances_map[0,:,:,0] + instances_map[2,:,:,1]

        probs_map /= instances_map

        probs_map = np.argmax(probs_map[:,:,:,2], axis=0)
        probs_map = probs_map.astype(float)

        instances_map = instances_map.sum(axis=(0,3))
        probs_map[instances_map == 0] = np.nan

        act_map = np.zeros((self.env.width-2, self.env.height-2), dtype=str)
        act_map[probs_map == 0] = '→'
        act_map[probs_map == 1] = '↓'
        act_map[probs_map == 2] = '←'
        act_map[probs_map == 3] = '↑'

        fig = go.Figure(data=go.Heatmap(
                   z=probs_map.T))
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(text=act_map.T,
                        textfont_size=26,
                        texttemplate="%{text}",
                        showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=500, width=600,
                        title_text='Policy',
                        title_x=0.5)
        if show_fig:
            fig.show()
        return fig

    def calculate_errors(self, xref, yref):

        instances_map = np.zeros((4, self.env.width-2, self.env.height-2))
        errors_map = np.zeros((4, self.env.width-2, self.env.height-2))
        
        ref_ts=[]
        for t,pos in enumerate(self.data['state']['agent_pos'][:-1]):
            if pos[0] == xref and pos[1] == yref:
                ref_ts.append(t)

        h_ref = np.zeros_like(self.data['h'][0,0].to('cpu').numpy())
        for t in ref_ts:
            h_ref += self.data['h'][0,t].to('cpu').numpy()
        h_ref /= len(ref_ts)

        for t in range(self.timesteps):
            error = cosine(self.data['h'][0,t].to('cpu').numpy(), h_ref)
            if self.PC:
                n = t
            else:
                n = t+1
            errors_map[self.data['state']['agent_dir'][n],
                       self.data['state']['agent_pos'][n, 0]-1,
                       self.data['state']['agent_pos'][n, 1]-1] += error

            instances_map[self.data['state']['agent_dir'][n],
                          self.data['state']['agent_pos'][n, 0]-1,
                          self.data['state']['agent_pos'][n, 1]-1] += 1
        
        errors_map /= instances_map

        return errors_map
    
    def error_map(self, xref=7, yref=7, zmin=None, zmax=None, HDs=True, scale='viridis', show_fig=True):
        """
        Plot the heatmap of h_{ref} errors.
        """
        errors_map = self.calculate_errors(xref, yref)

        return plot_heatmaps(errors_map, 'Errors', zmin, zmax, HDs, scale, show_fig)


class OnPolicyAnalysis:
    """
    Class for analyzing the on-policy representations of the environment learned or used by RL agent.
    """
    def __init__(self, PPOalgo=None, timesteps=10000, **kwargs):
        # Imported lazily because ``RLutils.algo`` imports ``mutual_info_policy``
        # from this module during its own initialization.
        from RLutils.algo import PredictivePPOAlgo

        self.timesteps = timesteps
        if PPOalgo is not None:
            # Build a new algo with same params, just shorter timesteps
            ppo_config = PPOalgo.config.copy()
            ppo_config.num_frames = timesteps
            self.algo = PredictivePPOAlgo(
                env=PPOalgo.env,
                acmodel=PPOalgo.acmodel,
                predictiveNet=PPOalgo.predictiveNet,
                ppo_config=ppo_config,
                spatial_config=PPOalgo.spatial_config,
                reward_config=PPOalgo.reward_config,
                device=PPOalgo.device,
                preprocess_obss=PPOalgo.preprocess_obss,
            )
        else:
            required_keys = [
                'env', 'acmodel', 'predictiveNet', 'device',
                'preprocess_obss', 'ppo_config', 'spatial_config', 'reward_config'
            ]
            for key in required_keys:
                if key not in kwargs:
                    raise ValueError(f"Missing required argument: {key}")
            ppo_config.num_frames = timesteps
            self.algo = PredictivePPOAlgo(**kwargs)
        
        self.algo.collect_experiences()
        self.joint_probs = self.algo._logs_collect["joint_dist"]
        self.mi = mutual_info_policy(self.joint_probs)
        self.deltas = (self.algo.experience_buffer.advantages[:-1] - self.algo.config.discount * \
                      self.algo.config.gae_lambda * self.algo.experience_buffer.advantages[1:] * self.algo.experience_buffer.masks[1:]).cpu().numpy()
        self.deltas = np.append(self.deltas, self.algo.experience_buffer.advantages[-1].cpu().numpy())

    def plot_advantages(self, zmin=None, zmax=None, HDs=True, scale='default'):
        """
        Plot the heatmaps of advantages.
        """
        instances_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        adv_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        for t in range(self.timesteps):
            adv_map[self.algo.experience_buffer.obss[t]['direction'],
                    self.algo.experience_buffer.locs[t][0]-1,
                    self.algo.experience_buffer.locs[t][1]-1] += self.algo.experience_buffer.advantages[t].cpu().numpy()
            
            instances_map[self.algo.experience_buffer.obss[t]['direction'],
                          self.algo.experience_buffer.locs[t][0]-1,
                          self.algo.experience_buffer.locs[t][1]-1] += 1
        
        adv_map /= np.maximum(instances_map, 1e-6) 

        return plot_heatmaps(adv_map, 'Advantages', zmin, zmax, HDs, scale)

    def plot_deltas(self, zmin=None, zmax=None, HDs=True, scale='default'):
        """
        Plot the heatmaps of advantage deltas.
        """
        instances_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        delta_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        for t in range(self.timesteps):
            delta_map[self.algo.experience_buffer.obss[t]['direction'],
                    self.algo.experience_buffer.locs[t][0]-1,
                    self.algo.experience_buffer.locs[t][1]-1] += self.deltas[t]
            
            instances_map[self.algo.experience_buffer.obss[t]['direction'],
                          self.algo.experience_buffer.locs[t][0]-1,
                          self.algo.experience_buffer.locs[t][1]-1] += 1
        
        delta_map /= np.maximum(instances_map, 1e-6) 

        return plot_heatmaps(delta_map, 'Deltas', zmin, zmax, HDs, scale)
    
    def plot_values(self, zmin=None, zmax=None, HDs=True, scale='default'):
        """
        Plot the heatmaps of values.
        """
        instances_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        values_map = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        for t in range(self.timesteps):
            values_map[self.algo.experience_buffer.obss[t]['direction'],
                       self.algo.experience_buffer.locs[t][0]-1,
                       self.algo.experience_buffer.locs[t][1]-1] += self.algo.experience_buffer.values[t].cpu().numpy()
            
            instances_map[self.algo.experience_buffer.obss[t]['direction'],
                          self.algo.experience_buffer.locs[t][0]-1,
                          self.algo.experience_buffer.locs[t][1]-1] += 1
        
        values_map /= np.maximum(instances_map, 1e-6) 

        return plot_heatmaps(values_map, 'Values', zmin, zmax, HDs, scale)
    
    def plot_policy_heatmaps(self, scale='default'):
        """
        Visualise π(a|s). A 4×4 grid:
            Rows = head-direction  (↑, →, ↓, ←)
            Cols = actions         (↺, ↻, ↑, ·)
        """
        A = self.algo.acmodel.act_dim
        assert A == 4, "Only supports 4 actions for now"

        # Arrow labels
        hd_labels  = ['↑', '→', '↓', '←']   # rows (rotated layout)
        act_labels = ['↺', '↻', '↑', '·']   # columns

        joint  = self.joint_probs.copy()
        denom  = joint.sum(axis=3, keepdims=True)
        denom[denom == 0] = 1.0
        policy = joint / denom  # [hd, x, y, a]

        fig = make_subplots(
            rows=4, cols=4,
            specs=[[{}]*4]*4,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
            row_titles=[f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)],
            column_titles=[f"A {i}: {lbl}" for i, lbl in enumerate(act_labels)]
        )

        for hd in range(4):
            for a in range(4):
                z = policy[hd, :, :, a].T
                fig.add_trace(
                    go.Heatmap(z=z, showscale=False, colorscale=SCALES[scale]),
                    row=hd+1, col=a+1
                )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, autorange="reversed")
        fig.update_layout(
            height=900,
            width=900,
            title="Policy π(a|s)  (rows = HD, cols = action)",
            title_x=0.5,
            font_family="Courier New"
        )

        # Manually set font for subplot labels (row/col titles)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font = dict(size=24, family='Courier New', color='black')

        fig.show()
        return fig

    def plot_occupancy(self, scale='viridis'):
        """
        Show state-occupancy counts (no action dimension).
        1×4 layout – one heat-map per head-direction.
        """
        hd_labels = ['→', '↓', '←', '↑']

        occ = np.zeros((4, self.algo.env.width-2, self.algo.env.height-2))
        for t in range(self.timesteps):
            hd = self.algo.experience_buffer.obss[t]['direction']
            x, y = self.algo.experience_buffer.locs[t][0]-1, self.algo.experience_buffer.locs[t][1]-1
            occ[hd, x, y] += 1

        fig = make_subplots(
            rows=1, cols=4,
            specs=[[{}]*4],
            horizontal_spacing=0.03,
            column_titles=[f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)]
        )

        for hd in range(4):
            fig.add_trace(
                go.Heatmap(z=occ[hd].T, showscale=False, colorscale=SCALES[scale]),
                row=1, col=hd+1
            )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, autorange="reversed")
        fig.update_layout(
            height=280,
            width=900,
            title="State-occupancy per head-direction",
            title_x=0.5,
            font_family="Courier New"
        )

        # Make HD labels bigger + bold
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font = dict(size=24, family='Courier New', color='black')

        fig.show()
        return fig

    


        
        

import time
import datetime
import torch
import sys
import os
import shutil
import numpy as np

from omegaconf import OmegaConf, DictConfig, open_dict
import hydra
import wandb

import RLutils
from RLutils.other import device
from RLutils.model import ACModel, RecACModel, ACModelSR, ACModelTheta, ACModelThetaShared, ACModelThetaSingle
from RLutils.agent import ActorCriticAgent
from RLutils.algo_new import PredictivePPOAlgo, GoalConditionedPPOAlgo
from RLutils.analysis_new import EnvironmentFeaturesAnalysis, OnPolicyAnalysis
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.thetaRNN import LayerNormRNNCell, RNNCell
from prnn.utils.agent import create_agent

RNNoptions = {'LayerNormRNNCell' : LayerNormRNNCell ,
              'RNNCell' : RNNCell
                }


class RL_Trainer(object):

    def __init__(self, params):
 
        #############
        ## INIT
        #############

        # Get params, init WandB !!change WandB default folder
        self.params = params

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        if params.logging.focus:
            par = eval('params.'+params.logging.focus)
            name = f"{params.exp.exp_name}_{params.logging.focus}_{par}_seed{params.exp.seed}"
        else:
            name = f"{params.exp.exp_name}_seed{params.exp.seed}"
        name_date = f"{name}_{date}/"
        self.model_name = f"{params.logging.project}/{name_date}"
        if params.logging.focus:
            self.group = f"{params.exp.exp_name}_{params.logging.focus}_{par}"
        else:
            self.group = params.exp.exp_name

        # Set run dir

        if params.logging.load_acmodel:
            self.model_dir = params.logging.load_acmodel
        else:
            self.model_dir = RLutils.get_model_dir(self.model_name)
            RLutils.create_folders_if_necessary(self.model_dir)
        
        self.video_dir = RLutils.get_video_dir(self.model_name) if params.logging.video_log_freq!=0 else ''
        RLutils.create_folders_if_necessary(self.video_dir)

        print("\n\n\nLOGGING TO: ", self.model_dir, "\n\n\n")

        self.run = wandb.init(
                                # set the wandb project where this run will be logged
                                project = params.logging.project,
                                group = self.group,
                                name=name,
                                id = name_date[:-1],
                                dir = self.model_dir,
                                resume='allow',
                                # track hyperparameters and run metadata
                                config = OmegaConf.to_container(params, resolve=True)
                                )
        
    def _initialize_predictive_net(self, args, env):
        mask_indices = None
        if args.SR.predictive_net.load:
            predictiveNet = PredictiveNet.loadNet(args.SR.predictive_net.path,
                                                  args.SR.predictive_net.folder)
            # if not hasattr(predictiveNet.pRNN, 'hidden_size'):
            #     predictiveNet.pRNN.hidden_size = predictiveNet.pRNN.rnn.cell.hidden_size
            predictiveNet.env_shell.env = env.env
            env = predictiveNet.env_shell
            if args.SR.predictive_net.mask>0:
                if args.SR.predictive_net.mask_type=='SI':
                    si = predictiveNet.TrainingSaver.SI.item().squeeze()
                    if args.SR.predictive_net.mask_bottom:
                        mask_indices = np.argsort(si)[:int(args.SR.predictive_net.mask*len(si))]
                    else:
                        mask_indices = np.argsort(si)[-int(args.SR.predictive_net.mask*len(si)):]
                elif args.SR.predictive_net.mask_type=='EV':
                    evs = predictiveNet.TrainingSaver.EVs.item()
                    if args.SR.predictive_net.mask_bottom:
                        mask_indices = np.argsort(evs)[:int(args.SR.predictive_net.mask*len(evs))]
                    else:
                        mask_indices = np.argsort(evs)[-int(args.SR.predictive_net.mask*len(evs)):]
                elif args.SR.predictive_net.mask_type=='random':
                    np.random.seed(args.exp.seed+1234)
                    mask_indices = np.random.choice(np.arange(predictiveNet.pRNN.hidden_size),
                                                    size=int(args.SR.predictive_net.mask*predictiveNet.pRNN.hidden_size))
                else:
                    raise ValueError("Mask type not recognized")
                print(f"Masking {len(mask_indices)} neurons")
            print("pRNN model loaded\n")
        else:
            predictiveNet = PredictiveNet(env,
                                            hidden_size = args.SR.cells,
                                            pRNNtype = args.SR.predictive_net.pRNNtype,
                                            learningRate = args.SR.predictive_net.lr,
                                            bias_lr = args.SR.predictive_net.bias_lr,
                                            bptttrunc = args.SR.predictive_net.bptttrunc,
                                            weight_decay = args.SR.predictive_net.weight_decay,
                                            neuralTimescale = args.SR.predictive_net.ntimescale,
                                            dropp = args.SR.predictive_net.dropout,
                                            trainNoiseMeanStd = (args.SR.predictive_net.noisemean,
                                                                args.SR.predictive_net.noisestd),
                                            f = args.SR.predictive_net.sparsity,
                                            wandb_log=True)
            print("pRNN model initialized\n")

        return predictiveNet, env, mask_indices
        # predictiveNet.pRNN.to(device)
        # predictiveNet.env_shell.hd_trans = np.array([-1,1,0,0])

    def run_training_loop(self):

        args = self.params

        RLutils.seed(args.exp.seed)

        print(f"Device: {device}\n")

        # Load environment
        env_key = args.exp.env_name
        env = RLutils.make_env(
                               env_key=env_key,
                               input_type=args.exp.input_type,
                               spatial_config=args.SR,
                               seed=args.exp.seed + 10000,
                               vid_folder=self.video_dir,
                               vid_n_episodes=args.logging.video_log_freq,
                               act_enc = args.SR.action_encoding
                                )
        print("Environment loaded\n")


        # Create random agent for analysis and goals-generation
        randomagent = create_agent(envname=env_key, env=env,
                                   agentkey='RandomActionAgent')

        # Load training status

        try:
            status = RLutils.get_status(self.model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        print("Training status loaded\n")

        # Load observations preprocessor

        obs_space, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)
        # if "vocab" in status:
        #     preprocess_obss.vocab.load_vocab(status["vocab"])
        print("Observations preprocessor loaded\n")

        # Load pRNN
        if args.SR.predictive_net:
            predictiveNet, env, args.SR.mask_indices = self._initialize_predictive_net(args, env)
            args.SR.cells = predictiveNet.hidden_size
        else:
            predictiveNet = None
            
        prnn_eval_bool = args.exp.offpolicy_prnn_eval or args.exp.onpolicy_prnn_eval

        # Load models
        if args.exp.goal_conditioned:
            acmodel = ACModelSR(obs_space, env.action_space,
                                args.SR.cells*2, args.exp.with_obs,
                                args.exp.rgb, args.exp.with_HD)
        elif args.SR.spatial:
            acmodel = ACModelSR(obs_space, env.action_space,
                                args.SR.cells, args.exp.with_obs,
                                args.exp.rgb, args.exp.with_HD)

        else:
            acmodel = ACModel(obs_space, env.action_space, args.exp.with_HD,
                              args.exp.rgb)

        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
            print("Existing model found")
        acmodel.to(device)
        print("AC model loaded\n")

        # Load algo
        if args.exp.goal_conditioned:
            EFS = EnvironmentFeaturesAnalysis(env, randomagent,
                                              prnn_model=predictiveNet,
                                              timesteps=15000)
            goal_pool = EFS.data
            if args.exp.exclude_goals:
                exclude_x = np.arange(args.exp.x_min, args.exp.x_max + 1)
                exclude_y = np.arange(args.exp.y_min, args.exp.y_max + 1)
                exclude = np.stack(np.meshgrid(exclude_x, exclude_y, indexing="ij"), axis=-1).reshape(-1, 2)
            else:
                exclude = None
            algo = GoalConditionedPPOAlgo(
                env=env,
                acmodel=acmodel,
                predictiveNet=predictiveNet,
                device=device,
                preprocess_obss=preprocess_obss,
                ppo_config=args.ppo,
                spatial_config=args.SR,
                reward_config=args.rewards,
                goal_pool=goal_pool,
                goal_threshold=args.exp.goal_threshold,
                check_location=args.exp.check_location,
                exclude_loactions=exclude
                )
        else:
            algo = PredictivePPOAlgo(
                env=env,
                acmodel=acmodel,
                predictiveNet=predictiveNet,
                device=device,
                preprocess_obss=preprocess_obss,
                ppo_config=args.ppo,
                spatial_config=args.SR,
                reward_config=args.rewards,
                )


        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        print("Optimizer loaded\n")

        # Train model

        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        n_performance = 0
        error_map = None

        while num_frames < args.exp.steps:
            # Update model parameters
            update_start_time = time.time()

            if args.exp.random_action_agent:
                # Random agent: train pRNN with random exploration
                processed_logs = algo.randomAgent_collect_exp_and_update(randomagent)
            else:
                # Regular agent: collect experiences and update parameters
                exps = algo.collect_experiences()
                algo.update_parameters(exps, update_params=not args.exp.random_init_control)
                
                # Process stored logs into final metrics
                processed_logs = algo.process_logs()
        
            update_end_time = time.time()

            num_frames += processed_logs["num_frames"]
            update += 1

            # Log metrics

            if update % args.logging.log_interval == 0:
                # Add script-specific metrics
                fps = processed_logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                
                processed_logs["frames"] = num_frames
                processed_logs["FPS"] = fps
                processed_logs["duration"] = duration
                
                # Log to wandb
                wandb.log(processed_logs)

            # Do analysis

            if (update < args.logging.initial_analysis_step or 
                (args.logging.analysis_interval > 0 and 
                 update % args.logging.analysis_interval == 0)):
                print('Starting analysis at step {}'.format(update))
                EFS = EnvironmentFeaturesAnalysis(env, randomagent, acmodel, predictiveNet, 20000)
                if args.rewards.internal_enabled and not error_map:
                    error_map = EFS.error_map(*env.env.target_pos, HDs=False)
                    error_map.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                    paper_bgcolor='rgba(0, 0, 0, 0)')
                    error_map.write_image(self.model_dir+"/"+str(update)+"_errors.png")
                    print('Error map generated at step {}'.format(update))

                fig = EFS.policy_map()
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                  paper_bgcolor='rgba(0, 0, 0, 0)')
                fig.write_image(self.model_dir+"/"+str(update)+"_policy.png")
                print('Policy map generated at step {}'.format(update))

                fig = EFS.values_map(HDs=False)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                  paper_bgcolor='rgba(0, 0, 0, 0)')
                fig.write_image(self.model_dir+"/"+str(update)+"_values.png")
                print('Values map generated at step {}'.format(update))

                # OPA = OnPolicyAnalysis(algo, 20000)
                # fig = OPA.plot_advantages()
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_advantages.png")
                # fig = OPA.plot_deltas()
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_deltas_true.png")
                # fig = OPA.plot_deltas(zmin=-0.2)
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_deltas.png")

                if prnn_eval_bool:
                    if args.exp.onpolicy_prnn_eval:
                        
                        analysisagent = randomagent if args.exp.random_action_agent else ActorCriticAgent(env.action_space, 
                                                                                                          acmodel, 
                                                                                                          predictiveNet, 
                                                                                                          device)
                        
                        _, _, _ = predictiveNet.calculateSpatialRepresentation(env, analysisagent,
                                                                trainDecoder=True, trainHDDecoder = False,
                                                                saveTrainingData=False, bitsec= False,
                                                                calculatesRSA = True, sleepstd=0.03,
                                                                wandb_nameext='_onPolicy')  

                    if args.exp.offpolicy_prnn_eval:
                        
                        analysisagent = ActorCriticAgent(env.action_space, 
                                                         acmodel, 
                                                         predictiveNet, 
                                                         device) if args.exp.random_action_agent else randomagent

                        _, _, _ = predictiveNet.calculateSpatialRepresentation(env, analysisagent,
                                                                trainDecoder=True, trainHDDecoder = False,
                                                                saveTrainingData=False, bitsec= False,
                                                                calculatesRSA = True, sleepstd=0.03,
                                                                wandb_nameext='_offPolicy')  
                
                if args.exp.analyze_agent_behav:
                    opa = OnPolicyAnalysis(algo, timesteps=25600)
                    wandb.log({"MI_policy_eval": opa.mi})
                    RLutils.save_analysis_of_agent_behav(opa, self.model_dir, update)

            if args.logging.early_stop:
                return_mean = processed_logs.get('return_mean', 0)
                return_std = processed_logs.get('return_std', float('inf'))
                if return_mean > args.exp.opt_return and return_std < 0.05:
                    n_performance += 1
                    if n_performance == 50:
                        break

            # Save status

            if args.logging.save_interval > 0 and update % args.logging.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict() if not args.exp.random_action_agent else None, 
                        "optimizer_state": algo.optimizer.state_dict() if not args.exp.random_action_agent else None}
                RLutils.save_status(status, self.model_dir)
                if args.SR.train:
                    predictiveNet.saveNet(args.SR.predictive_net.pRNNtype, self.model_dir)
                print("Status saved")

@hydra.main(config_path="Configs", config_name="config")
def my_main(cfg: DictConfig):
    my_app(cfg)

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))

    # Add an environment variable for storage
    os.environ['RL_STORAGE'] = cfg.logging.logdir

    ###################
    ### RUN TRAINING
    ###################

    trainer = RL_Trainer(cfg)
    try:
        trainer.run_training_loop()
    finally:
        wandb.finish()



if __name__ == "__main__":
    my_main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:07:04 2025

@author: Alxec
"""





#%%
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.agent import RandomActionAgent
from prnn.utils.env import make_env
from prnn.utils.agent import create_agent
from prnn.utils.data import generate_trajectories, create_dataloader, mergeDatasets
from prnn.utils.figures import TrainingFigure
from prnn.utils.figures import SpontTrajectoryFigure
# from prnn.examples.Miniworld.VAE import VarAutoEncoder, VAE

from omegaconf import DictConfig, OmegaConf

import hydra
import wandb
import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os

import matplotlib
matplotlib.use('Agg')


@hydra.main(config_path="ConfigsRNN",
            config_name="config",
            version_base="1.1")
def main(config: DictConfig):
    # File Management 
    savename = config['prnn']['pRNNtype'] + '-' + \
        config['fm']['namext'] + '-s' + str(config['hparams']['seed'])
    figfolder = config['fm']['netsfolder'] + 'nets/' + \
        config['fm']['savefolder'] + '/trainfigs/' + savename
    
    # Logging
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run = wandb.init(
                    # set the wandb project where this run will be logged
                    project = config.logging.project,
                    # group = f"{params.exp.exp_name}_width{params.rl.pc_sd}",
                    name=savename,
                    id = f"{savename}_{date}",
                    dir = figfolder,
                    resume='allow',
                    # track hyperparameters and run metadata
                    config = config
                    )

    # Set seeds
    torch.manual_seed(config['hparams']['seed'])
    random.seed(config['hparams']['seed'])
    np.random.seed(config['hparams']['seed'])
        
    if config['env_agent']['env_package'] == 'miniworld_vae':
        if config['encoder']['do_train']:
            ae = VAE(
                    learning_rate=config["encoder"]["learning_rate"],
                    net_config=config["encoder"]["net_config"].values(),
                    in_channels=config["encoder"]["in_channels"],
                    latent_dim=config["encoder"]["latent_dim"],
                    kld_weight=config["encoder"]["kld_weight"],
                )
        else: # Load the pretrained VAE
            ae = VarAutoEncoder.load_from_checkpoint(
                checkpoint_path=config['fm']['encoder_path'],
                in_channels=config["encoder"]["in_channels"],
                net_config=config["encoder"]["net_config"].values()
            )
    else:
        ae = None

    if config['fm']['contin']:
        ##NOTE THIS IS NOT UPDATED TO USE DATALOADER/prnn package...
        predictiveNet = PredictiveNet.loadNet(config['fm']['loadfolder']+savename)
        if config['env_agent']['env_key'] == '':
            env = predictiveNet.loadEnvironment(config['fm']['load_env'])
            predictiveNet.addEnvironment(env)
        else:
            env = make_env(env_key=config['env_agent']['env_key'],
                           package=config['env_agent']['env_package'],
                           act_enc=config['env_agent']['actenc'],
                           riab_cfg=config['riab'],
                        #    repeats=np.array(config['env_agent']['repeats']),
                        #    multiply=config['env_agent']['multiply'],
                        #    encoder=ae
                           )
            predictiveNet.addEnvironment(env)
            agent = create_agent(config['env_agent']['env_key'],
                                 env,
                                 config['env_agent']['agent_key'],
                                 config['env_agent']['agent_name'])
    else:
        #Make the environment shell and the dataloader
        env = make_env(env_key=config['env_agent']['env_key'],
                       package=config['env_agent']['env_package'],
                       act_enc=config['env_agent']['actenc'],
                    #    repeats=np.array(config['env_agent']['repeats']),
                    #    multiply=config['env_agent']['multiply'],
                    #    riab_cfg=config['riab'],
                    #    encoder=ae
                       )
        
        agent = create_agent(config['env_agent']['env_key'],
                             env,
                             config['env_agent']['agent_key'],
                             config['env_agent']['agent_name'])
        if config['fm']['use_dataloader']:
            tmpfolder = os.path.expandvars('${SLURM_TMPDIR}')
            create_dataloader(env=env, agent=agent, n_trajs=config['data']['datasetSize'],
                              folder=config['fm']['datasetfolder'],
                              tmp_folder=tmpfolder,
                              batch_size=config['data']['batch_size'],
                              seq_length=config['env_agent']['seqdur'],
                              num_workers= config['data']['num_workers'],
                            #   save_raw=config['data']['save_raw'],
                            #   load_raw=config['encoder']['encode'] or 'AE' in config['prnn']['pRNNtype'],
                              )
            
        #Make the predictive net
        predictiveNet = PredictiveNet(env, 
                                      hidden_size=config['prnn']['hiddensize'],
                                      pRNNtype=config['prnn']['pRNNtype'],
                                      learningRate = config['hparams']['lr'],
                                      weight_decay = config['hparams']['weight_decay'],
                                      trainNoiseMeanStd= (config['hparams']['noisemean'],
                                                          config['hparams']['noisestd']),
                                      trainBias = config['prnn']['trainBias'],
                                      bias_lr = config['hparams']['bias_lr'],
                                      eg_lr = config['hparams']['eg_lr'],
                                    #   ae_lr = config['hparams']['ae_lr'],
                                      eg_weight_decay=config['hparams']['eg_weight_decay'],
                                      identityInit = config['prnn']['identityInit'],
                                      dataloader=config['fm']['use_dataloader'],
                                      bptttrunc = config['hparams']['bptttrunc'],
                                      neuralTimescale = config['prnn']['ntimescale'],
                                      dropp=config['hparams']['dropout'],
                                      f = config['hparams']['f'],
                                      mean_std_ratio = config['prnn']['mean_std_ratio'],
                                      sparsity = config['prnn']['sparsity'],
                                      fig_type='pdf',
                                    #   train_encoder=config['encoder']['do_train'],
                                    #   encoder_grad=config['encoder']['pass_grad'],
                                    #   enc_loss_weight= config['encoder']['loss_weight'],
                                    #   enc_loss_power= config['encoder']['loss_power'],
                                    #   latent_dim= config['prnn']['latent_dim'],
                                      )
        predictiveNet.seed = config['hparams']['seed']
        predictiveNet.trainArgs = OmegaConf.to_container(config)
        predictiveNet.plotSampleTrajectory(env,agent,
                                        savename=savename+'exTrajectory_untrained',
                                        savefolder=figfolder)
        predictiveNet.savefolder = config['fm']['savefolder']
        predictiveNet.savename = savename
        print('Predictive Net Created')



    # Training Epoch
    numepochs = config['data']['numepochs']
    sequence_duration = config['env_agent']['seqdur']
    num_trials = config['data']['numtrials']

    predictiveNet.trainingCompleted = False
    if predictiveNet.numTrainingTrials == -1:
        #Calculate initial spatial metrics etc
        print('Training Baseline')
        predictiveNet.trainingEpoch(env, agent,
                                sequence_duration=sequence_duration,
                                num_trials=1)
        print('Calculating Spatial Representation...')
        place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                    trainDecoder=True,saveTrainingData=True,
                                                    bitsec= False,
                                                    calculatesRSA = True, sleepstd=0.03)
        predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)
        print('Calculating Decoding Performance...')
        predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                    savename=savename, savefolder=figfolder,
                                                    saveTrainingData=True)
        
    while predictiveNet.numTrainingEpochs<numepochs:
        print(f'Training Epoch {predictiveNet.numTrainingEpochs}')
        predictiveNet.trainingEpoch(env, agent,
                                sequence_duration=sequence_duration,
                                num_trials=num_trials)
        print('Calculating Spatial Representation...')
        place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                    trainDecoder=True,
                                                    # trainHDDecoder = True,
                                                    saveTrainingData=True, bitsec= False,
                                                    calculatesRSA = True, sleepstd=0.03)
        print('Calculating Decoding Performance...')
        predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                    savename=savename, savefolder=figfolder,
                                                    saveTrainingData=True)
        predictiveNet.plotLearningCurve(savename=savename,savefolder=figfolder,
                                        incDecode=True)
        #predictiveNet.plotSampleTrajectory(env,agent,savename=savename,savefolder=figfolder)
        predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)

        
        plt.show()
        plt.close('all')
        predictiveNet.saveNet(config['fm']['savefolder']+'/'+savename,
                              savefolder = config['fm']['netsfolder'])

    predictiveNet.trainingCompleted = True
    TrainingFigure(predictiveNet,savename=savename,savefolder=figfolder)

    #If the user doesn't want to save all that training data, delete it except the last one
    if config['fm']['saveTrainData'] is False:
        predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(predictiveNet.TrainingSaver.index[:-1])
        predictiveNet.saveNet(config['fm']['savefolder']+savename,savefolder = config['fm']['netsfolder'])

if __name__ == "__main__":
    try:
        main()
    finally:
        wandb.finish()
        print("WandB run finished successfully.")
#!/usr/bin/env python
import argparse
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench, logger
from baselines.common.mpi_fork import mpi_fork

import os.path as osp
import gym, logging
from mpi4py import MPI
from gym import utils

from baselines import logger
import sys


def train(env_id, num_timesteps, timesteps_per_batch, seed, num_cpu, resume, 
          agentName, logdir, hid_size, filter_sizes,filter_nb,
		  filter_strides,filter_activation, im_channels,
		  im_dim, other_input_length, clip_param, entcoeff, 
          optim_epochs, optim_stepsize, optim_batchsize, gamma, lam,
          portnum
):
    from baselines.ppo1 import golf_cnn_policy, pposgd_golf
    print("num cpu = " + str(num_cpu))

    whoami  = mpi_fork(num_cpu)
    if whoami == "parent": return
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()

    if rank != 0: logger.set_level(logger.DISABLED)
    utils.portnum = portnum + rank

    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(seed)

    if logger.get_dir():
        env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))

    def policy_fn(name, ob_space, ac_space):
        return golf_cnn_policy.golf_cnn_policy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, filter_sizes=filter_sizes,filter_nb=filter_nb,
			filter_strides=filter_strides,filter_activation=filter_activation, im_channels=im_channels,
			im_dim=im_dim, other_input_length=other_input_length)

    gym.logger.setLevel(logging.WARN)
    pposgd_golf.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=timesteps_per_batch,
            clip_param=clip_param, entcoeff=entcoeff,
            optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
            gamma=gamma, lam=lam,
            resume=resume, agentName=agentName, logdir=logdir
        )
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Custom0-v0') #'Humanoid2-v1') # 'Walker2d2-v1')
    parser.add_argument('--num-cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--logdir', type=str, default='.') #default=None)
    parser.add_argument('--agentName', type=str, default='PPO-Agent')
    parser.add_argument('--resume', type=int, default = 0)

    parser.add_argument('--num_timesteps', type=int,default = 1e8)
    parser.add_argument('--timesteps_per_batch', type=int, default=5400)
	
    parser.add_argument('--hid_size',nargs='+', type=int, default=[32,32])
    parser.add_argument('--cnn_filter_sizes',nargs='+', type=int, default=[3,3,3])
    parser.add_argument('--cnn_filter_nb',nargs='+', type=int, default=[5,5,5])
    parser.add_argument('--cnn_filter_strides',nargs='+', type=int, default=[1,1,1])
    parser.add_argument('--cnn_filter_activation',type=str, default='selu')
    
    parser.add_argument('--im_channels', type=int, default=1)
    parser.add_argument('--im_dim',nargs='+', type=int, default=[32,32])
    parser.add_argument('--other_input_length', type=int, default = 23) #baxter golf
	

    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entcoeff', type=float, default=0.0)
    parser.add_argument('--optim_epochs', type=int, default=20)
    parser.add_argument('--optim_stepsize', type=float, default=3e-4)
    parser.add_argument('--optim_batchsize', type=int, default=384)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)

    parser.add_argument("--portnum", required=False, type=int, default=5000)
    parser.add_argument("--server_ip", required=False, default="localhost")

    return vars(parser.parse_args())

def main():
    args = parse_args()
    utils.portnum = args['portnum']
    utils.server_ip = args['server_ip']
    del args['portnum']
    del args['server_ip']

    train(args['env_id'], 
	      num_timesteps=args['num_timesteps'], 
		  timesteps_per_batch=args['timesteps_per_batch'],
          seed=args['seed'], 
		  num_cpu=args['num_cpu'], 
		  resume=args['resume'], 
		  agentName=args['agentName'], 
          logdir=args['logdir'], 
		  hid_size=args['hid_size'], 
		  filter_sizes=args['cnn_filter_sizes'],
		  filter_nb=args['cnn_filter_nb'],
		  filter_strides=args['cnn_filter_strides'],
		  filter_activation=args['cnn_filter_activation'],
		  im_channels=args['im_channels'],
		  im_dim=args['im_dim'],
		  other_input_length=args['other_input_length'],
          clip_param=args['clip_param'], 
		  entcoeff=args['entcoeff'],
          optim_epochs=args['optim_epochs'], 
		  optim_stepsize=args['optim_stepsize'], 
		  optim_batchsize=args['optim_batchsize'],
          gamma=args['gamma'], 
		  lam=args['lam'], 
		  portnum=utils.portnum,)


if __name__ == '__main__':
    main()
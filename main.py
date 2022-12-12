from pathlib import Path
import gym, d4rl
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from lightATAC.policy import GaussianPolicy
from lightATAC.value_functions import TwinQ, ValueFunction
from lightATAC.util import Log, set_seed
from lightATAC.bc import BehaviorPretraining
from lightATAC.atac import ATAC
from lightATAC.util import (evaluate_policy, sample_batch, traj_data_to_qlearning_data, tuple_to_traj_data, DEFAULT_DEVICE)

EPS=1e-6

def eval_agent(*, env, agent, discount, n_eval_episodes, max_episode_steps=1000,
               deterministic_eval=True, normalize_score=None):

    all_returns = np.array([evaluate_policy(env, agent, max_episode_steps, deterministic_eval, discount) \
                             for _ in range(n_eval_episodes)])
    eval_returns = all_returns[:,0]
    discount_returns = all_returns[:,1]

    info_dict = {
        "return mean": eval_returns.mean(),
        "return std": eval_returns.std(),
        "discounted returns": discount_returns.mean()
    }

    if normalize_score is not None:
        normalized_returns = normalize_score(eval_returns)
        info_dict["normalized return mean"] = normalized_returns.mean()
        info_dict["normalized return std"] =  normalized_returns.std()
    return info_dict


def main(args):
    # ------------------ Initialization ------------------ #
    torch.set_num_threads(1)
    env = gym.make(args.env_name)  # d4rl ENV
    dataset = env.get_dataset()
    set_seed(args.seed, env=env)

    # Setup logger
    log_path = Path(args.log_dir) / args.env_name / ('_beta' + str(args.beta))
    log = Log(log_path, vars(args))
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

    # Assume vector observation and action
    obs_dim, act_dim = dataset['observations'].shape[1], dataset['actions'].shape[1]
    vf = ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    qf = TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, use_tanh=True).to(DEFAULT_DEVICE)
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)  # due to tanh
    rl = ATAC(
        policy=policy,
        qf=qf,
        optimizer=torch.optim.Adam,
        discount=args.discount,
        action_shape=act_dim,
        buffer_batch_size=args.batch_size,
        policy_lr=args.slow_lr,
        qf_lr=args.fast_lr,
        # ATAC main parameters
        beta=args.beta, # the regularization coefficient in front of the Bellman error
    )
    rl.to(DEFAULT_DEVICE)
    networks = dict(qf=qf, vf=vf, policy=policy)
    optimizers = [torch.optim.Adam(list(vf.parameters())+list(policy.parameters())+list(qf.parameters()), lr=args.fast_lr)]

    # ------------------ Pretraining ------------------ #

    # Train policy and value to fit the behavior data
    bp = BehaviorPretraining(networks, optimizers, lambd=0.99, discount=args.discount)
    traj_data = bp.train(tuple_to_traj_data(dataset), args.n_warmstart_steps, log_fun= lambda x: print(x))

    # Main Training
    for step in trange(args.n_steps):
        train_metrics = rl.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % max(int(args.eval_period/10),1) == 0:
            print(train_metrics)
            for k, v in train_metrics.items():
                writer.add_scalar('Train/'+k, v, step)
        if (step+1) % args.eval_period == 0:
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=lambda returns: d4rl.get_normalized_score(args.env_name, returns)*100.0)
            log.row(eval_metrics)
            for k, v in eval_metrics.items():
                writer.add_scalar('Eval/'+k, v, step)
    # Final processing
    torch.save(rl.state_dict(), log.dir/'final.pt')
    log.close()
    writer.close()
    return eval_metrics['normalized return mean']



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fast_lr', type=float, default=5e-4)
    parser.add_argument('--slow_lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--n_warmstart_steps', type=int, default=100*10**3)
    main(parser.parse_args())
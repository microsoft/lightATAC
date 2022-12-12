import copy, csv, json, random, string, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Methods for processing trajectories

def traj_to_tuple_data(traj_data, ignores=("metadata",)):
    """Concatenate a list of trajectory dicts to a dict of np.arrays of the same length."""
    tuple_data = dict()
    for k in traj_data[0].keys():
        if not any([ig in k for ig in ignores]):
            tuple_data[k] = np.concatenate([traj[k] for traj in traj_data])
    return tuple_data


def tuple_to_traj_data(tuple_data, ignores=("metadata",)):
    """Split a tuple_data dict in d4rl format to list of trajectory dicts."""
    tuple_data["timeouts"][-1] = not tuple_data["terminals"][-1]
    ends = (tuple_data["terminals"] + tuple_data["timeouts"]) > 0
    ends[-1] = False  # don't need to split at the end

    inds = np.arange(len(ends))[ends] + 1
    tmp_data = dict()
    for k, v in tuple_data.items():
        if not any([ig in k for ig in ignores]):
            tmp_data[k] = np.split(v, inds)
    traj_data = [
        dict(zip(tmp_data, t)) for t in zip(*tmp_data.values())
    ]  # convert to list of dict
    return traj_data


def traj_data_to_qlearning_data(traj_data, ignores=("metadata",)):
    """Convert a list of trajectory dicts into d4rl qlearning data format."""
    traj_data = copy.deepcopy(traj_data)
    for traj in traj_data:
        # process 'observations'
        if traj["terminals"][-1] > 0:
            traj["observations"] = np.append(
                traj["observations"], traj["observations"][-1:], axis=0
            )  # duplicate
        else:  # ends because of timeout
            for k, v in traj.items():
                if k != "observations":
                    traj[k] = v[:-1]
        # At this point, traj['observations'] should have one more element than the others.
        traj["next_observations"] = traj["observations"][1:]
        traj["observations"] = traj["observations"][:-1]
        lens = [len(v) for k, v in traj.items()]
        assert all([lens[0] == l for l in lens[1:]])

    return traj_to_tuple_data(traj_data, ignores=ignores)


def discount_cumsum(x, discount):
    """Discounted cumulative sum.
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.
    Returns:
        np.ndarrary: Discounted cumulative sum.
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=-1)[::-1]



# Below are modified from gwthomas/IQL-PyTorch

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, *inputs):
    """
        Args:
            f : The function to evaluate, which returns a tensor or a list of tensors.

            Suppose output = f(arg1, arg2), and we have list_arg1, list_arg2 which we wish to batch, where
            we assume len(list_arg_1) = len(list_arg_2). Then
                tuple_outputs = compute_batched(f, list_arg_1, list_arg_2)
            where len(tuple_outputs) = len(list_arg_1).

        Returns:
            A tuple of the original outputs of f.

    """
    # return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])
    if len(inputs)>1:  # assert the number of
        lens =  [len(xs) for xs in inputs]
        assert all(x == lens[0] for x in lens)

    outputs = f(*[torch.cat(xs, dim=0) for xs in inputs])
    if torch.is_tensor(outputs):
        return outputs.split([len(x) for x in inputs[0]])
    else:  # suppose that's iterable.
        outputs = (o.split([len(x) for x in inputs[0]]) for o in outputs)
        return tuple(zip(*outputs))


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), DEFAULT_DEVICE  # dataset[k].device
    for v in dataset.values():
        assert len(v) == n, "Dataset values must have same length"
    indices = np.random.randint(low=0, high=n, size=(batch_size,))  # , device=device)
    return {k: torchify(v[indices]) for k, v in dataset.items()}


def evaluate_policy(env, policy, max_episode_steps, deterministic=True, discount = 0.99):
    obs = env.reset()
    total_reward = 0.
    discount_total_reward = 0.
    for i in range(max_episode_steps):
        with torch.no_grad():
            try:
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            except:
                action = policy.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        discount_total_reward += reward * discount**i
        if done:
            break
        else:
            obs = next_obs
    return [total_reward, discount_total_reward]


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'


class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()
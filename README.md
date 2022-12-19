# lightATAC

This is a lightweight reimplementation of **Adversarially Trained Actor Critic** ([ATAC](https://github.com/microsoft/ATAC)), a model-free offline reinforcement learning algorithm with SoTA performance on [D4RL](https://github.com/Farama-Foundation/D4RL) by Ching-An Cheng*, Tengyang Xie*, Nan Jiang, and Alekh Agarwal (<https://arxiv.org/abs/2202.02446>).

To install, simply clone the repo and run `pip install -e . `.  Then you can start the training by, e.g.,

    python main.py --log_dir ./tmp_results --env_name hopper-medium-expert-v2 --beta 1.0

More instructions can be found in `main.py`, and please see the [original paper](https://arxiv.org/abs/2202.02446) for hyperparameters (e.g., `beta`). The code was tested with python 3.9.


This reimplementation is based on [gwthomas/IQL-PyTorch](https://github.com/gwthomas/IQL-PyTorch). It is minimalistic, so users can easily modify it for their needs. It follows mostly the logic in the original [ATAC](https://github.com/microsoft/ATAC) code, but with some code optimization leading to 1.5X-2X speed up.
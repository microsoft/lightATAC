# lightATAC

This is a lightweight reimplementation of **Adversarially Trained Actor Critic** ([ATAC](https://github.com/microsoft/ATAC)), a model-free offline reinforcement learning algorithm with SoTA performance on [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark.

To install, simply clone the repo and  run `pip install -e . `. It uses mujoco210, which can be installed, if needed, following the commands below.

```
bash install_mujoco.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia" >> ~/.bashrc
source ~/.bashrc
```



 Then you can start the training by running, e.g.,

    python main.py --log_dir ./tmp_results --env_name hopper-medium-expert-v2 --beta 1.0

More instructions can be found in `main.py`, and please see the [original paper](https://arxiv.org/abs/2202.02446) for hyperparameters (e.g., `beta`). The code was tested with python 3.9.


This reimplementation is based on [gwthomas/IQL-PyTorch](https://github.com/gwthomas/IQL-PyTorch). It is minimalistic, so users can easily modify it for their needs. It follows mostly the logic in the original [ATAC](https://github.com/microsoft/ATAC) code, but with some code optimization, which gives about 1.5X-2X speed up.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
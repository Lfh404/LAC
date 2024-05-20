# Langevin Policy for Safe Reinforcement Learning

This repository provides an implementation of Langevin Actor-Critic (LAC).


## Requirements
- [PyTorch 2.1.2](https://pytorch.org/)
- [Gym 0.21.0](https://github.com/openai/gym)
- [MuJoCo 2.3.3](https://github.com/deepmind/mujoco)
- [Safety Gym](https://github.com/openai/safety-gym.git)
- [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)

## Usage
Example Safexp-CarGoal1-v0 tasks
```
python main.py --env-id Safexp-CarGoal1-v0 --constraint safety --seed 0
```

## Citation
If you find our work helpful, please cite:

    @inproceedings{
        lei2024langevin,
        title={Langevin Policy for Safe Reinforcement Learning},
        author={Lei, Fenghao and Yang, Long and Wen, Shiting and Huang, Zhixiong and Zhang, Zhiwang and Pang, Chaoyi},
        booktitle={Forty-first International Conference on Machine Learning},
        year={2024},
        url={https://openreview.net/forum?id=xgoilgLPGD}
    }
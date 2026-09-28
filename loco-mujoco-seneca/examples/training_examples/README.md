
### Training Examples

Within this directory, you can find different imitation and reinforcement learning training examples:

- `jax_rl`: Reinforcement Learning example.
- `jax_gail`: GAIL (Generative Adversarial Imitation Learning) example.
- `jax_amp`: AMP (Adversarial Motion Prior) example.
- `jax_rl_mimic`: DeepMimic imitation learning example.

We use WandB to log the training process and results. So you have to install this dependency to run the training examples:

```bash
pip install wandb
```

Each example has its own `README.md` file, which provides more detailed instructions on how to
run the training and evaluation process.

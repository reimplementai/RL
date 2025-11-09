import torch

config = {
    'data_prefix': '_math',

    'vocab_size': 17,
    'block_size': 20,
    'n_layer': 6,
    'n_head': 4,
    'n_embd': 32,
    'dropout': 0.1,
    'bias': False,

    'batch_size': 100,
    'learning_rate': 3e-4,
    'max_iters': 100000,
    'eval_interval': 1000,

    'pad_token': 0,
    'BOS': 1,
    'EOS': 2,

    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'ckpt': 'checkpoints/smallgpt-interrupt.pt',
}
# rlvr - Reinforcement Learning with Verifiable Rewards

rlvr/RL0.py implements RLVR with 0 training data, only rollouts.  A small transformer model tries to perform 2 digit addition by trial and error.

The model is given prompts of the form "25+24=" and has to fill in the next tokens.
The next tokens can be digits and numbers.

For example, the model might fill in the next tokens as "48" or "49=" or "49+" etc.
The model is then rewarded for the correct answer using a reward function defined in rlvr/rewards_math.py.

See GRPO, Dr GRPO, DAPO and Tulu for more context:

Dr GRPO: https://arxiv.org/pdf/2503.20783

DAPO: https://arxiv.org/pdf/2503.14476

TULU: https://arxiv.org/pdf/2411.15124

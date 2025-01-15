# README

## Overview
This project implements a Deep Q-Network (DQN) for reinforcement learning tasks using PyTorch. The code is modular and supports multiple DQN variants, including standard DQN, Double DQN (DDQN), and Dueling DQN. The agent is tested on OpenAI Gym environments, such as `LunarLander-v2`, with customizable hyperparameters and configurations.

---

## Features
- **Multiple DQN Variants**: Standard DQN, DDQN, and Dueling DQN.
- **Environment Support**: Includes setups for `CartPole-v1`, `MountainCar-v0`, and `LunarLander-v2`.
- **GPU Acceleration**: Utilizes CUDA if available.
- **Model Checkpoints**: Automatically saves and loads model checkpoints.
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation.
- **TensorBoard Logging**: Tracks rewards during training.

---

## Requirements
### Python Environment
- Python 3.8+

### Dependencies
Install the required Python packages:
```bash
pip install torch gymnasium tensorboard numpy
```

---

## Usage
### Running the Code
Simply execute the script:
```bash
python DQN_new.py
```

### Key Hyperparameters
You can modify the following hyperparameters in the script:
- `EPISODES`: Number of episodes to train/test.
- `BATCH_SIZE`: Size of training batches.
- `LR`: Learning rate for the optimizer.
- `GAMMA`: Discount factor for future rewards.
- `EPSILON`: Exploration rate for epsilon-greedy policy.
- `MEMORY_CAPACITY`: Replay memory size.
- `MIN_CAPACITY`: Minimum replay memory size to start training.

### Model Variants
Set the `choice` variable to select a DQN variant:
- `1`: Double DQN with standard DQN architecture.
- `2`: Double DQN with dueling architecture.
- `3`: Standard DQN with standard architecture.
- `4`: Standard DQN with dueling architecture (default).

### Environment Selection
Set the `env_id` variable to select the environment:
- `1`: `CartPole-v1`
- `2`: `MountainCar-v0`
- `3`: `LunarLander-v2` (default)

### Testing a Pre-trained Model
Set `TEST = True` and provide the path to the pre-trained model in `MODEL_PATH`. For example:
```python
MODEL_PATH = '3_DQN_DuelingDQN/566000.pth'
```

---

## File Structure
- **Main Script**: Contains the main DQN implementation.
- **Classes**:
  - `Model`: Defines the standard DQN architecture.
  - `DuelingModel`: Defines the dueling DQN architecture.
  - `Memory`: Implements experience replay.
  - `DQN`: Handles learning, action selection, and training logic.

---

## Implementation Details
1. **Replay Memory**: Experience replay is implemented using a deque with a maximum size (`MEMORY_CAPACITY`).
2. **Training Loop**:
   - Stores experiences (`state`, `action`, `reward`, `next_state`, `done`) into memory.
   - Learns by sampling a batch of transitions once memory reaches `MIN_CAPACITY`.
   - Updates the target network every `Q_NETWORK_ITERATION` steps.
3. **Loss Function**: Uses Mean Squared Error (MSE) between the Q-value predictions and the target Q-values.
4. **Optimization**: Updates the model parameters using Adam optimizer.

---

## Logging and Checkpoints
- **TensorBoard**: Logs the episode rewards. Run the following command to visualize:
  ```bash
  tensorboard --logdir=save_results
  ```
- **Checkpoint Saving**: Saves the model at regular intervals (`SAVING_IETRATION`) to the directory `save_results/ckpt/`.

---


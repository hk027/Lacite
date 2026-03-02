{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Deep Q-Learning (DQN) avec explications\n",
    "Ce notebook explique étape par étape le fonctionnement du DQN et fournit un exemple complet sur l'environnement CartPole-v1 de Gym."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1️⃣ Importation des bibliothèques"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["import gym\n", "import random\n", "import numpy as np\n", "from collections import deque\n", "import torch\n", "import torch.nn as nn\n", "import torch.optim as optim"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2️⃣ Hyperparamètres"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["ENV_NAME = \"CartPole-v1\"\n", "GAMMA = 0.99\n", "LR = 0.001\n", "BATCH_SIZE = 64\n", "MEMORY_SIZE = 10000\n", "EPSILON_START = 1.0\n", "EPSILON_END = 0.01\n", "EPSILON_DECAY = 0.995\n", "TARGET_UPDATE = 10\n", "EPISODES = 500"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3️⃣ Définition du Q-Network"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["class QNetwork(nn.Module):\n", "    def __init__(self, state_size, action_size):\n", "        super(QNetwork, self).__init__()\n", "        self.fc1 = nn.Linear(state_size, 64)\n", "        self.fc2 = nn.Linear(64, 64)\n", "        self.fc3 = nn.Linear(64, action_size)\n", "    def forward(self, x):\n", "        x = torch.relu(self.fc1(x))\n", "        x = torch.relu(self.fc2(x))\n", "        return self.fc3(x)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4️⃣ Replay Memory"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["class ReplayMemory:\n", "    def __init__(self, capacity):\n", "        self.memory = deque(maxlen=capacity)\n", "    def push(self, transition):\n", "        self.memory.append(transition)\n", "    def sample(self, batch_size):\n", "        return random.sample(self.memory, batch_size)\n", "    def __len__(self):\n", "        return len(self.memory)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5️⃣ Initialisation de l'environnement et des réseaux"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["env = gym.make(ENV_NAME)\n", "state_size = env.observation_space.shape[0]\n", "action_size = env.action_space.n\n", "policy_net = QNetwork(state_size, action_size)\n", "target_net = QNetwork(state_size, action_size)\n", "target_net.load_state_dict(policy_net.state_dict())\n", "target_net.eval()\n", "optimizer = optim.Adam(policy_net.parameters(), lr=LR)\n", "memory = ReplayMemory(MEMORY_SIZE)\n", "epsilon = EPSILON_START"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 6️⃣ Fonction pour choisir l'action (ε-greedy)"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["def select_action(state, epsilon):\n", "    if random.random() < epsilon:\n", "        return random.randrange(action_size)\n", "    else:\n", "        with torch.no_grad():\n", "            state = torch.FloatTensor(state)\n", "            return policy_net(state).argmax().item()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 7️⃣ Fonction d'entraînement"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["def train():\n", "    if len(memory) < BATCH_SIZE:\n", "        return\n", "    transitions = memory.sample(BATCH_SIZE)\n", "    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)\n", "    batch_state = torch.FloatTensor(batch_state)\n", "    batch_action = torch.LongTensor(batch_action).unsqueeze(1)\n", "    batch_reward = torch.FloatTensor(batch_reward)\n", "    batch_next_state = torch.FloatTensor(batch_next_state)\n", "    batch_done = torch.FloatTensor(batch_done)\n", "    current_q = policy_net(batch_state).gather(1, batch_action)\n", "    next_q = target_net(batch_next_state).max(1)[0].detach()\n", "    expected_q = batch_reward + (1 - batch_done) * GAMMA * next_q\n", "    loss = nn.MSELoss()(current_q.squeeze(), expected_q)\n", "    optimizer.zero_grad()\n", "    loss.backward()\n", "    optimizer.step()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 8️⃣ Boucle principale"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": ["for episode in range(EPISODES):\n", "    state = env.reset()[0]\n", "    total_reward = 0\n", "    done = False\n", "    while not done:\n", "        action = select_action(state, epsilon)\n", "        next_state, reward, terminated, truncated, _ = env.step(action)\n", "        done = terminated or truncated\n", "        memory.push((state, action, reward, next_state, float(done)))\n", "        state = next_state\n", "        total_reward += reward\n", "        train()\n", "    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)\n", "    if episode % TARGET_UPDATE == 0:\n", "        target_net.load_state_dict(policy_net.state_dict())\n", "    print(f\"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}\")\n", "env.close()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## ✅ Résumé et explications\n", "- Replay Memory: stocke les transitions pour un apprentissage stable\n", "- Target Network: stabilise la mise à jour des Q-values\n", "- ε-greedy: équilibre exploration/exploitation\n", "- Boucle principale: interaction avec l'environnement et apprentissage du DQN"]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5
}


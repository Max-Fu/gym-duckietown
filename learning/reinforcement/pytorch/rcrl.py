import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # this is the vanilla implementation
        # but we're using a slightly different one
        # x = self.max_action * self.tanh(self.lin2(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1])

        return x


class CriticCNN(nn.Module):
    def __init__(self, action_dim, prior_dim, output_activation=None):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim + prior_dim, 128)
        self.lin3 = nn.Linear(128, 1)
        if output_activation:
            self.output_activation = _str_to_activation(output_activation)

    def forward(self, states, actions, prior):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions, prior], 1)))  # c
        x = self.lin3(x)

        return x


class PriorCNN(nn.Module):
    def __init__(self, action_dim, prior_dim, output_activation=None):
        super(PriorCNN, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, prior_dim)
        if output_activation:
            self.output_activation = _str_to_activation[output_activation]

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)
        x = self.output_activation(x)

        return x


class RCRL(object):
    # adapted from DDPG file (in duckietown, we are only predicting the chance of collision, since other than that, we have limited rules)
    def __init__(self, state_dim, action_dim, max_action, prior_dim, lr_actor=1e-4, lr_critic=1e-3, lr_prior=1e-4):
        super(RCRL, self).__init__()
        print("Starting RCRL init")

        self.state_dim = state_dim
        self.prior_dim = prior_dim 
        self.flat = False
        self.actor = ActorCNN(action_dim, max_action).to(device)
        self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        print("Initialized Target+Opt [Actor]")
        self.critic = CriticCNN(action_dim, prior_dim).to(device)
        self.critic_target = CriticCNN(action_dim, prior_dim).to(device)
        print("Initialized Critic")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        print("Initialized Target+Opt [Critic]")
        self.prior_regressor = PriorCNN(action_dim, prior_dim, output_activation='sigmoid').to(device)
        self.prior_optimizer = torch.optim.Adam(self.prior_regressor.parameters(), lr=lr_prior)
        print("Initialized Prior+Opt")

    def predict(self, state):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):
    
        total_prior_loss = 0 
        total_critic_loss = 0 
        total_actor_loss = 0 

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)
            
            additional_t = torch.FloatTensor(sample["additional"][:,0]).to(device)
            additional_t = torch.reshape(additional_t, (-1, self.prior_dim))
            additional_tp1 = torch.FloatTensor(sample["additional"][:,1]).to(device)
            additional_tp1 = torch.reshape(additional_tp1, (-1, self.prior_dim))
            
            # Compute prior and optimize prior network 
            pred_additional = self.prior_regressor(state, action)
            prior_loss = F.mse_loss(pred_additional, additional_t)

            # Optimize the prior
            self.prior_optimizer.zero_grad()
            prior_loss.backward()
            self.prior_optimizer.step()
            
            # Detach pred_additional (no gradient into prior network)
            pred_additional = pred_additional.detach()

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state), additional_tp1)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action, pred_additional) # this can be additional_t

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state), pred_additional).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            total_prior_loss += prior_loss.detach().cpu().numpy()
            total_critic_loss += critic_loss.detach().cpu().numpy()
            total_actor_loss += actor_loss.detach().cpu().numpy()

        return {
            "prior_loss": total_prior_loss / iterations, 
            "critic_loss": total_critic_loss / iterations, 
            "actor_loss": total_actor_loss / iterations,
        }

    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.actor.state_dict(), "{}/{}_actor.pth".format(directory, filename))
        print("Saved Actor")
        torch.save(self.critic.state_dict(), "{}/{}_critic.pth".format(directory, filename))
        print("Saved Critic")

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("{}/{}_actor.pth".format(directory, filename), map_location=device)
        )
        self.critic.load_state_dict(
            torch.load("{}/{}_critic.pth".format(directory, filename), map_location=device)
        )

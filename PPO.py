import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')

print("Device set to : cpu")
print("============================================================================================")
torch.autograd.set_detect_anomaly(True)

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.is_terminals = []
        self.rewards = []

        self.v_pred_true = []
        self.v_pred = []
        self.f_phi_s = []
        self.function = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.is_terminals[:]
        del self.rewards[:]


        del self.v_pred_true[:]
        del self.v_pred[:]
        del self.f_phi_s[:]
        del self.function[:]


class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id = cur_id + cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, last_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.first_optim = True
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        
        if has_continuous_action_space :
            self.pol = nn.Sequential(
                            nn.Linear(state_dim + last_dim, 8),
                            nn.Tanh(),
                            nn.Linear(8, 8),
                            nn.Tanh(),
                            nn.Linear(8, action_dim),
                            nn.Tanh()
                        )
        else:
            self.pol = nn.Sequential(
                            nn.Linear(state_dim + last_dim, 8),
                            nn.Tanh(),
                            nn.Linear(8, 8),
                            nn.Tanh(),
                            nn.Linear(8, action_dim),
                            nn.Softmax(dim=-1)
                        )
        self.vf_true = nn.Sequential(
                        nn.Linear(state_dim, 32),
                        nn.Tanh(),
                        nn.Linear(32, 32),
                        nn.Tanh(),
                        nn.Linear(32, 1)
                    )
        
        self.vf_shaped = nn.Sequential(
                nn.Linear(state_dim+ last_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )

        
        self.f_phi = nn.Sequential(
                        nn.Linear(state_dim, 16),
                        nn.Tanh(),
                        nn.Linear(16,8),
                        nn.Tanh(),
                        nn.Linear(8, last_dim),
                        nn.Tanh()
                    ) 
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        vpred_true = self.vf_true(state)
        f_phi_s = self.f_phi(state)
        last_out = torch.concat([state, f_phi_s], axis=0)

        if self.has_continuous_action_space:
            action_mean = self.pol(last_out)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.pol(last_out)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        vpred = self.vf_shaped(last_out)

        return action.detach(), action_logprob.detach(), vpred.detach(), vpred_true.detach(), f_phi_s.detach()
    
    def evaluate(self, state, action):

        vpred_true = self.vf_true(state)
        f_phi_s = self.f_phi(state)
        last_out = torch.concat([state, f_phi_s], axis=1)
        
        if self.has_continuous_action_space:
            action_mean = self.pol(last_out)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.pol(last_out)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()

        vpred = self.vf_shaped(last_out) 

        return action_logprobs.cpu().flatten(),  dist_entropy.cpu().flatten(), vpred.cpu().flatten(), vpred_true.cpu().flatten(), f_phi_s.cpu().flatten()


class PPO:
    def __init__(self, state_dim, action_dim,last_dim, lr_actor, lr_critic, lr_f, gamma, lam, K_epochs, batch_size, eps_clip, entropy_coeff, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space
        self.optimize_policy = True

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.entropy_coeff = entropy_coeff
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.first_optim = True
        self.optim_batch_size =  batch_size
        self.buffer = RolloutBuffer()

        self.pi = ActorCritic(state_dim, action_dim, last_dim, has_continuous_action_space, action_std_init).to(device)
       
        self.policy_opt = torch.optim.Adam([
                        {'params': self.pi.pol.parameters(), 'lr': lr_actor,"eps":1e-5 },
                    ])
        self.critic_opt = torch.optim.Adam([
                        {'params': self.pi.vf_shaped.parameters(), 'lr': lr_critic, "eps":1e-5},
                    ])

        self.true_critic_opt = torch.optim.Adam([
                        {'params': self.pi.vf_true.parameters(), 'lr': lr_critic, "eps":1e-5},
                    ])

        self.f_opt = torch.optim.Adam([
                        {'params': self.pi.f_phi.parameters(), 'lr': lr_f, "eps":1e-5},
                    ])
        self.pi_old = ActorCritic(state_dim, action_dim,last_dim, has_continuous_action_space, action_std_init).to(device)
        self.pi_old.load_state_dict(self.pi.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.pi.set_action_std(new_action_std)
            self.pi_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def choose_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, vpred ,vpred_true , f_phi_s = self.pi_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.v_pred.append(vpred)
            self.buffer.v_pred_true.append(vpred_true)
            self.buffer.f_phi_s.append(f_phi_s)

            return action.detach().cpu().numpy().flatten(), vpred.detach().cpu().numpy().flatten(), vpred_true.detach().cpu().numpy().flatten() , f_phi_s.detach().cpu().numpy().flatten() 
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, vpred ,vpred_true ,f_phi_s = self.pi_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.v_pred.append(vpred)
            self.buffer.v_pred_true.append(vpred_true)
            self.buffer.f_phi_s.append(f_phi_s)

            return action.cpu().numpy().flatten(), vpred.cpu().numpy().flatten(), vpred_true.cpu().numpy().flatten(), f_phi_s.detach().cpu().numpy().flatten() 

    def ppo_update(self, **kwargs):
        self.ppo_update_policy( **kwargs)
        self.ppo_update_shaping_weight_func( **kwargs)

    def ppo_update_policy(self, **kwargs):
        truncation_size = len(self.buffer.states)-1

        seg = {"ob": torch.tensor([self.buffer.states[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
               "ac": torch.tensor([self.buffer.actions[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
                "old_logprobs": torch.tensor([self.buffer.logprobs[0].cpu().clone()  for _ in range(truncation_size)]),
               "rew": torch.zeros(truncation_size, dtype=float),
               "v_pred": torch.zeros(truncation_size, dtype=float),
               "done": torch.zeros(truncation_size, dtype=int),
               "F": torch.zeros(truncation_size, dtype=float),
               "f_phi_s": torch.zeros(truncation_size, dtype=float)
               }
        
        for t in range(truncation_size):
            seg["ob"][t] = self.buffer.states[t].cpu().clone()
            seg["ac"][t] = self.buffer.actions[t].cpu().clone()
            seg["old_logprobs"][t] = self.buffer.logprobs[t].cpu().clone()
            seg["rew"][t] = self.buffer.rewards[t]
            seg["done"][t] = self.buffer.is_terminals[t]
            seg["v_pred"][t] = self.buffer.v_pred[t]
            seg["f_phi_s"][t] = self.buffer.f_phi_s[t].cpu().clone()
            seg["F"][t] = self.buffer.function[t]
        
        vpred = torch.tensor(np.append(seg["v_pred"], kwargs.get("next_v_pred")))
        seg_done = seg["done"]
        seg_rewards = seg["rew"]
        seg_F = seg["F"]
        seg_f = seg["f_phi_s"]

        gae_lam = torch.empty(truncation_size, dtype = float)
        last_gae_lam = 0
        for t in reversed(range(truncation_size)):
            non_terminal = 1 - seg_done[t]
            delta = seg_rewards[t] + seg_f[t] * seg_F[t] + self.gamma  * vpred[t + 1] * non_terminal - vpred[t]
            gae_lam[t] = delta + self.gamma * self.lam * non_terminal * last_gae_lam
            last_gae_lam = gae_lam[t]
        
        seg["adv"] = gae_lam
        seg["td_lam_ret"] = seg["adv"] + seg["v_pred"]
        self.learn(ob=seg["ob"], ac=seg["ac"], adv=seg["adv"],
                             td_lam_ret=seg["td_lam_ret"],
                             f_phi_s=seg["f_phi_s"], old_logprobs= seg["old_logprobs"])
        self.switch_optimization()

    def ppo_update_shaping_weight_func(self, **kwargs):

        truncation_size = len(self.buffer.states)-1

        seg =  {"ob": torch.tensor([self.buffer.states[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
               "ac": torch.tensor([self.buffer.actions[0].cpu().clone().tolist()  for _ in range(truncation_size)]),
                "old_logprobs": torch.tensor([self.buffer.logprobs[0].cpu().clone()  for _ in range(truncation_size)]),
                "rew": torch.zeros(truncation_size, dtype=float),
                "v_pred_true": torch.zeros(truncation_size, dtype=float),
                "done": torch.zeros(truncation_size, dtype=int),
                "F": torch.zeros(truncation_size, dtype=float),
                "f_phi_s": torch.zeros(truncation_size, dtype=float)
                }

        for t in range(truncation_size):
            seg["ob"][t] = self.buffer.states[t].cpu().clone()
            seg["ac"][t] = self.buffer.actions[t].cpu().clone() 
            seg["old_logprobs"][t] = self.buffer.logprobs[t].cpu().clone()
            seg["rew"][t] = self.buffer.rewards[t] 
            seg["done"][t] = self.buffer.is_terminals[t]
            seg["v_pred_true"][t] = self.buffer.v_pred_true[t].cpu().clone()
            seg["f_phi_s"][t] = self.buffer.f_phi_s[t].cpu().clone()

        seg_done = seg["done"]
        seg_v_pred_true = torch.tensor(np.append(seg["v_pred_true"], kwargs.get("next_v_pred_true")))

        gae_lam = torch.empty(truncation_size, dtype = float)
        seg_rewards = seg["rew"]
        last_gae_lam = 0

        for t in reversed(range(truncation_size)):
            non_terminal = 1 - seg_done[t]          
            delta = seg_rewards[t] + self.gamma * seg_v_pred_true[t + 1] * non_terminal - seg_v_pred_true[t]
            gae_lam[t] = delta + self.gamma * self.lam * non_terminal * last_gae_lam
            last_gae_lam = gae_lam[t]

        seg["adv_true"] = gae_lam
        seg["ret_true"] = seg["adv_true"] + seg["v_pred_true"]

        self.learn(ob=seg["ob"], ac=seg["ac"], adv_true=seg["adv_true"],
                             ret_true=seg["ret_true"],
                             f_phi_s=seg["f_phi_s"], old_logprobs= seg["old_logprobs"])
        
        self.switch_optimization()
    
    
    def learn(self, **kwargs):
        if self.optimize_policy:
            self.update_policy(**kwargs)

        else:
            self.update_shaping_weight_func(**kwargs)
            
    def update_policy(self, **kwargs):
        bs = kwargs.get("ob")
        ba = kwargs.get("ac")
        batch_adv = kwargs.get("adv")
        batch_td_lam_ret = kwargs.get("td_lam_ret")
        batch_f_phi_s = kwargs.get("f_phi_s")
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()

        old_logprobs = kwargs.get("old_logprobs")

        d = Dataset(dict(bs=bs, 
                         ba=ba, 
                         badv=batch_adv,
                         bret=batch_td_lam_ret,
                         bf=batch_f_phi_s, 
                         old_logprobs = old_logprobs),

                         deterministic=False)
        self.pi_old.load_state_dict(self.pi.state_dict())
        
        batch_size = self.optim_batch_size or bs.shape[0]
        for _ in range(self.K_epochs):
            for batch in d.iterate_once(batch_size):
                atarg = batch["badv"]
                action_logprobs,  dist_entropy, vpred, _, _ = self.pi.evaluate(batch["bs"], batch["ba"])
                mean_ent = torch.mean(dist_entropy)

                ratio = torch.exp(action_logprobs - batch["old_logprobs"])
                # Finding Surrogate Loss  
                surr1 = torch.where(
                    torch.logical_or(torch.isinf(ratio * atarg ), torch.isnan(ratio * atarg)),
                    torch.zeros_like(ratio),
                    ratio * atarg
                    )
            
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * atarg
                # final loss of clipped objective PPO
                vf_loss = torch.mean(torch.square(vpred - batch["bret"])) 
                pol_surr = torch.mean(torch.minimum(surr1, surr2))
                pol_ent_pen = -1 * self.entropy_coeff * mean_ent
                total_loss = pol_surr + pol_ent_pen + vf_loss

                # take gradient step
                self.critic_opt.zero_grad()
                self.policy_opt.zero_grad()

                vf_loss.backward(retain_graph=True)
                total_loss.backward()

                # vf_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.pi.vf_shaped.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(self.pi.pol.parameters(), 1.0)

                self.critic_opt.step()
                self.policy_opt.step()
            

    def update_shaping_weight_func(self, **kwargs):

        bs = kwargs.get("ob")
        ba = kwargs.get("ac")
        batch_adv = kwargs.get("adv_true")
        batch_ret = kwargs.get("ret_true")
        batch_f_phi_s = kwargs.get("f_phi_s")
        batch_adv = (batch_adv - batch_adv.mean()) / batch_adv.std()
        old_logprobs = kwargs.get("old_logprobs")

        d = Dataset(dict(bs=bs,
                        ba=ba, 
                        badv=batch_adv, 
                        bret=batch_ret,
                        bf=batch_f_phi_s,
                        old_logprobs = old_logprobs),
                        deterministic=False)
        
        batch_size = self.optim_batch_size or bs.shape[0]
        self.pi_old.load_state_dict(self.pi.state_dict())

        for _ in range(self.K_epochs):
            for batch in d.iterate_once(batch_size):
                action_logprobs,  _, _, vpred_true, _ = self.pi.evaluate(batch["bs"], batch["ba"])

                atarg_true = batch["badv"]
                ratio_clip_param_f = 0.2
                ratio = torch.exp(action_logprobs - batch["old_logprobs"])

                vf_true_loss = torch.mean(torch.square(vpred_true - batch["bret"])) 
                surr1_f = torch.where(torch.logical_or(
                    torch.isinf(ratio * atarg_true),
                    torch.isnan(ratio * atarg_true)),
                    torch.zeros_like(ratio),
                    ratio * atarg_true)
                surr2_f = torch.clamp(ratio, 1-ratio_clip_param_f, 1+ratio_clip_param_f) * atarg_true
                f_loss = - torch.mean(torch.minimum(surr1_f, surr2_f))

                self.true_critic_opt.zero_grad()
                self.f_opt.zero_grad()

                vf_true_loss.backward()
                f_loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.pi.vf_true.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(self.pi.f_phi.parameters(), 10.0)

                self.true_critic_opt.step()
                self.f_opt.step()

        self.buffer.clear()
        

    def switch_optimization(self):
        self.optimize_policy = not self.optimize_policy

    def save(self, checkpoint_path):
        torch.save(self.pi_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.pi_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.pi.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       



import numpy as np
import torch
from safepo.algos.base import PolicyGradient
from safepo.algos.lagrangian_base import Lagrangian
from safepo.algos.policy_gradient import PG
from safepo.algos.trpo import TRPO
import safepo.common.mpi_tools as mpi_tools


class RCPO(TRPO, Lagrangian):
    def __init__(
            self,
            algo='rcpo', 
            cost_limit=25., 
            lagrangian_multiplier_init=0.001,
            lambda_lr=0.035, 
            lambda_optimizer='Adam',
            **kwargs
        ):
        super().__init__(
            self, 
            algo=algo,
            use_cost_value_function=True,
            **kwargs
        )

        

        Lagrangian.__init__(
            self, 
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init, 
            lambda_lr=lambda_lr, 
            lambda_optimizer=lambda_optimizer
        )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        loss_pi = (- ratio * data['adv']).mean()
        
        

        # ensure that lagrange multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += (ratio *  penalty * data['cost_adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = .5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def roll_out(self):
        """collect data and store to experience buffer."""
        o, ep_ret, ep_costs, ep_len = self.env.reset(), 0., 0., 0

        if self.use_reward_penalty:
            # Consider reward penalty parameter in reward calculation: r' = r - c
            assert hasattr(self, 'lagrangian_multiplier')
            assert hasattr(self, 'lambda_range_projection')
            penalty_param = self.lambda_range_projection(
                self.lagrangian_multiplier)
        else:
            penalty_param = 0
        
        # c_gamma_step = 0
        for t in range(self.local_steps_per_epoch):
            a, v, cv, logp = self.ac.step(
                torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, info = self.env.step(a)
            if self.env_id in ['Ant-v3', "Swimmer-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Walker2d-v3"]:
                if 'y_velocity' not in info:
                    c = np.abs(info['x_velocity'])
                else:
                    c = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
            else:
                c = info.get('cost', 0.)

            ep_ret += r
            if self.use_discount_cost_update_lag:
                ep_costs += (self.gamma ** ep_len) * c
            else: 
                ep_costs += c
            ep_len += 1

            # Save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buf
            self.buf.store(
                obs=o, act=a, rew=r, val=v, logp=logp, cost=c, cost_val=cv
            )

            # Store values for statistic purpose
            if self.use_cost_value_function:
                self.logger.store(**{
                    'Values/V': v,
                    'Values/C': cv})
            else:
                self.logger.store(**{'Values/V': v})

            # Update observation
            o = next_o

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, cv, _ = self.ac(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v, cv = 0., 0.

                # Automatically compute GAE in buffer
                self.buf.finish_path(v, cv, penalty_param=float(penalty_param))

                # Only save EpRet / EpLen if trajectory finished
                if terminal:  
                    self.update_lagrange_multiplier(ep_costs)
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len, EpCosts=ep_costs)
                    o, ep_ret, ep_costs, ep_len = self.env.reset(), 0., 0., 0

import numpy as np
import torch
from safepo.algos.base import PolicyGradient
from safepo.algos.lagrangian_base import Lagrangian
from safepo.algos.policy_gradient import PG
import safepo.common.mpi_tools as mpi_tools


class RCPO(PG, Lagrangian):
    def __init__(
            self,
            algo='rcpo', 
            cost_limit=25., 
            lagrangian_multiplier_init=0.4,
            lambda_lr=5e-5, 
            lambda_optimizer='Adam',
            pi_lr=3e-4,
            vf_lr=1.5e-4,
            # lam_c=1,
            entropy_coef=0,
            # use_cost_value_function=True,
            lambda_schedule_lm=1-1e-9,
            **kwargs
        ):
        PG.__init__(
            self, 
            algo=algo,
            entropy_coef=entropy_coef,
            # use_cost_value_function=use_cost_value_function,
            # use_discount_cost_update_lag=True,
            pi_lr=pi_lr,
            vf_lr=vf_lr,
            # lam_c=lam_c,
            **kwargs
        )
        

        Lagrangian.__init__(
            self, 
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init, 
            lambda_lr=lambda_lr, 
            lambda_optimizer=lambda_optimizer
        )

        self.lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.lambda_optimizer,
                lr_lambda=lambda _: lambda_schedule_lm
            )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        loss_pi = - (ratio * data['adv']).mean()

        # ensure that lagrange multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += (ratio * penalty * data['target_c']).mean()
        # loss_pi -= self.entropy_coef * dist.entropy().mean()
        

        # Useful extra info
        approx_kl = .5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        PG.update(self)

        ep_costs = self.logger.get_stats('EpCosts')[0]
        # print(f"EpCosts = {self.logger.get_stats('EpCosts')}\n")
        # print(f"EpCosts[0] = {ep_costs}")
        self.update_lagrange_multiplier(ep_costs)

    def update_value_net(self, data: dict) -> None:
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        data['target_v'] = data['target_v'] - penalty * data['target_c']
        

        PG.update_value_net(self, data)
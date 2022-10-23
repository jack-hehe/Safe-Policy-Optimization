
import torch
from safepo.algos.policy_gradient import PG
from safepo.algos.trpo import TRPO
from safepo.common.buffer_with_cum_cost import Buffer_With_Cum_Cost


class CRPO(PG):
    def __init__(
        self, 
        algo='crpo', 
        eta=0.5, 
        cost_limit=25.,
        pi_lr=3e-4,
        vf_lr=1.5e-4, 
        # use_cost_value_function=True,
        **kwargs):
        super().__init__(
            algo=algo, 
            # use_cost_value_function=use_cost_value_function,
            # use_discount_cost_update_lag=True,
            pi_lr=pi_lr,
            vf_lr=vf_lr,
            **kwargs
            )
        self.buf = Buffer_With_Cum_Cost(self.buf)
        self.eta = eta
        self.cost_limit = cost_limit
        self.need_update_constrains = False

    def update(self):
        cv = self.logger.get_stats('EpCosts')[0]
        # _, _, cv, _ = self.ac.step(torch.as_tensor(self.buf.obs_buf[0], dtype=torch.float32))
        if cv <= self.eta + self.cost_limit:
            self.need_update_constrains = False
        else:
            self.need_update_constrains = True
        super().update()

    def compute_loss_pi(self, data: dict):
        '''
            computing pi/actor loss

            Returns:
                torch.Tensor
        '''
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        if self.need_update_constrains:
            loss_pi = (ratio * data['cum_cost']).mean()
        else:
            loss_pi = (- ratio * data['adv']).mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info


import torch
from safepo.algos.policy_gradient import PG
from safepo.algos.trpo import TRPO


class CRPO(PG):
    def __init__(self, algo='crpo', eta=0.5, cost_limit=25., **kwargs):
        super().__init__(
            algo=algo, 
            use_cost_value_function=True, 
            use_discount_cost_update_lag=True,
            **kwargs
            )
        self.eta = eta
        self.cost_limit = cost_limit
        self.need_update_constrains = False

    def update(self):
        _, _, cv, _ = self.ac.step(torch.as_tensor(self.buf.obs_buf[0], dtype=torch.float32))
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
            loss_pi = (ratio * data['cost_adv']).mean()
        else:
            loss_pi = -(ratio * data['adv']).mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

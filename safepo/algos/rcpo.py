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
            lagrangian_multiplier_init=0.001,
            lambda_lr=0.035, 
            lambda_optimizer='Adam',
            **kwargs
        ):
        PG.__init__(
            self, 
            algo=algo,
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
        loss_pi += (ratio *  penalty * data['target_c']).mean()
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


    def update_value_net(self, data: dict) -> None:
        # Divide whole local epoch data into mini_batches which is mbs size
        mbs = self.local_steps_per_epoch // self.num_mini_batches
        assert mbs >= 16, f'Batch size {mbs}<16'

        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_v = self.compute_loss_v(data['obs'], data['target_v'] - penalty * data['target_c'])
        self.loss_v_before = loss_v.item()

        indices = np.arange(self.local_steps_per_epoch)
        val_losses = []
        for _ in range(self.train_v_iterations):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)  
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data['obs'][mb_indices],
                    ret=data['target_v'][mb_indices] - penalty * data['target_c'][mb_indices])
                loss_v.backward()
                val_losses.append(loss_v.item())
                # Average grads across MPI processes
                mpi_tools.mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })
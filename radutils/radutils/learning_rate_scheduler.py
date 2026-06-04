import numpy as np
import torch


def clamp_lr(out_lr):
    return max(min(out_lr, 1e-1), 1e-7)


class LRScheduler:
    def __init__(self, params, optimizer, *args, **kwargs):
        self.scheduler = None
        self.optimizer = optimizer
        self._had_called_init = False
        self.init_lr = params["learning_rate"]
        self.start_epoch = None

    def step(self, epoch=None, cost=None, others=None):
        self.scheduler.step()

    def init_scheduler(self, epoch=None, cost=None, others=None):
        if self._had_called_init:
            return False
        self._had_called_init = True
        self.start_epoch = epoch
        return True

    def had_init(self):
        return self._had_called_init

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        return False

    @staticmethod
    def reset_learning_rate(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def state_dict(self):
        if self.scheduler is None:
            return {}
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if self.scheduler is None:
            return
        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()


class LRConstant(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

        self.lr_constant = params["learning_rate_constant"]

        class DummyScheduler:
            def step(self, epoch):
                pass

            def state_dict(self):
                return {}

        self.scheduler = DummyScheduler()

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        self.reset_learning_rate(self.optimizer, self.lr_constant)
        return True

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        assert epoch is not None
        return False


class LRLinearRampUp(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

        self.lr_rampup_start_lr = params["learning_rate_linear_rampup_start_lr"]
        self.lr_rampup_last_epoch = int(params["num_epochs"] * params["learning_rate_linear_rampup_ratio"])
        self.lr_rampup_last_epoch = max(1, self.lr_rampup_last_epoch)

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False

        initial_lr = self.init_lr
        if "initial_lr" in self.optimizer.param_groups[0]:
            initial_lr = self.optimizer.param_groups[0]["initial_lr"]
        else:
            initial_lr = self.lr_rampup_start_lr
            self.reset_learning_rate(self.optimizer, self.lr_rampup_start_lr)

        def lambda_lr(epoch_in):
            x = epoch_in / self.lr_rampup_last_epoch
            out_lr = x * (self.init_lr - self.lr_rampup_start_lr) + self.lr_rampup_start_lr
            return clamp_lr(out_lr) / initial_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        return True

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        assert epoch is not None
        if epoch >= self.start_epoch + self.lr_rampup_last_epoch:
            return True
        return False


class LRLinearDecay(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

        self.num_epoch = int(params["num_epochs"] * params["learning_rate_linear_decay_epoch_multiplier"])
        self.num_epoch = max(1, self.num_epoch)
        self.start_epoch = None
        self.last_epoch = None

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        assert epoch is not None
        if epoch >= self.last_epoch:
            return True
        return False

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        self.start_epoch = epoch
        self.last_epoch = epoch + self.num_epoch - 1
        self.reset_learning_rate(self.optimizer, self.init_lr)

        initial_lr = self.init_lr
        if "initial_lr" in self.optimizer.param_groups[0]:
            initial_lr = self.optimizer.param_groups[0]["initial_lr"]

        def lambda_lr(epoch_in):
            out_lr = (self.num_epoch - epoch_in) / self.num_epoch * self.init_lr
            out_lr = clamp_lr(out_lr)
            return out_lr / initial_lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

        return True


class LRLinearCycling(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

        self.num_epoch = max(1, int(params["num_epochs"] * params["learning_rate_linear_cycling_end_multiplier"]))
        self.max_num_cycle = params["learning_rate_linear_cycling_num_cycles"]
        self.cycling_period_epoch = self.num_epoch / self.max_num_cycle
        self.warm_up_ratio = params["learning_rate_linear_cycling_warmup_ratio"]
        self.max_lr = params["learning_rate_linear_cycling_max_lr"]
        self.min_lr = params["learning_rate_linear_cycling_min_lr"]
        self.last_peak_lr = params["learning_rate_linear_cycling_last_peak_lr"]
        self.last_epoch = None

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        assert epoch is not None
        if epoch >= self.last_epoch:
            return True
        return False

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        self.last_epoch = epoch + self.num_epoch + 1
        self.reset_learning_rate(self.optimizer, self.min_lr)

        initial_lr = self.init_lr
        if "initial_lr" in self.optimizer.param_groups[0]:
            initial_lr = self.optimizer.param_groups[0]["initial_lr"]
        else:
            initial_lr = self.min_lr
            self.reset_learning_rate(self.optimizer, self.min_lr)

        def lambda_lr(epoch_in: int):
            cur_epoch = epoch_in
            cycle_idx = cur_epoch // self.cycling_period_epoch
            # Max lr linear decrease with each cycle
            cycle_max_lr = (1 - cycle_idx / (self.max_num_cycle - 1)) * (
                self.max_lr - self.last_peak_lr
            ) + self.last_peak_lr
            x = (cur_epoch % self.cycling_period_epoch) / self.cycling_period_epoch
            if x < self.warm_up_ratio:
                # warm up
                out_lr = self.min_lr + (cycle_max_lr - self.min_lr) / self.warm_up_ratio * x
            else:
                # decrease
                k = (cycle_max_lr - self.min_lr) / (1 - self.warm_up_ratio)
                out_lr = self.min_lr + (1 - x) * k
            factor = clamp_lr(out_lr) / initial_lr

            return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

        return True


class LRLinearCyclingAdaptive(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

        self.num_epoch = int(params["num_epochs"] * params["learning_rate_linear_cycling_end_multiplier"])
        self.cycling_period_epoch = int(params["num_epochs"] * params["learning_rate_linear_cycling_period_multiplier"])
        self.max_num_cycle = round(1 / params["learning_rate_linear_cycling_period_multiplier"])
        self.warm_up_ratio = params["learning_rate_linear_cycling_warmup_ratio"]
        self.min_lr = params["learning_rate_linear_cycling_min_lr"]
        self.min_lr = params["learning_rate_linear_cycling_min_lr"]
        self.lr_increment = (self.init_lr - self.min_lr) / (self.warm_up_ratio * self.cycling_period_epoch)
        self.lr_decrement = (self.init_lr - self.min_lr) / ((1 - self.warm_up_ratio) * self.cycling_period_epoch)
        self.num_epoch = max(1, self.num_epoch)
        self.last_epoch = None

        self.num_cost_points = params["learning_rate_linear_cycling_adaptive_avg_steps"]
        self.max_num_cost_points = self.num_cost_points * 2
        self.cost_list = []
        self.period_start = None
        self.last_lr = self.min_lr
        self.in_warm_up = True

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        assert epoch is not None
        if epoch >= self.last_epoch:
            return True
        if len(self.cost_list) > self.max_num_cost_points:
            self.cost_list.pop(0)
        self.cost_list.append(cost)
        return False

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        self.last_epoch = epoch + self.num_epoch + 1
        self.reset_learning_rate(self.optimizer, self.init_lr)
        self.period_start = epoch

        def lr(epoch: int):
            this = self
            if len(this.cost_list) >= this.max_num_cost_points:
                cost_new = np.mean(this.cost_list[-this.num_cost_points :])
                cost_old = np.mean(this.cost_list)
                if this.in_warm_up and cost_new > cost_old:
                    this.in_warm_up = False

            if this.in_warm_up:
                this.last_lr += this.lr_increment
            else:
                this.last_lr -= this.lr_decrement
                if this.last_lr < this.min_lr:
                    this.in_warm_up = True

            factor = this.last_lr / self.init_lr

            return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr)

        return True


class LRReduceOnPlateau(LRScheduler):
    def __init__(self, params, optimizer, patience, factor, stop_lr, *args, **kwargs):
        super().__init__(params, optimizer)

        self.stop_lr = stop_lr
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
        )

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        self.reset_learning_rate(self.optimizer, self.init_lr)
        return True

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        if self.stop_lr is not None and self.get_lr() <= self.stop_lr:
            return True
        return False

    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]["lr"]

    def step(self, epoch=None, cost=None, others=None):
        self.scheduler.step(cost)


class LRConstant(LRScheduler):
    def __init__(self, params, optimizer, *args, **kwargs):
        super().__init__(params, optimizer)

    def is_scheduler_done(self, epoch=None, cost=None, others=None):
        return False

    def init_scheduler(self, epoch=None, cost=None, others=None):
        run_init = super().init_scheduler(epoch)
        if not run_init:
            return False
        target_lr = self.init_lr
        if "lr" in self.optimizer.param_groups[0]:
            target_lr = self.optimizer.param_groups[0]["lr"]

        initial_lr = self.init_lr
        if "initial_lr" in self.optimizer.param_groups[0]:
            initial_lr = self.optimizer.param_groups[0]["initial_lr"]
        else:
            initial_lr = self.init_lr

        def lambda_lr(epoch_in: int):
            factor = target_lr / initial_lr
            return factor

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

        return True


class LRSchedulerSequence:
    def __init__(self, params):
        self.schedulers = []

    def init(self, epoch):
        assert len(self.schedulers) > 0, "Must provide at least 1 LR scheduler"
        scheduler = self.schedulers[0]
        if not scheduler.had_init():
            scheduler.init_scheduler(epoch, 1e50, {})

    def add_scheduler(self, scheduler: LRScheduler):
        self.schedulers.append(scheduler)

    def step(self, epoch=None, cost=None, others=None):
        if len(self.schedulers) == 0:
            return
        scheduler = self.schedulers[0]
        if not scheduler.had_init():
            scheduler.init_scheduler(epoch, cost, others)
        scheduler.step(epoch, cost, others)

        if scheduler.is_scheduler_done(epoch, cost, others):
            assert len(self.schedulers) > 1, "All schedulers finished, no more learning rate scheduler."
            self.schedulers.pop(0)
            self.schedulers[0].init_scheduler(epoch, cost, others)

    def state_dict(self):
        return [d.state_dict() for d in self.schedulers]

    def load_state_dict(self, state_dict):
        [s.load_state_dict(state_dict[i]) for i, s in enumerate(self.schedulers)]

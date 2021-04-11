import torch

class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                optimizer,
                warmup_steps=5000,
                init_lr=1e-3,
                min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        
        self.gstep = 0

        super(LRScheduler, self).__init__(optimizer)

    def state_dict(self):
        return {k:v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        self.gstep += 1
        if self.gstep < self.warmup_steps:
            lr = self.gstep * self.init_lr / self.warmup_steps
        else:
            lr = self.init_lr * (self.warmup_steps / self.gstep) ** 0.5
        lr = max(lr, self.min_lr)
        return [lr for _ in self.optimizer.param_groups]
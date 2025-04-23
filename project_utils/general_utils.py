import os
from datetime import datetime

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args):
    args.cuda = torch.cuda.is_available()

    root_dir = args.log_dir

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # get corrent time formatting year_month_day_hour_minute
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")

    log_dir = os.path.join(root_dir, now)
    while os.path.exists(log_dir):
        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        log_dir = os.path.join(root_dir, now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, "model.pt")

    print(f"Experiment saved to: {args.log_dir}")
    print(args)

    return args

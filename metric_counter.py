import numpy as np
from tensorboardX import SummaryWriter
import logging

class MetricCounter():
    

    def __init__(self, exp_name, print_every):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.loss = []
        self.print_every = print_every
        self.best_metric = 0

    def clear(self):
        self.loss = []

    def add_losses(self, loss):
        self.loss.append(loss.item())

    def loss_message(self):
        mean_loss = np.mean(self.loss[-MetricCounter.REPORT_EACH:])
        return f'loss={round(mean_loss, 3)};'

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        self.writer.add_scalar(f'{scalar_prefix}_loss', np.mean(self.loss), epoch_num)


    def update_best_model(self):
        cur_metric = np.mean(self.loss)
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
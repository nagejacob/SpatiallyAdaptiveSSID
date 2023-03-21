from model.base import BaseModel
import os
import torch
import torch.nn as nn

class BNNModel(BaseModel):
    def __init__(self, opt):
        super(BNNModel, self).__init__(opt)
        self.criteron = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.networks['BNN'].parameters(), lr=opt['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, opt['num_iters'])

    def train_step(self, data):
        input = data['L']
        self.networks['BNN'].train()
        output = self.networks['BNN'](input)

        self.loss = self.criteron(output, input)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iter += 1

    def validation_step(self, data):
        input = data['L']
        self.networks['BNN'].eval()
        with torch.no_grad():
            output = self.networks['BNN'](input)

        return output

    def save_model(self):
        save_dict = {'iter': self.iter,
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict(),
                     'BNN': self.networks['BNN'].state_dict()}
        torch.save(save_dict, os.path.join(self.opt['log_dir'], 'model_iter_%08d.pth' % self.iter))

    def load_model(self, path):
        load_dict = torch.load(path)
        self.iter = load_dict['iter']
        self.optimizer.load_state_dict(load_dict['optimizer'])
        self.scheduler.load_state_dict(load_dict['scheduler'])
        self.networks['BNN'].load_state_dict(load_dict['BNN'])



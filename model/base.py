from abc import abstractmethod
import os
import torch
from torch.nn.parallel import DataParallel
from utils.build import build
from utils.io import log

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.iter = 0 if 'iter' not in opt else opt['iter']
        self.networks = {}
        for network_opt in opt['networks']:
            Net = getattr(__import__('network'), network_opt['type'])
            net = build(Net, network_opt['args'])
            if 'path' in network_opt.keys():
                self.load_net(net, network_opt['path'])
            self.networks[network_opt['name']] = net

    @abstractmethod
    def train_step(self, data):
        pass

    @abstractmethod
    def validation_step(self, data):
        pass

    def data_parallel(self):
        for name in self.networks.keys():
            net = self.networks[name]
            net = net.cuda()
            net = DataParallel(net)
            self.networks[name] = net

    def save_net(self):
        for name, net in self.networks.items():
            if isinstance(net, DataParallel):
                net = net.module
            torch.save(net.state_dict(), os.path.join(self.opt['log_dir'], '%s_iter_%08d.pth' % (name, self.iter)))

    def load_net(self, net, path):
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    def log(self):
        log(self.opt['log_file'], 'iter: %d, loss: %f\n' % (self.iter, self.loss.item()))
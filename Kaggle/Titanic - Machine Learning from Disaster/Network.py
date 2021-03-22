import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.fc_layer = self._init_layer()

    def forward(self, x):
        out = self.fc_layer(x)
        return out

    def _init_layer(self):
        layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 4, self.hid_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hid_dim * 4, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim)
        )
        return layers


if __name__ == '__main__':
    # check output
    def dim_check(net):
        inputs = torch.randn(4, 8)
        outputs = net.forward(inputs)
        print(outputs.shape)

    model = Net(in_dim=8, hid_dim=30, out_dim=2)
    dim_check(model)
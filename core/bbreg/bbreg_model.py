import torch
from torch import nn
import torch.nn.functional as F

from tracking.options import opts


class BBregNet(nn.Module):
    def __init__(self, in_dim=opts["in_dim"]):
        super(BBregNet, self).__init__()

        self.fc1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        # self.fc3 = nn.Linear(2048, 2048)
        # self.bn3 = nn.BatchNorm1d(2048)

        self.boxreg = nn.Linear(1024, 4)
        self.boxstd = nn.Linear(1024, 4)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.01)
        # nn.init.kaiming_normal_(self.fc3.weight)
        # nn.init.constant_(self.fc3.bias, 0.01)
        nn.init.kaiming_normal_(self.boxreg.weight)
        nn.init.constant_(self.boxreg.bias, 0.01)
        nn.init.normal_(self.boxstd.weight, std=0.0001, mean=0)
        nn.init.constant_(self.boxstd.bias, 0.01)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn3(self.fc3(x)))
        boxreg = self.boxreg(x)
        boxstd = self.boxstd(x)
        return boxreg, torch.abs(boxstd)


class BBregNet_Raw(nn.Module):
    def __init__(self, in_dim=opts["in_dim"]):
        super(BBregNet_Raw, self).__init__()

        self.fc1 = nn.Linear(in_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.boxreg = nn.Linear(2048, 4)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.1)
        nn.init.kaiming_normal_(self.boxreg.weight)
        nn.init.constant_(self.boxreg.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        boxreg = self.boxreg(x)
        return boxreg



if __name__ == '__main__':
    pass

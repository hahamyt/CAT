import torch
import torch.nn as nn

class FC_HF(nn.Module):
    def __init__(self, in_dim=128, l_dim=256, out_dim=2):
        super(FC_HF, self).__init__()

        # dimension reduction
        self.dr1 = nn.Linear(in_dim, l_dim)
        self.bbn1 = nn.BatchNorm1d(l_dim)
        self.dr1_relu = nn.ReLU(inplace=True)
        self.drop0_1 = nn.Dropout(0.5)

        self.dr2 = nn.Linear(l_dim, l_dim)
        self.bbn2 = nn.BatchNorm1d(l_dim)
        self.dr2_relu = nn.ReLU(inplace=True)
        self.drop0_2 = nn.Dropout(0.5)

        self.dr3 = nn.Linear(l_dim, in_dim)
        # self.mask = nn.Sigmoid()

        self.fc1 = nn.Linear(in_dim, l_dim)
        self.bn1 = nn.BatchNorm1d(l_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(l_dim, l_dim)
        self.bn2 = nn.BatchNorm1d(l_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(l_dim, out_dim)

        # network init
        nn.init.normal_(self.dr1.weight, 0, 0.01)
        nn.init.constant_(self.dr1.bias, 0)
        nn.init.normal_(self.dr2.weight, 0, 0.01)
        nn.init.constant_(self.dr2.bias, 0)
        nn.init.normal_(self.dr3.weight, 0, 0.01)
        nn.init.constant_(self.dr3.bias, 0)

        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, 0, 0.01)
        nn.init.constant_(self.fc3.bias, 0)

        self.m = None
        self.steps = 0

    def forward(self, x):
        # calculate the input codes' mask to reduction dimension
        mask = self.drop0_1(self.dr1_relu(self.dr1(x)))
        mask = self.drop0_2(self.dr2_relu(self.dr2(mask)))
        mask = self.dr3(mask)

        self.m = mask
        # mask = torch.abs(mask)
        # mask input code
        new_codes = mask * x
        # calculate score
        score = self.drop1(self.relu1(self.bn1(self.fc1(new_codes))))
        score = self.drop2(self.relu2(self.bn2(self.fc2(score))))
        score = self.fc3(score)
        return torch.sigmoid(score), new_codes  # torch.sigmoid(score, dim=1)



import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(12, 256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                )

        self.p_head = nn.Sequential(
                nn.Linear(128, 18),
                nn.Softmax(dim=-1),
                )

        self.v_head = nn.Sequential(
                nn.Linear(128, 1)
                )

    def forward(self, x):
        x = self.net(x)

        return self.p_head(x), self.v_head(x)


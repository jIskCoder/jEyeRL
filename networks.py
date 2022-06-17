import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from torch.nn import Linear, ReLU, Sequential

class LinActv(nn.Module):
    def __init__(self): nn.Module.__init__(self)

    def forward(self, x): return x

    
class ActionMapper(nn.Module):
    def __init__(this, inpshp, outshp):
        nn.Module.__init__(this)
        
        this.k = nn.Linear(inpshp, outshp)
        this.x0 = nn.Linear(inpshp, outshp)
        
    def forward(this, x):
        kmin = .25
        x0 = this.x0(x)
        k = this.k(x).sigmoid() + kmin
        out = 0 + 1 * (k*x-k*x0).sigmoid()
            
        return out


class SimpleMLP(nn.Module):
    def __init__(self, inpshp, hidshp, outshp,
                 _fn=LinActv(), fn_=LinActv()):
        nn.Module.__init__(self)

        layers = [_fn,
                  Linear(inpshp, hidshp),
                  ReLU(),
                  Linear(hidshp, outshp),
                  fn_]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
class Observer(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.obs = SimpleMLP(27, 64, 64, fn_=ReLU())

    def forward(self, x):
        return self.obs(x)
    
class EyeController(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        #pdb.set_trace()
        self.horizontalNN = SimpleMLP(64, 256, 2)
        self.hcoordinationNN = SimpleMLP(64, 256, 2)
        self.verticalNN = SimpleMLP(64, 256, 4)

        self.LHorzActMap = ActionMapper(2, outshp=2)
        self.RHorzActMap = ActionMapper(2, outshp=2)
        self.VertActmap = ActionMapper(4, outshp=4)

    def forward(self, x):
        # Horizontal Network
        hact = self.horizontalNN(x)
            
        LLR, LMR = hact[:, 0].unsqueeze(1), hact[:, 1].unsqueeze(1)

        # Left/Right Coordination Factors
        c12 = self.hcoordinationNN(x)
        c1, c2 = c12[:, 0].unsqueeze(1), c12[:, 1].unsqueeze(1)

        RMR = c1 * LLR
        RLR = c2 * LMR

        LeftHorz = self.LHorzActMap(torch.cat([LLR, LMR], dim=1))
        RightHorz = self.RHorzActMap(torch.cat([RLR, RMR], dim=1))

        # Vertical Network
        vact = self.verticalNN(x)

        vact = self.VertActmap(vact)
        vactL, vactR = [vact]*2

        # Stitching Actuations
        action_list = [RightHorz, vact, LeftHorz, vact]
        for a in action_list: a.retain_grad()
        actions = torch.cat(action_list, dim=1)

        return actions
        

class Actor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.aobs = Observer()
        self.anet = EyeController()

    def forward(self, x):
        x = self.aobs(x)
        x = self.anet(x)

        return x


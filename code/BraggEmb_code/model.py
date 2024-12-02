import torch, copy

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class MLPhead(torch.nn.Module):
    def __init__(self, ic, hdim, oc):
        super(MLPhead, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(ic, hdim),
            torch.nn.BatchNorm1d(hdim),
            torch.nn.ReLU(inplace=True),
            torch. nn.Linear(hdim, oc)
        )

    def forward(self, x):
        return self.net(x)

class BraggPeakBYOL(torch.nn.Module):
    def __init__(self, psz, hdim, proj_dim, enc_chs=(32, 32, 32)):
        super().__init__()
        
        enc_in_chs  = (1, ) + enc_chs[:-1]
        enc_ops = []
        for ic, oc, in zip(enc_in_chs, enc_chs):
            enc_ops += [
                        torch.nn.Conv2d(in_channels=ic, out_channels=ic, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(num_features=ic),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, stride=2, padding=0),
                        torch.nn.ReLU(inplace=True),
            ]
            psz = psz // 2

        self.encoder   = torch.nn.Sequential(*enc_ops[:-1])

        self.projector = MLPhead(psz * psz * enc_chs[-1], hdim, proj_dim)

    def forward(self, x, rety=True):
        rep_vec = self.encoder(x)
        rep_vec = rep_vec.flatten(start_dim=1)

        if rety:
            return rep_vec
        else:
            proj = self.projector(rep_vec)
            return rep_vec, proj

class targetNN():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.old  = None
        
    def reset(self):
        self.old = None
        
    def update(self, new):
        if self.old is None:
            self.old = copy.deepcopy(new)
        else:
            for old_para, new_para in zip(self.old.parameters(), new.parameters()):
                old_para.data = old_para.data * self.beta + (1 - self.beta) * new_para.data
        
        return None
    
    def predict(self, x):
        with torch.no_grad():
            _, proj = self.old.forward(x, rety=False)

        return proj

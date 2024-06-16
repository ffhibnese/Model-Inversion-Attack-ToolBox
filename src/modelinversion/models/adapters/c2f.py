from torch.nn.modules import Module
from .base import *


# @register_model('c2f_mlp2')
class C2fOutputMapping(BaseAdapter):

    # @ModelMixin.register_to_config_init
    def __init__(self, input_dim, map: nn.Module, trunc: int = 1):
        super(C2fOutputMapping, self).__init__()

        self.trunc = trunc
        self.input_dim = input_dim
        self.map = map
        # 10575

    def forward(self, x):
        # input_dim = x.shape[-1]
        topk, index = torch.topk(x, self.trunc)
        topk = torch.clamp(torch.log(topk), min=-1000) + 50.0
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros_like(x).scatter_(1, index, topk)
        x = x.view(-1, self.input_dim)
        x = F.normalize(x, 2, dim=1)
        x = self.map(x)
        # x = F.normalize(x, 2, dim=1)
        return x


@register_adapter('c2f_mlp3')
class C2fThreeLayerMlpOutputMapping(C2fOutputMapping):

    @ModelMixin.register_to_config_init
    def __init__(self, input_dim, hidden_dim, output_dim, trunc: int = 1):
        map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            # nn.Linear(4096, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # nn.BatchNorm1d(128, eps=0.0000001, momentum=0.1, affine=True),
        )

        super().__init__(input_dim, map, trunc)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .seghead import SegFormerHead
# from . import MixT_block
# from .MixT_block_select_UMD import mit_b0, mit_b1, mit_b2, mit_b4
from semseg.models.segformer.seghead import SegFormerHead
from semseg.models.segformer.MixT import mit_b0, mit_b1, mit_b2, mit_b4


class Seg(nn.Module):
    def __init__(self, backbone, num_classes=19, embedding_dim=768, pretrained=None, modals=None):
        super().__init__()
        self.modals = modals
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]

        if backbone == 'mit_b0':
            self.encoder = mit_b0()
            if pretrained:
                state_dict = torch.load('/home/dell/Datasets/Deliver/segformers/mit_b0.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                state_dict.pop('patch_embed1.proj.weight')
                self.encoder.load_state_dict(state_dict, strict=False)
        if backbone == 'mit_b1':
            self.encoder = mit_b1()
            if pretrained:
                state_dict = torch.load('/home/jinjing/zhengxu/DELIVER/semseg/models/segformer/mit_b1.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict, )
        if backbone == 'mit_b4':
            self.encoder = mit_b4()
            if pretrained:
                state_dict = torch.load('/home/jinjing/zhengxu/DELIVER/semseg/models/segformer/mit_b4.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict, )
        if backbone == 'mit_b2':
            self.encoder = mit_b2()
            if pretrained:
                state_dict = torch.load('/home/dell/Datasets/Deliver/segformers/mit_b2.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict, strict=False)
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        self.backbone = backbone
        self.embed_dim = 768
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = SegFormerHead(self.in_channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, self.embed_dim, num_classes)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1,
                                    bias=False)

    def _forward_cam(self, x):

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):
        x = torch.stack(x).float()
        m, b, c, h, w = x.shape
        x = x.reshape(b,m * c,h,w)
        # _, _, _, height, width = x.shape
        _, _, height, width = x.shape

        _x = self.encoder(x)

        feature = self.decoder(_x)

        pred = F.interpolate(feature, size=(height, width), mode='bilinear', align_corners=False)

        pred = pred.reshape(b, self.num_classes, h, w)
        # pred = torch.mean(pred, dim=0)

        return pred


if __name__ == "__main__":

    from fvcore.nn import FlopCountAnalysis

    model = Seg("mit_b0", num_classes=19, pretrained=True)
    input = [torch.zeros(4, 3, 1024, 1024), torch.ones(4, 3, 1024, 1024), torch.ones(4, 3, 1024, 1024),
             torch.ones(4, 3, 1024, 1024)]
    output = model(input)
    # print(output.size())

    # Calculate the number of parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:,}")

    # Calculate GFLOPs
    example_inputs = (input,)
    flops = FlopCountAnalysis(model, example_inputs)
    gflops = flops.total() / 1e9  # Convert to GFLOPs
    print(f"GFLOPs: {gflops:.2f}")

    # Run the model and print output shapes
    outs = model(input)
    for i, y in enumerate(outs):
        print(f"Output shape at stage {i + 1}: {y.shape}")

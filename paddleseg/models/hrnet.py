import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class ProjectionHead(nn.Layer):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2D(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2D(dim_in, dim_in, kernel_size=1),
                layers.SyncBatchNorm(dim_in),
                nn.ReLU(),
                nn.Conv2D(dim_in, dim_in, kernel_size=1),
            )
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, axis=1)


@manager.MODELS.add_component
class HRNetW48Contrast(nn.Layer):
    def __init__(self, in_channels, backbone, num_classes, drop_prob, proj_dim, align_corners=False):
        super(HRNetW48Contrast, self).__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.align_corners = align_corners

        self.cls_head = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            layers.SyncBatchNorm(in_channels),
            nn.ReLU(),
            nn.Dropout2D(drop_prob),
            nn.Conv2D(in_channels, num_classes, kernel_size=1, stride=1, bias_attr=False),
        )

        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)


    def forward(self, x):
        feats = self.backbone(x)[0]
        out = self.cls_head(feats)
        if self.training:
            emb = self.proj_head(feats)
            return [F.interpolate(out, x.shape[2:], mode='bilinear', align_corners=self.align_corners), {'seg': out, 'embed': emb}]
        else:
            out = F.interpolate(out, x.shape[2:], mode='bilinear', align_corners=self.align_corners)
            return [out]

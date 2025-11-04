# import torch
# from torch import nn as nn

# from basicsr.archs.arch_utils import ResidualBlockNoBN, Upsample, make_layer
# from basicsr.utils.registry import ARCH_REGISTRY


# @ARCH_REGISTRY.register()
# class EDSR(nn.Module):
#     """
#     EDSR network that can work as both teacher and student.
    
#     - As TEACHER: Uses original architecture (upscale + conv_last)
#     - As STUDENT: Can optionally use reconstruction_head for CustomKD training
#     - After training: Always uses original architecture for inference
#     """

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=64,
#                  num_block=16,
#                  upscale=4,
#                  res_scale=1,
#                  img_range=255.,
#                  rgb_mean=(0.4488, 0.4371, 0.4040),
#                  use_reconstruction_head=False):
#         super(EDSR, self).__init__()

#         self.img_range = img_range
#         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#         self.use_reconstruction_head = use_reconstruction_head

#         # --- Feature Extractor Components (same for both) ---
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(
#             ResidualBlockNoBN,
#             num_block,
#             num_feat=num_feat,
#             res_scale=res_scale,
#             pytorch_init=True)
#         self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
#         # --- Reconstruction Components ---
#         if use_reconstruction_head:
#             # For student training: grouped reconstruction head
#             self.reconstruction_head = nn.Sequential(
#                 Upsample(upscale, num_feat),
#                 nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#             )
#             # Keep original layers for weight loading compatibility
#             self.upsample = Upsample(upscale, num_feat)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#         else:
#             # For teacher/inference: original architecture
#             self.upsample = Upsample(upscale, num_feat)
#             self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#     def get_features(self, x):
#         """
#         Extracts deep features from the main body of the network.
#         This is the output of the "feature extractor" part.
#         """
#         # Normalize the input image
#         self.mean = self.mean.type_as(x)
#         x_norm = (x - self.mean) * self.img_range

#         # Pass through the feature extraction part
#         feat_shallow = self.conv_first(x_norm)
#         feat_deep = self.conv_after_body(self.body(feat_shallow))
        
#         # The final features are the sum from the residual connection
#         features = feat_deep + feat_shallow
#         return features

#     def forward(self, x):
#         """
#         Forward pass that works for both teacher and student.
#         """
#         # Normalize the input image
#         self.mean = self.mean.type_as(x)
#         x_norm = (x - self.mean) * self.img_range

#         # Feature extraction
#         feat_shallow = self.conv_first(x_norm)
#         feat_deep = self.conv_after_body(self.body(feat_shallow))
#         feat = feat_deep + feat_shallow

#         # Reconstruction
#         if self.use_reconstruction_head:
#             # Use reconstruction head (for student training)
#             out = self.reconstruction_head(feat)
#         else:
#             # Use original architecture (for teacher/inference)
#             out = self.conv_last(self.upsample(feat))

#         # Denormalize the output
#         out = out / self.img_range + self.mean

#         return out
    
#     def get_reconstruction_head(self):
#         """
#         Get the reconstruction head for CustomKD training.
#         This allows external access to the reconstruction head during training.
#         """
#         if not self.use_reconstruction_head:
#             raise RuntimeError("Reconstruction head not available. Set use_reconstruction_head=True during initialization.")
#         return self.reconstruction_head
    # edsr_arch.py (Refactored Version)

# import torch
# from torch import nn as nn

# from basicsr.archs.arch_utils import ResidualBlockNoBN, Upsample, make_layer
# from basicsr.utils.registry import ARCH_REGISTRY


# @ARCH_REGISTRY.register()
# class EDSR(nn.Module):
#     """
#     EDSR network refactored for efficiency and clarity.
#     """

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=64,
#                  num_block=16,
#                  upscale=4,
#                  res_scale=1,
#                  img_range=1.,
#                  rgb_mean=(0.4488, 0.4371, 0.4040),
#                  use_reconstruction_head=False):
#         super(EDSR, self).__init__()

#         self.img_range = img_range
#         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
#         self.use_reconstruction_head = use_reconstruction_head

#         # --- Feature Extractor Components ---
#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(
#             ResidualBlockNoBN,
#             num_block,
#             num_feat=num_feat,
#             res_scale=res_scale,
#             pytorch_init=True)
#         self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
#         # --- Reconstruction Components ---
#         # The reconstruction head for student training in CustomKD
#         self.reconstruction_head = nn.Sequential(
#             Upsample(upscale, num_feat),
#             nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
#         )
        
#         # The original layers for inference or standard training
#         self.upsample = Upsample(upscale, num_feat)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#     def _feature_extractor(self, x):
#         """Internal method to extract features, avoiding redundant code."""
#         feat_shallow = self.conv_first(x)
#         feat_deep = self.conv_after_body(self.body(feat_shallow))
#         return feat_deep + feat_shallow

#     def get_features(self, x):
#         """
#         Public method to get features for distillation.
#         Handles normalization.
#         """
#         self.mean = self.mean.type_as(x)
#         x_norm = (x - self.mean) * self.img_range
#         return self._feature_extractor(x_norm)

#     def forward(self, x):
#         """
#         Forward pass that efficiently reuses the feature extractor.
#         """
#         # Extract features (this also handles normalization)
#         features = self.get_features(x)

#         # Reconstruct the image
#         if self.training and self.use_reconstruction_head:
#             # During training as a student, use the dedicated head
#             out = self.reconstruction_head(features)
#         else:
#             # During inference or standard training, use the original path
#             out = self.conv_last(self.upsample(features))

#         # Denormalize the output
#         self.mean = self.mean.type_as(out)
#         out = out / self.img_range + self.mean

#         return out
    
#     def get_reconstruction_head(self):
#         """Get the reconstruction head for CustomKD training."""
#         return self.reconstruction_head


import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
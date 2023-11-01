import torch
import argparse
import sys
from timm.models.layers import to_2tuple
from torch import nn

sys.path.append("../src")
from audiomae import models_vit

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
args = parser.parse_args()


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride)  # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


model = models_vit.__dict__["vit_base_patch16"](
    num_classes=527,
    drop_path_rate=0.1,
    global_pool=True,
    mask_2d=True,
    use_custom_patch=False,
)

target_length = {
    'audioset': 1024,
    'k400': 1024,
    'esc50': 512,
    'speechcommands': 128
}
img_size = (target_length["audioset"], 128)  # 1024, 128
in_chans = 1
emb_dim = 768

model.patch_embed = PatchEmbed_new(img_size=img_size,
                                   patch_size=(16, 16),
                                   in_chans=1,
                                   embed_dim=emb_dim,
                                   stride=16)  # no overlap. stride=img_size=16
num_patches = model.patch_embed.num_patches
#num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim),
                               requires_grad=False)  # fixed sin-cos embedding

checkpoint = torch.load(args.ckpt_path, map_location='cpu')
print("Load pre-trained checkpoint from: %s" % args.ckpt_path)
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()

# if not args.eval:
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[
#                 k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]
# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

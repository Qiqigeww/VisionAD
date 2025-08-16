# import timm, torch
# model = timm.create_model('vit_small_patch16_224', features_only=True, out_indices=[2, 3, 4, 5, 6, 7, 8, 9], pretrained=True, pretrained_cfg_overlay=dict(file="Dinomaly/backbones/weights/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"))
# x = torch.randn([8, 3, 224, 224])
# features = model(x)

# for a in features:
#     print(a.shape)
# from urllib.request import urlopen
# from PIL import Image
# import timm

# img = Image.open("image.png")

# model = timm.create_model(
#     'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k',
#     pretrained=True,
#     features_only=True,
#     pretrained_cfg_overlay=dict(file="Dinomaly/backbones/weights/pytorch_model.bin"),
#     # custom_load=False
# )
# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# for o in output:
#     # print shape of each feature map in output
#     # e.g.:
#     #  torch.Size([1, 96, 96, 128])
#     #  torch.Size([1, 48, 48, 256])
#     #  torch.Size([1, 24, 24, 512])
#     #  torch.Size([1, 12, 12, 768])

#     print(o.shape)
# from urllib.request import urlopen
# from PIL import Image
# import timm

# img = Image.open("image.png")

# model = timm.create_model(
#     'vit_base_patch16_rope_reg1_gap_256.sbb_in1k',
#     pretrained=True,
#     features_only=True,
#     pretrained_cfg_overlay=dict(file="Dinomaly/backbones/weights/pytorch_model.bin"),
# )

# model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# for o in output:
#     # print shape of each feature map in output
#     # e.g.:
#     #  torch.Size([1, 768, 16, 16])
#     #  torch.Size([1, 768, 16, 16])
#     #  torch.Size([1, 768, 16, 16])

#     print(o.shape)


from PIL import Image
from ml_aim.aim_v2.aim.v2.utils import load_pretrained
from ml_aim.aim_v1.aim.v1.torch.data import val_transforms

img = Image.open("image.png")
model = load_pretrained('aimv2-large-patch14-224', backend="torch")
transform = val_transforms(img_size=224)

inp = transform(img).unsqueeze(0)
features = model(inp)
print(len(features))

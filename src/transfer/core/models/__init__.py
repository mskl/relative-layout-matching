from tensorflow.keras.models import Model

from core.models.vgg16 import vgg16
from core.models.resnet import resnet, resnet_unet_101, resnet_101_flat, resnet_50_flat, \
	resnet_unet_101_bos, resnet_unet_50, resnet_unet_50_bos
from core.models.unet import unet


def backbone_factory(
	backbone: str, embdim: int = 256, use_bos: bool = False, transfer: bool = True
) -> Model:
	if use_bos:
		if backbone in ("resnet_unet", "resnet_unet_101"):
			return resnet_unet_101_bos(embdim, transfer)
		elif backbone == "resnet_unet_50":
			return resnet_unet_50_bos(embdim, transfer)
	else:
		if backbone == "vgg16":
			return vgg16(transfer)
		elif backbone == "resnet":
			return resnet(embdim)
		elif backbone in ("resnet_unet", "resnet_unet_101"):
			return resnet_unet_101(embdim, transfer)
		elif backbone == "resnet_unet_50":
			return resnet_unet_50(embdim, transfer)
		elif backbone == "unet":
			return unet(embdim)
		elif backbone == "resnet_flat_101":
			return resnet_101_flat(embdim, transfer)
		elif backbone == "resnet_flat_50":
			return resnet_50_flat(embdim, transfer)
	raise ValueError(f"Uknown backbone kind {backbone}")

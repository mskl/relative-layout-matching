import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, Activation, BatchNormalization


def vgg16(transfer: bool = True) -> Model:
	"""Take pretrained VGG16 and cut it in half."""
	vgg = tf.keras.applications.vgg16.VGG16(
		include_top=False,
		weights='imagenet' if transfer else None,
		classes=1000
	)

	vgg_layermap = {l.name: l for l in vgg.layers}
	return Model(vgg.input, vgg_layermap["block3_pool"].output)


def vgg_unet(input_shape=None, embdim: int = 256) -> Model:
	vgg = tf.keras.applications.vgg16.VGG16(
		include_top=False,
		weights='imagenet',
		classes=1000,
		input_shape=input_shape
	)
	vgg_layermap = {l.name: l for l in vgg.layers}
	x = vgg_layermap["block3_pool"].output

	x = Convolution2D(embdim, 3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Convolution2D(embdim, 3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('sigmoid')(x)

	return Model(inputs=vgg.inputs, outputs=x)

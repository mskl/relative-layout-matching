import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D


def resnet(embdim: int = 256) -> Model:
	"""Take pretrained VGG16 and cut it in half."""
	resnet = tf.keras.applications.ResNet50(
		include_top=False, weights="imagenet", classes=1000
	)
	resnet_layermap = {l.name: l for l in resnet.layers}
	x = resnet_layermap["conv3_block4_out"].output
	# 1x1 convolutions to subsample the channels from 512 to 256
	x = Conv2D(embdim, (1, 1), activation='sigmoid')(x)
	return Model(resnet.input, x)


def resnet152(embdim: int = 256) -> Model:
	resnet = tf.keras.applications.ResNet152V2(
		include_top=False, weights="imagenet", classes=1000
	)
	resnet_layermap = {l.name: l for l in resnet.layers}
	x = resnet_layermap["conv3_block4_out"].output
	# 1x1 convolutions to subsample the channels from 512 to 256
	x = Conv2D(embdim, (1, 1), activation='sigmoid')(x)
	return Model(resnet.input, x)


def _to_unet(resnet: Model, down_block: str, up_block: str, embdim: int = 256) -> Model:
	resnet_layermap = {l.name: l for l in resnet.layers}

	p = resnet_layermap[down_block].output
	p = layers.Conv2D(embdim, (1, 1), activation='relu')(p)

	x = resnet_layermap[up_block].output
	x = layers.UpSampling2D(2)(x)
	x = layers.Conv2D(embdim, (1, 1), activation='relu')(x)

	m = layers.Concatenate(axis=-1)([x, p])

	m = layers.Conv2D(embdim, (3, 3), padding="same")(m)
	m = layers.BatchNormalization()(m)
	m = layers.ReLU()(m)

	m = layers.Conv2D(embdim, (3, 3), padding="same")(m)
	m = layers.BatchNormalization()(m)
	m = layers.ReLU()(m)

	m = layers.Conv2D(embdim, (1, 1), activation="sigmoid")(m)
	return Model(resnet.input, m)


def resnet_unet_50(embdim: int = 256, transfer: bool = True) -> Model:
	resnet = tf.keras.applications.ResNet50(
		include_top=False,
		weights="imagenet",
		pooling=None,
		classes=1000,
	)
	if not transfer:
		resnet = tf.keras.models.clone_model(resnet)
	return _to_unet(resnet, "conv3_block4_out", "conv4_block6_out", embdim)


def resnet_unet_101(embdim: int = 256, transfer: bool = True) -> Model:
	resnet = tf.keras.applications.ResNet101(
		include_top=False,
		weights="imagenet",
		pooling=None,
		classes=1000
	)
	if not transfer:
		resnet = tf.keras.models.clone_model(resnet)
	return _to_unet(resnet, "conv3_block4_out", "conv4_block23_out", embdim)


def _to_bos(original: Model) -> Model:
	cfg = original.get_config()
	cfg["layers"][0]["config"]["batch_input_shape"] = (None, None, None, 51)
	model = Model.from_config(cfg)

	old_layers = {l.name: l for l in original.layers}
	new_layers = {l.name: l for l in model.layers}

	for name, old_layer in old_layers.items():
		new_layer = new_layers[name]
		if old_layer.count_params() == 0:
			continue
		elif old_layer.count_params() != new_layer.count_params():
			wo, bo = old_layer.get_weights()
			wn, bn = new_layer.get_weights()
			wn[:wo.shape[0], :wo.shape[1], :wo.shape[2], :wo.shape[3]] = wo
			new_layer.set_weights([wn, bo])
		else:
			new_layer.set_weights(old_layer.get_weights())

	return model


def resnet_unet_50_bos(embdim: int = 256, transfer: bool = True) -> Model:
	original = resnet_unet_50(embdim, transfer)
	return _to_bos(original)


def resnet_unet_101_bos(embdim: int = 256, transfer: bool = True) -> Model:
	original = resnet_unet_101(embdim, transfer)
	return _to_bos(original)


def _to_flat(resnet: Model, last_block: str, embdim: int = 256) -> Model:
	rl = {l.name: l for l in resnet.layers}

	x = rl[last_block].output
	x = layers.Conv2D(embdim, (1, 1), activation='sigmoid')(x)
	short_model = Model(resnet.input, x)

	# Change the stride to keep larger output
	config = short_model.get_config()
	config["layers"][81]["strides"] = (1, 1)
	config["layers"][81]["config"]["strides"] = (1, 1)
	config["layers"][87]["strides"] = (1, 1)
	config["layers"][87]["config"]["strides"] = (1, 1)
	model = Model.from_config(config)
	model.set_weights(short_model.get_weights())
	return model


def resnet_101_flat(embdim: int = 256, transfer: bool = True) -> Model:
	resnet = tf.keras.applications.ResNet101(
		include_top=False,
		weights="imagenet",
		pooling=None,
		classes=1000
	)
	if not transfer:
		resnet = tf.keras.models.clone_model(resnet)
	return _to_flat(resnet, "conv4_block23_out", embdim)


def resnet_50_flat(embdim: int = 256, transfer: bool = True) -> Model:
	resnet = tf.keras.applications.ResNet50(
		include_top=False,
		weights="imagenet",
		pooling=None,
		classes=1000
	)
	if not transfer:
		resnet = tf.keras.models.clone_model(resnet)
	return _to_flat(resnet, "conv4_block6_out", embdim)

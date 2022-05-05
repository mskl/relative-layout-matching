from tensorflow.keras.layers import Input, Convolution2D, Activation, add, \
	SeparableConv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model


def conv_block(x, filters, t=False):
	conv = SeparableConv2D if not t else Conv2DTranspose

	x = Activation("relu")(x)
	x = conv(filters, 3, padding="same")(x)
	x = BatchNormalization()(x)

	x = Activation("relu")(x)
	x = conv(filters, 3, padding="same")(x)
	x = BatchNormalization()(x)
	return x


def unet(input_shape=(624, 880, 3), n_classes=12) -> Model:
	inputs = Input(shape=input_shape)

	# Entry block
	x = Convolution2D(32, 3, strides=2, padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	previous_block_activation = x  # Set aside residual

	# Blocks 1, 2, 3 are identical apart from the feature depth.
	for filters in [64, 128, 256]:
		x = conv_block(x, filters, False)
		x = MaxPooling2D(3, strides=2, padding="same")(x)

		# Project residual
		residual = Convolution2D(filters, 1, strides=2, padding="same")(
			previous_block_activation
		)
		x = add([x, residual])
		previous_block_activation = x

	for filters in [256]:
		x = conv_block(x, filters, True)
		x = UpSampling2D(2)(x)

		# Project residual
		residual = UpSampling2D(2)(previous_block_activation)
		residual = Convolution2D(filters, 1, padding="same")(residual)
		x = add([x, residual])
		previous_block_activation = x

	# Add a per-pixel classification layer, Sigmoid activation for multiclass multilabel
	outputs = Convolution2D(n_classes, 3, activation="sigmoid", padding="same")(x)
	return Model(inputs, outputs)

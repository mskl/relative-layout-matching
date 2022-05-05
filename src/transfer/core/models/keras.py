import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import data_adapter

from core.function import mean_embs
from core.losses import reconstruction_loss, triplet_loss, consistency_loss, simclr_loss


class PairModel(Model):
	def __init__(
		self,
		backbone_model: Model,
		consistency: float = 0.0,
		triplet: float = 0.0,
		reconstruction: float = 0.0,
		contrastive_kind: str = "triplet",
		consistency_kind: str = "pairs",
		l2_normalize: bool = True,
		*args,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.consistency = consistency
		self.triplet = triplet
		self.reconstruction = reconstruction
		self.backbone = backbone_model
		self.contrastive_kind = contrastive_kind
		self.consistency_kind = consistency_kind
		self.l2_normalize = l2_normalize

	def call(self, image_pairs, **kwargs):
		pred = self.backbone(image_pairs)
		if self.l2_normalize:
			pred = tf.math.l2_normalize(pred, axis=-1)
		return pred

	@tf.function()
	def train_step(self, data):
		x, y = data_adapter.expand_1d(data)
		with tf.GradientTape() as tape:
			p = self(x, training=True)

			emb1, emb2 = p[0::2], p[1::2]
			cls1, cls2 = y[0::2], y[1::2]
			memb1 = mean_embs(emb1, cls1)
			memb2 = mean_embs(emb2, cls2)

			losses = list()
			loss_dict = {}
			if self.triplet > 0:
				triplet = tf.constant(self.triplet, dtype="float32")
				if self.contrastive_kind == "triplet":
					triplet_loss_sum = triplet * triplet_loss(memb1, memb2)
				elif self.contrastive_kind == "simclr":
					triplet_loss_sum = triplet * simclr_loss(memb1, memb2)
				else:
					raise ValueError(f"Unknown loss {self.contrastive_kind}")
				loss_dict["triplet_loss"] = triplet_loss_sum
				losses.append(triplet_loss_sum)
			if self.consistency > 0:
				consistency = tf.constant(self.consistency, dtype="float32")
				consistency_loss_sum = consistency * consistency_loss(p, y, self.consistency_kind)
				loss_dict["consistency_loss"] = consistency_loss_sum
				losses.append(consistency_loss_sum)
			if self.reconstruction > 0:
				reconstruction = tf.constant(self.reconstruction, dtype="float32")
				reconstruction_loss_sum = reconstruction * reconstruction_loss(
					emb1, cls1, memb1, emb2, cls2, memb2
				)
				loss_dict["reconstruction_loss"] = reconstruction_loss_sum
				losses.append(reconstruction_loss_sum)

		grads = tape.gradient(losses, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return loss_dict

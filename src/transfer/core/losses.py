import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def onesim(embedding: tf.Tensor, mask: tf.Tensor, subsample_size: int = 100):
	"""
	embedding: (78, 110, 256)
	mask: (78, 110)
	"""
	masked_embs = tf.boolean_mask(embedding, mask)
	all_indexes = tf.range(tf.shape(masked_embs)[0])
	sel_indexes = tf.random.shuffle(all_indexes)[:subsample_size]
	sampled_embs = tf.gather(masked_embs, sel_indexes)
	# sampled_embs = tf.math.l2_normalize(sampled_embs, axis=1)
	similarity = tf.reduce_mean(
		1.0 - tf.matmul(sampled_embs, sampled_embs, transpose_b=True)
	)
	return tf.where(tf.math.is_nan(similarity), 0., similarity)


@tf.function
def varsim(embedding: tf.Tensor, mask: tf.Tensor, subsample_size: int = 100):
	masked_embs = tf.boolean_mask(embedding, mask)
	all_indexes = tf.range(tf.shape(masked_embs)[0])
	sel_indexes = tf.random.shuffle(all_indexes)[:subsample_size]
	sampled_embs = tf.gather(masked_embs, sel_indexes)
	variance = tf.math.reduce_variance(sampled_embs, axis=-1)
	return tf.where(tf.math.is_nan(variance), 0., variance)


@tf.function
def embsim(se: tf.Tensor, sm: tf.Tensor, consistency_kind: str):
	"""
	se: (78, 110, 256)
	sm: (78, 110, 12)
	"""
	class_size = tf.shape(sm)[-1]
	similarities = tf.TensorArray(tf.float32, size=class_size)
	for i in range(class_size):
		if consistency_kind == "pairs":
			val = onesim(se, sm[:, :, i])
		elif consistency_kind == "variance":
			val = varsim(se, sm[:, :, i])
		else:
			raise ValueError(f"Unkown consistency kind {consistency_kind}.")
		similarities = similarities.write(i, val)
	return tf.reduce_mean(similarities.stack())


@tf.function
def consistency_loss(emb: tf.Tensor, cls: tf.Tensor, consistency_kind: str):
	"""
	emb: (6, 78, 110, 256)
	cls: (6, 78, 110, 12)
	"""
	batch_size = tf.shape(emb)[0]
	batch_loss = tf.TensorArray(tf.float32, size=batch_size)
	for i in tf.range(batch_size):
		loss = embsim(emb[i], cls[i], consistency_kind)
		batch_loss = batch_loss.write(i, loss)
	return tf.reduce_mean(batch_loss.stack())


@tf.function
def triplet_loss(memb1, memb2):
	@tf.function
	def triplet_loss_fn(args):
		memb1 = tf.expand_dims(args[0], axis=0)
		memb2 = tf.expand_dims(args[1], axis=0)

		trange = tf.range(tf.shape(memb1)[-1])
		y_true = tf.concat((trange, trange), axis=0)
		y_pred = tf.concat((memb1, memb2), axis=2)
		y_pred = tf.transpose(y_pred[0], perm=[1, 0])

		return tfa.losses.triplet_hard_loss(y_true, y_pred, distance_metric="angular")
	return tf.reduce_mean(
		tf.vectorized_map(triplet_loss_fn, (memb1, memb2))
	)


@tf.function
def simclr_loss(memb1, memb2, temperature=1.0):
	"""
	memb1: TensorShape([3, 256, 12])
	memb2: TensorShape([3, 256, 12])

	Only compute similarities between pairs of different documents.
	"""
	memb1 = tf.transpose(memb1, perm=[0, 2, 1])
	memb2 = tf.transpose(memb2, perm=[0, 2, 1])

	hidden1 = tf.math.l2_normalize(memb1, axis=-1)
	hidden2 = tf.math.l2_normalize(memb2, axis=-1)

	class_size = tf.shape(hidden1)[1]
	batch_size = tf.shape(hidden1)[0]

	onemask = tf.one_hot(tf.range(class_size), class_size)
	onemask = tf.expand_dims(onemask, axis=0)

	# Repeat the mask for each sample in batch
	masks = tf.repeat(onemask, batch_size, axis=0)

	logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
	logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

	loss_ab = tf.losses.categorical_crossentropy(
		masks, logits_ab, from_logits=True
	)
	loss_ba = tf.losses.categorical_crossentropy(
		masks, logits_ba, from_logits=True
	)

	return tf.reduce_mean(loss_ab + loss_ba)


@tf.function
def reconstruction_loss(emb1, cls1, memb1, emb2, cls2, memb2):
	sim_aa = tf.einsum('bwhe,bec->bwhc', emb1, memb1)
	sim_bb = tf.einsum('bwhe,bec->bwhc', emb2, memb2)
	sim_ab = tf.einsum('bwhe,bec->bwhc', emb1, memb2)
	sim_ba = tf.einsum('bwhe,bec->bwhc', emb2, memb1)
	return tf.reduce_mean([
		tfa.losses.sigmoid_focal_crossentropy(cls1, sim_aa),
		tfa.losses.sigmoid_focal_crossentropy(cls2, sim_bb),
		tfa.losses.sigmoid_focal_crossentropy(cls2, sim_ab),
		tfa.losses.sigmoid_focal_crossentropy(cls1, sim_ba),
	])

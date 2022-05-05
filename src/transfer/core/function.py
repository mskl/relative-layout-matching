import tensorflow as tf


@tf.function
def mean_embs(embs: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    """Average embeddings when masks are given

    Parameters
    ----------
    embs:  Tensor of float32 [b, w, h, e]
    masks: Tensor of int32   [b, w, h, c]
    """
    emb_dim = tf.shape(embs)[-1]

    masks_repeat = tf.repeat(tf.expand_dims(masks, axis=-1), emb_dim, axis=-1)
    masks_repeat = tf.transpose(masks_repeat, perm=[0, 1, 2, 4, 3])
    masks_repeat = tf.cast(masks_repeat, dtype="float32")

    counts = tf.reduce_sum(masks, axis=(1, 2), keepdims=True)
    counts_match = tf.squeeze(counts, axis=1)
    counts_match = tf.cast(counts_match, dtype="float32")

    masked_embs = masks_repeat * tf.expand_dims(embs, axis=-1)
    return tf.math.divide_no_nan(
        tf.reduce_sum(masked_embs, axis=(1, 2)), counts_match
    )


@ tf.function
def query_embedding_distance(query: tf.Tensor, emb: tf.Tensor, metric: str = "cosine") -> tf.Tensor:
    """Get distances between 0 and 1 between query and embedding.

    Query `query` can be of shape (E, C) as well as (B, E, C). The `emb` is expected
    to be of shape (W, H, E). Supported metrics are `euclidean` and `cosine`.

    Returns tensor of shape (W, H, E).
    """

    # Add missing batch dimension if needed
    # if tf.rank(query) == 2:
    #     query = tf.expand_dims(query, axis=0)

    if metric == "euclidean":
        x = tf.expand_dims(emb, axis=-1) - query
        x = tf.norm(x, axis=2)
        x = x / tf.reduce_max(x, axis=(0, 1))
        return 1 - x
    elif metric == "cosine":
        # NOTE: We expect the embeddings and query to be l2 normalized
        return tf.matmul(emb, query)
    raise NotImplementedError(f"Uknown metric {metric}")

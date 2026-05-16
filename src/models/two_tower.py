"""
Two-Tower neural recommendation model built with Keras.

User Tower: userId embedding + user features → 64-d normalized vector
Item Tower: movieId embedding + item features → 64-d normalized vector
Score: dot product (cosine similarity after L2 norm)
"""
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
class L2Normalize(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


def build_tower(input_dim: int, embedding_dim: int, extra_feature_dim: int, name: str):
    embedding_input = keras.Input(shape=(1,), name=f"{name}_id")
    extra_input = keras.Input(shape=(extra_feature_dim,), name=f"{name}_features")

    emb = keras.layers.Embedding(input_dim, embedding_dim, name=f"{name}_embedding")(
        embedding_input
    )
    emb = keras.layers.Flatten()(emb)

    x = keras.layers.Concatenate()([emb, extra_input])

    for units in [128, 64, 32]:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.2)(x)

    # L2 normalize so dot product == cosine similarity
    output = L2Normalize(name=f"{name}_vector")(x)

    return keras.Model(
        inputs=[embedding_input, extra_input], outputs=output, name=f"{name}_tower"
    )


def build_two_tower_model(
    num_users: int,
    num_items: int,
    user_feature_dim: int,
    item_feature_dim: int,
    embedding_dim: int = 64,
):
    user_tower = build_tower(num_users, embedding_dim, user_feature_dim, "user")
    item_tower = build_tower(num_items, embedding_dim, item_feature_dim, "item")

    user_id = keras.Input(shape=(1,), name="user_id")
    user_feats = keras.Input(shape=(user_feature_dim,), name="user_features")
    item_id = keras.Input(shape=(1,), name="item_id")
    item_feats = keras.Input(shape=(item_feature_dim,), name="item_features")

    user_vec = user_tower([user_id, user_feats])
    item_vec = item_tower([item_id, item_feats])

    score = keras.layers.Dot(axes=1, normalize=False, name="dot_product")(
        [user_vec, item_vec]
    )
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(score)

    model = keras.Model(
        inputs=[user_id, user_feats, item_id, item_feats],
        outputs=output,
        name="two_tower",
    )
    return model, user_tower, item_tower

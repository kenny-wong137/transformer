from pipeline import get_data
from transformer import Transformer

import tensorflow as tf


def train_step(model, loss_obj, optimizer, inputs, targets):
    with tf.GradientTape as tape:
        preds = model(inputs)
        loss_val = loss_obj(targets, preds)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def demo_step():
    pass
    
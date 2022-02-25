#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from tensorflow.keras.utils import Progbar


class Trainer():

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 train_metrics,
                 val_metrics=dict()):
        self.model = model
        self.optimizer = optimizer
        assert 'loss' not in train_metrics, "train metrics dictionary must not contain a 'loss' item, it must be passed through loss_fn argument"
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            result = self.model(x, training=True)
            loss_value = self.loss_fn(y, result)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        for key, metric in self.train_metrics.items():
            if key != 'loss':
                metric.update_state(y, result)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_result = self.model(x, training=False)
        for key, metric in self.val_metrics.items():
            metric.update_state(y, val_result)

    def fit(self,
            train_dataset,
            val_dataset,
            epochs,
            _callbacks=None,
            interval=0.5):
        logs = {}
        callbacks = tf.keras.callbacks.CallbackList(_callbacks,
                                                    add_history=True,
                                                    model=self.model)
        callbacks.on_train_begin(logs=logs)
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            callbacks.on_epoch_begin(epoch, logs=logs)
            pb_i = Progbar(len(train_dataset.x),
                           interval=interval,
                           unit_name="step")

            for step, (x_batch_train,
                       y_batch_train) in enumerate(train_dataset):

                self.model.reset_states()
                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_train_batch_begin(step, logs=logs)

                loss_val = self.train_step(x_batch_train, y_batch_train)

                self.train_metrics['loss'] = loss_val.eval()

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)

                pb_i.add(x_batch_train.shape[0],
                         [(key, metric.result()) if key != 'loss' else
                          (key, metric)
                          for key, metric in self.train_metrics.items()])

            if val_dataset != None and self.val_metrics != None:
                assert len(
                    self.val_metrics
                ) > 0, "You have passed an empty dictionary for validation metrics"
                pb_i = Progbar(len(val_dataset.x),
                               interval=interval,
                               unit_name="step")
                # Run a validation loop at the end of each epoch.
                for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                    self.test_step(x_batch_val, y_batch_val)
                    pb_i.add(x_batch_val.shape[0],
                             values=[
                                 (key, metric.result())
                                 for key, metric in self.val_metrics.items()
                             ])
            # Reset training metrics at the end of each epoch
            for key, metric in self.val_metrics.items():
                logs[key] = metric.result()
                metric.reset_states()
            for key, metric in self.train_metrics.items():
                if key != 'loss':
                    logs[key] = metric.result()
                    metric.reset_states()
                else:
                    logs[key] = metric
            callbacks.on_epoch_end(epoch, logs=logs)

        callbacks.on_train_end(logs=logs)
        history_object = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.History):
                history_object = cb
        return history_object

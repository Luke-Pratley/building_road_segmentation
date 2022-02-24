#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from tensorflow.keras.utils import Progbar


class Trainer():

    def __init__(self, model, loss_fn, optimizer, train_acc_metric,
                 val_acc_metric):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.val_acc_metric = val_acc_metric

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            result = self.model(x, training=True)
            loss_value = self.loss_fn(y, result)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        self.train_acc_metric.update_state(y, result)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_result = self.model(x, training=False)
        self.val_acc_metric.update_state(y, val_result)

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
            print(f'epoch: {epoch}')
            callbacks.on_epoch_begin(epoch, logs=logs)
            pb_i = Progbar(len(train_dataset.x),
                           interval=interval,
                           unit_name="step")

            for step, (x_batch_train,
                       y_batch_train) in enumerate(train_dataset):

                self.model.reset_states()
                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_train_batch_begin(step, logs=logs)

                loss_value = self.train_step(x_batch_train, y_batch_train)

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)
                train_acc = self.train_acc_metric.result()

                loss_val = loss_value
                acc_val = train_acc
                pb_i.add(train_dataset.batch_size,
                         values=[('loss', loss_val), ('acc', acc_val)])

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            if val_dataset != None:
                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in val_dataset:
                    self.test_step(x_batch_val, y_batch_val)

                val_acc = self.val_acc_metric.result()
                self.val_acc_metric.reset_states()
                print("Validation acc: %.4f" % (float(val_acc), ))
                print("Time taken: %.2fs" % (time.time() - start_time))
            callbacks.on_epoch_end(epoch, logs=logs)

        callbacks.on_train_end(logs=logs)
        history_object = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.History):
                history_object = cb
        return history_object

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
        """
        In the case that we need to do a custom training loop we can use this object to do the trianing, e.g., have fine control over the optimization process.

        Inputs:
            model: the model to be trained
            loss_fn: the loss function object
            optimzer: the optimization object
            train_metrics: a dictionary of metrics to use while training
            val_metrics: a dictionary of metrics to use while doing validation
        """
        self.model = model
        self.optimizer = optimizer
        assert 'loss' not in train_metrics, "train metrics dictionary must not contain a 'loss' item, it must be passed through loss_fn argument"
        self.train_metrics = train_metrics
        self.train_metrics['loss'] = tf.keras.metrics.Mean()
        self.val_metrics = val_metrics
        self.loss_fn = loss_fn

    # this is the decorator converts the function into a tensorflow "Function" where the function is run in a TensorFlow graph, e.g., it allows us to compute gradients and execute eagerly
    @tf.function
    def train_step(self, x, y):
        """
        The training step that updates the weights through backpropigation and minimizes the loss.

        Inputs:
            x: training features
            y: training labels

        Outputs:
            loss_value: a tensorflow object that holds the loss value

        """
        with tf.GradientTape() as tape:
            result = self.model(x, training=True)
            loss_value = self.loss_fn(y, result)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        for key, metric in self.train_metrics.items():
            if key != 'loss':
                metric.update_state(y, result)
            else:
                metric.update_state(loss_value)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        """
        The test step that updates the metrics using the validation data.

        Inputs:
            x: training features
            y: training labels

        """
        val_result = self.model(x, training=False)
        for key, metric in self.val_metrics.items():
            metric.update_state(y, val_result)

    def fit(self,
            train_dataset,
            val_dataset,
            epochs,
            _callbacks=None,
            interval=0.5):
        """
        The fit function performs the training loop.

        Inputs:
            train_dataset: the dataset as a generator
            val_dataset: the validation set as a generator
            epochs: the total amount of epochs
            _callbacks: the list of callbacks, e.g. ModelCheckpoint
            interval: the interval (seconds) for updating the training 
                      progress bar
        """
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

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)

                pb_i.add(x_batch_train.shape[0],
                         [(key, metric.result())
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
            callbacks.on_epoch_end(epoch, logs=logs)
            # Reset training metrics at the end of each epoch
            for key, metric in self.val_metrics.items():
                logs[key] = metric.result()
                metric.reset_states()
            for key, metric in self.train_metrics.items():
                    logs[key] = metric.result()
                    metric.reset_states()

        callbacks.on_train_end(logs=logs)
        history_object = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.History):
                history_object = cb
        return history_object

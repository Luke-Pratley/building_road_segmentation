#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from tqdm import tqdm


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

    def fit(self, train_dataset, val_dataset, epochs):
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch, ))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train,
                       y_batch_train) in enumerate(tqdm(train_dataset)):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                # Log every 200 batches.
                if step % 100 == 0:
                    print(f'Batch {step}')
                    tf.print('Loss', loss_value)
                    print("Seen so far: %d samples" %
                          ((step + 1) * train_dataset.batch_size))

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            tf.print("Training acc", (train_acc))

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

import argparse
from functools import partial
import time

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from ocddetection import data, models


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('path', type=str)

    # Hyperparameter
    parser.add_argument('--learning-rate', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--keep-state-rate', type=float, default=.5)

    # Model
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--dropout-rate', type=float, default=.4)

    return parser


def main():
    args = arg_parser().parse_args()

    # Data
    print('Reading files from {}'.format(args.path))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(data.LABEL2IDX.keys()), list(data.LABEL2IDX.values())),
        0
    )

    df_train, df_val, _ = data.split(
        data.files(args.path),
        validation=[(1, 2)],
        test=[(2, 3), (2, 4), (3, 3), (3, 4)]
    )

    train = data \
        .to_centralised(df_train) \
        .map(partial(data.preprocess, sensors=data.SENSORS, label=data.MID_LEVEL, table=table)) \
        .filter(data.filter_nan) \
        .window(args.window_size, shift=args.window_size // 2) \
        .flat_map(partial(data.windows, window_size=args.window_size)) \
        .batch(args.batch_size, drop_remainder=True)

    val = data \
        .to_centralised(df_val) \
        .map(partial(data.preprocess, sensors=data.SENSORS, label=data.MID_LEVEL, table=table)) \
        .filter(data.filter_nan) \
        .window(args.window_size, shift=args.window_size // 2) \
        .flat_map(partial(data.windows, window_size=args.window_size)) \
        .batch(args.batch_size, drop_remainder=True)

    # Model
    print('Building model')
    model = models.build(
        (args.window_size, len(data.SENSORS)),
        args.batch_size,
        len(data.LABEL2IDX),
        args.hidden_size,
        args.dropout_rate
    )

    # Optimizer
    print('Setting up training')
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Checkpoints
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')
    
    # Logs
    train_log_dir = './logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    val_log_dir = './logs/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Train Loop
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mean_f1_score = tfa.metrics.F1Score(len(data.LABEL2IDX), average='macro', name='train_mean_f1_score')
    train_weighted_f1_score = tfa.metrics.F1Score(len(data.LABEL2IDX), average='weighted', name='train_weighted_f1_score')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_mean_f1_score = tfa.metrics.F1Score(len(data.LABEL2IDX), average='macro', name='val_mean_f1_score')
    val_weighted_f1_score = tfa.metrics.F1Score(len(data.LABEL2IDX), average='weighted', name='val_weighted_f1_score')

    signature = [
        tf.TensorSpec(shape=(None, args.window_size, len(data.SENSORS)), dtype=tf.float32),
        tf.TensorSpec(shape=(None, args.window_size), dtype=tf.int32)
    ]

    train_fn = tf.function(
        partial(
            models.train_step,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            output_size=len(data.LABEL2IDX),
            train_loss=train_loss,
            train_mean_f1_score=train_mean_f1_score,
            train_weighted_f1_score=train_weighted_f1_score
        ),
        input_signature=signature
    )

    val_fn = tf.function(
        partial(
            models.test_step,
            model=model,
            loss_fn=loss_fn,
            output_size=len(data.LABEL2IDX),
            test_loss=val_loss,
            test_mean_f1_score=val_mean_f1_score,
            test_weighted_f1_score=val_weighted_f1_score
        ),
        input_signature=signature
    )

    print('Training for {} epochs'.format(args.epochs))
    for epoch in range(args.epochs):
        start = time.time()
        
        train_loss.reset_states()
        train_mean_f1_score.reset_states()
        train_weighted_f1_score.reset_states()

        val_loss.reset_states()
        val_mean_f1_score.reset_states()
        val_weighted_f1_score.reset_states()

        for (X_train, y_train) in train:
            train_fn(X_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('mean_f1_score', train_mean_f1_score.result(), step=epoch)
            tf.summary.scalar('weighted_f1_score', train_weighted_f1_score.result(), step=epoch)

        for (X_val, y_val) in val:
            val_fn(X_val, y_val)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('mean_f1_score', val_mean_f1_score.result(), step=epoch)
            tf.summary.scalar('weighted_f1_score', val_weighted_f1_score.result(), step=epoch)

        if (epoch + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print(
            'Epoch {} Loss {:.4f} Mean F1 {:.4f} Weighted F1 {:.4f} Validation Loss {:.4f} Validation Mean F1 {:.4f} Validation Weighted F1 {:.4f}'.format(
                epoch + 1,
                train_loss.result(),
                train_mean_f1_score.result(),
                train_weighted_f1_score.result(),
                val_loss.result(),
                val_mean_f1_score.result(),
                val_weighted_f1_score.result(),
            )
        )

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    main()

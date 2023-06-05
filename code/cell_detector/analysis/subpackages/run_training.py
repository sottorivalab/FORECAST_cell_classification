import tensorflow as tf
import h5py
import numpy as np
import os
import scipy.io as sio
import time
from datetime import datetime


def run_training(network, opts):
    hft = h5py.File(os.path.join(opts.data_dir, opts.train_data_filename), 'r')
    data_set_train = hft.get('data')
    label_set_train = hft.get('labels')
    hfv = h5py.File(os.path.join(opts.data_dir, opts.valid_data_filename), 'r')
    data_set_valid = hfv.get('data')
    label_set_valid = hfv.get('labels')

    opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = data_set_train.shape
    _, opts.label_height, opts.label_width, opts.in_label_dim = label_set_train.shape
    opts.num_examples_per_epoch_valid, _, _, _ = data_set_valid.shape

    train = np.arange(opts.num_examples_per_epoch_train)
    np.random.shuffle(train)
    valid = np.arange(opts.num_examples_per_epoch_valid)
    np.random.shuffle(valid)

    network.run_checks(opts=opts)
    data = np.zeros([opts.batch_size, opts.image_height, opts.image_width, opts.in_feat_dim],
                          dtype=np.float32)
    label = np.zeros([opts.batch_size, opts.label_height, opts.label_width, opts.in_label_dim],
                           dtype=np.float32)

    train_count = int((len(train) / opts.batch_size) + 1)
    valid_count = int((len(valid) / opts.batch_size) + 1)

    global_step = tf.Variable(0.0, trainable=False)
    network.LearningRate = tf.placeholder(tf.float32)

    logits = network.inference()

    imr0 = logits[0:1, :, :, 0:1]
    imr1 = network.images[0:1, :, :, 0:3]
    imr2 = network.labels[0:1, :, :, 0:1]
    avg_loss_tensor = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    _ = tf.summary.image('Output_1', imr0)
    _ = tf.summary.image('Input_1', imr1)
    _ = tf.summary.image('label', imr2)
    _ = tf.summary.scalar('Average Training Loss', avg_loss_tensor)

    loss = network.loss_function()
    train_op = network.train(global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    avg_training_loss = 0.0
    avg_validation_loss = 0.0
    training_loss = 50000.0
    validation_loss = 50000.0

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'valid'), sess.graph)
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        curr_epoch = 0
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            curr_epoch = int(global_step)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
            workspace = sio.loadmat(os.path.join(opts.exp_dir, 'avg_training_loss.mat'))
            avg_training_loss = np.array(workspace['avg_training_loss'])
            avg_validation_loss = np.array(workspace['avg_validation_loss'])
            training_loss = avg_training_loss[0, -1]
            validation_loss = avg_validation_loss[0, -1]
        else:
            sess.run(init)
            print('No checkpoint file found', flush=True)

        for epoch in range(curr_epoch, opts.num_of_epoch):
            lr = 0.001
            opts.current_epoch_num = global_step
            start = 0
            avg_loss = 0.0
            start_time = time.time()
            for step in range(train_count):
                avg_train_loss_op = tf.assign(avg_loss_tensor, training_loss)
                end = start + opts.batch_size
                start_time_step = time.time()
                temp_indices = train[start:end]
                np.random.shuffle(temp_indices)
                indices = np.sort(temp_indices)
                data_set_train.read_direct(data, np.s_[indices, :, :, :])
                label_set_train.read_direct(label, np.s_[indices, :, :, :])
                data_float32 = np.float32(data)
                label_float32 = np.float32(label)
                if (step % int(50) == 0) or (step == 0) or (step == train_count-1):
                    summary_str, _, loss_value, logits_out, _ = sess.run(
                        [summary_op, train_op, loss, logits, avg_train_loss_op],
                        feed_dict={network.images: data_float32,
                                   network.labels: label_float32,
                                   network.LearningRate: lr})
                    train_writer.add_summary(summary_str, step + epoch * train_count)
                    inter = {'logits': logits_out,
                             'input': data_float32,
                             'label': label_float32}
                    sio.savemat(os.path.join(opts.exp_dir, 'inter_train.mat'), inter)
                    duration = time.time() - start_time_step
                    format_str = (
                        '%s: epoch %d, step %d/ %d, Training Loss = %.2f, (%.2f sec/step)')
                    print(
                        format_str % (
                            datetime.now(), epoch + 1, step + 1, train_count, loss_value, float(duration)), flush=True)
                else:
                    _, loss_value = sess.run([train_op, loss],
                                             feed_dict={network.images: data_float32,
                                                        network.labels: label_float32,
                                                        network.LearningRate: lr})

                if end + opts.batch_size > len(train) - 1:
                    end = len(train) - opts.batch_size

                start = end
                avg_loss += loss_value
                training_loss = avg_loss / (step + 1)

            training_loss = avg_loss / train_count

            start = 0
            avg_loss = 0.0
            for step in range(valid_count):
                avg_valid_loss_op = tf.assign(avg_loss_tensor, validation_loss)
                end = start + opts.batch_size
                start_time_step = time.time()
                indices = np.sort(valid[start:end])
                data_set_valid.read_direct(data, np.s_[indices, :, :, :])
                label_set_valid.read_direct(label, np.s_[indices, :, :, :])
                data_float32 = np.float32(data)
                label_float32 = np.float32(label)
                if (step % int(5) == 0) or (step == 0) or (step == valid_count-1):
                    summary_str, loss_value, logits_out, _ = sess.run(
                        [summary_op, loss, logits, avg_valid_loss_op],
                        feed_dict={network.images: data_float32,
                                   network.labels: label_float32,
                                   })
                    valid_writer.add_summary(summary_str, step + epoch * valid_count)
                    inter = {'logits': logits_out,
                             'input': data_float32,
                             'label': label_float32}
                    sio.savemat(os.path.join(opts.exp_dir, 'inter_valid.mat'), inter)
                    duration = time.time() - start_time_step
                    format_str = (
                        '%s: epoch %d, step %d/ %d, Validation Loss = %.2f, (%.2f sec/step)')
                    print(
                        format_str % (
                            datetime.now(), epoch + 1, step + 1, valid_count, loss_value, float(duration)), flush=True)
                else:
                    loss_value, logits_out = sess.run([loss, logits],
                                                      feed_dict={network.images: data_float32,
                                                                 network.labels: label_float32,
                                                                 })

                if end + opts.batch_size > len(valid) - 1:
                    end = len(valid) - opts.batch_size

                start = end
                avg_loss += loss_value
                validation_loss = avg_loss / (step + 1)

            validation_loss = avg_loss / valid_count
            # Save the model after each epoch.
            checkpoint_path = os.path.join(opts.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            global_step = global_step + 1
            # Average loss on training and validation datasets.
            duration = time.time() - start_time
            format_str = (
                '%s: epoch %d, Training Loss = %.2f, Validation Loss = %.2f, (%.2f sec/epoch)')
            print(format_str % (
                datetime.now(), epoch + 1, training_loss, validation_loss, float(duration)), flush=True)
            if epoch == 0:
                avg_training_loss = [int(training_loss)]
                avg_validation_loss = [int(validation_loss)]
            else:
                avg_training_loss = np.append(avg_training_loss, [int(training_loss)])
                avg_validation_loss = np.append(avg_validation_loss, [int(validation_loss)])
            avg_training_loss_dict = {'avg_training_loss': avg_training_loss,
                                      'avg_validation_loss': avg_validation_loss}
            sio.savemat(file_name=os.path.join(opts.exp_dir, 'avg_training_loss.mat'), mdict=avg_training_loss_dict)
            print(avg_training_loss, flush=True)
            print(avg_validation_loss, flush=True)
    hft.close()
    hfv.close()
    return avg_training_loss

import config
import core
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys
sys.dont_write_bytecode = True


np.set_printoptions(3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=16)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tag', type=str, default='default')
    args = parser.parse_args()
    return args


def build_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()

    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]

    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)

    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))

    gradient_vars = [(accumulator / accumulation_counter, var)
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    gradient_vars_h = []
    gradient_vars_a = []
    for accumulate_grad, var in gradient_vars:
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        else:
            raise ValueError

    train_step_h = optimizer.apply_gradients(gradient_vars_h)
    train_step_a = optimizer.apply_gradients(gradient_vars_a)

    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_h, train_step_a


def build_training_graph(num_agents):

    s = tf.placeholder(tf.float32, [num_agents, 8])
    s_ref = tf.placeholder(tf.float32, [num_agents, 8])

    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    h, mask, indices = core.network_cbf(
        x=x, r=config.DIST_MIN_THRES, indices=None)
    u = core.network_action(
        s=s, s_ref=s_ref, obs_radius=config.OBS_RADIUS, indices=indices)
    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier(
        h=h, s=s, indices=indices)
    (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv, acc_dang_deriv,
     acc_safe_deriv, acc_medium_deriv) = core.loss_derivatives(
        s=s, u=u, h=h, x=x, indices=indices)

    loss_action = core.loss_actions(s=s, u=u, s_ref=s_ref, indices=indices)

    # the weight of each loss item requires careful tuning
    loss_list = [loss_dang, loss_safe, 3 * loss_dang_deriv,
                 loss_safe_deriv, 2 * loss_medium_deriv, 0.5 * loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv,
                acc_safe_deriv, acc_medium_deriv]

    weight_loss = [
        config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, s_ref, u, loss_list, loss, acc_list


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    s, s_ref, u, loss_list, loss, acc_list = build_training_graph(
        args.num_agents)
    zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(
        loss)
    accumulate_ops.append(loss_list)
    accumulate_ops.append(acc_list)

    accumulation_steps = config.INNER_LOOPS

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if args.model_path:
            saver.restore(sess, args.model_path)

        loss_lists_np = []
        acc_lists_np = []
        dist_errors_np = []
        dist_errors_baseline_np = []

        safety_ratios_epoch = []
        safety_ratios_epoch_baseline = []
    
        scene = core.Cityscape(args.num_agents)
        start_time = time.time()
        
        for istep in range(config.TRAIN_STEPS):
            scene.reset()
            s_np = np.concatenate(
                [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
            sess.run(zero_ops)
            for t in range(scene.steps):
                s_ref_np = np.concatenate(
                    [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
                for i in range(accumulation_steps):
                    u_np, out = sess.run([u, accumulate_ops], feed_dict={
                                         s: s_np, s_ref: s_ref_np})
                    dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                    s_np = s_np + dsdt * config.TIME_STEP
                    safety_ratio = 1 - np.mean(
                        core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                    safety_ratio = np.mean(safety_ratio == 1)
                    safety_ratios_epoch.append(safety_ratio)
                    loss_list_np, acc_list_np = out[-2], out[-1]
                    loss_lists_np.append(loss_list_np)
                    acc_lists_np.append(acc_list_np)

            dist_errors_np.append(np.mean(np.linalg.norm(
                s_np[:, :3] - s_ref_np[:, :3], axis=1)))

            s_np = np.concatenate(
                [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
            for t in range(scene.steps):
                s_ref_np = np.concatenate(
                    [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
                for i in range(accumulation_steps):
                    u_np = core.quadrotor_controller_np(s_np, s_ref_np)
                    dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                    s_np = s_np + dsdt * config.TIME_STEP
                    safety_ratio = 1 - np.mean(
                        core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                    safety_ratio = np.mean(safety_ratio == 1)
                    safety_ratios_epoch_baseline.append(safety_ratio)
            dist_errors_baseline_np.append(
                np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))

            if np.mod(istep // 10, 2) == 0:
                sess.run(train_step_h)
            else:
                sess.run(train_step_a)

            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                print('Step: {}, Time: {:.1f}, Loss: {}, Dist: {:.3f}, Safety Rate: {:.3f}'.format(
                    istep, time.time() - start_time, np.mean(loss_lists_np, axis=0),
                    np.mean(dist_errors_np), np.mean(safety_ratios_epoch)))
                start_time = time.time()
                (loss_lists_np, acc_lists_np, dist_errors_np, dist_errors_baseline_np, safety_ratios_epoch,
                 safety_ratios_epoch_baseline) = [], [], [], [], [], []

            if np.mod(istep, config.SAVE_STEPS) == 0 or istep + 1 == config.TRAIN_STEPS:
                saver.save(
                    sess, 'models/model_{}_iter_{}'.format(args.tag, istep))


if __name__ == '__main__':
    main()

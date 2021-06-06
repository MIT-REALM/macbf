import sys
sys.dont_write_bytecode = True
import os
from yaml import load, Loader

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import config
import pickle
from yaml import load, Loader
from scipy import interpolate
from dijkstra import Dijkstra


class Cityscape(object):
    OBSTACLES = [[0, 4,  2, 6], # x1, y1, x2, y2
                 [4, 4,  6, 6],
                 [8, 4, 10, 6],
    ]

    def __init__(self, num_agents, area_size=10, max_steps=12, show=False):
        self.num_agents = num_agents
        self.area_size = area_size
        self.show = show
        self.max_steps = max_steps
        
    def reset(self):
        self.init_obstacles()
        self.init_broadlines()
        self.reset_starting_points()
        self.reset_end_points()
        self.reset_reference_paths()

    def init_obstacles(self):
        self.obstacle_points = []
        for x1, y1, x2, y2 in self.OBSTACLES:
            xrnd = np.linspace(x1, x2, int(abs(x1-x2) * 3))[:, np.newaxis]
            for r in np.linspace(0, 1, int(abs(y1-y2) * 3)):
                yrnd = np.ones_like(xrnd) * (r * y1 + (1-r) * y2)
                self.obstacle_points.append(np.concatenate([xrnd, yrnd], axis=1))
            
            yrnd = np.linspace(y1, y2, int(abs(y1-y2) * 3))[:, np.newaxis]
            for r in np.linspace(0, 1, int(abs(x1-x2) * 3)):
                xrnd = np.ones_like(yrnd) * (r * x1 + (1-r) * x2)
                self.obstacle_points.append(np.concatenate([xrnd, yrnd], axis=1))
        self.obstacle_points = np.concatenate(self.obstacle_points, axis=0)

    def init_broadlines(self):
        area_size = self.area_size
        bx, by = [], []
        bx.append(np.linspace(-1, area_size + 1, int(area_size * 2)))
        by.append(-np.ones(int(area_size * 2)))
        bx.append(np.linspace(-1, area_size + 1, int(area_size * 2)))
        by.append(np.ones(int(area_size * 2)) + area_size)
        by.append(np.linspace(-1, area_size + 1, int(area_size * 2)))
        bx.append(-np.ones(int(area_size * 2)))
        by.append(np.linspace(-1, area_size + 1, int(area_size * 2)))
        bx.append(np.ones(int(area_size * 2)) + area_size)
        bx = np.concatenate(bx, axis=0)[:, np.newaxis]
        by = np.concatenate(by, axis=0)[:, np.newaxis]
        self.broadline_points = np.concatenate([bx, by], axis=1)

    def reset_starting_points(self):
        start_points = np.zeros((self.num_agents, 3), dtype=np.float32)
        for i in range(self.num_agents):
            random_start = np.random.randint(
                low=1, high=self.area_size-1, size=(1, 2))
            dist = min(np.amin(np.linalg.norm(
                start_points[:, :2] - random_start, axis=1)), 
                np.amin(np.linalg.norm(
                    self.obstacle_points - random_start, axis=1)))
            while dist < 1:
                random_start = np.random.randint(
                    low=1, high=self.area_size-1, size=(1, 2))
                dist = min(np.amin(np.linalg.norm(
                    start_points[:, :2] - random_start, axis=1)), 
                    np.amin(np.linalg.norm(
                        self.obstacle_points - random_start, axis=1)))
            start_points[i, :2] = random_start
        self.start_points = start_points

    def reset_end_points(self):
        end_points = np.zeros((self.num_agents, 3), dtype=np.float32)
        for i in range(self.num_agents):
            random_end = np.random.randint(
                low=1, high=self.area_size-1, size=(1, 3))
            random_end[0, 2] = min(6, random_end[0, 2])
            dist = min(np.amin(np.linalg.norm(
                    self.obstacle_points - random_end[0, :2], axis=1)),
                    np.linalg.norm(
                        self.start_points[i, :2] - random_end[0, :2]) / 5)
            while dist < 1:
                random_end = np.random.randint(
                    low=1, high=self.area_size-1, size=(1, 3))
                random_end[0, 2] = min(6, random_end[0, 2])
                dist = min(np.amin(np.linalg.norm(
                    self.obstacle_points - random_end[0, :2], axis=1)),
                    np.linalg.norm(
                        self.start_points[i, :2] - random_end[0, :2]) / 5)
            end_points[i] = random_end
        self.end_points = end_points

    def reset_reference_paths(self):
        max_time = 0
        reference_paths = []
        o = np.concatenate(
            [self.obstacle_points, self.broadline_points], axis=0)
        dijkstra = Dijkstra(o[:, 0], o[:, 1], 1.0, 0.5)
        if self.show:
            plt.ion()
            fig = plt.figure()
        for i in range(self.num_agents):
            sx, sy, sz = self.start_points[i]
            gx, gy, gz = self.end_points[i]
            rx, ry = dijkstra.planning(sx, sy, gx, gy)
            rx, ry = np.reshape(rx[::-1], (-1, 1)), np.reshape(ry[::-1], (-1, 1))
            rz = np.reshape(np.linspace(sz, gz, rx.shape[0]), (-1, 1))
            path = np.concatenate([rx, ry, rz], axis=1)
            reference_paths.append(path)
            max_time = max(max_time, rx.shape[0])
            if self.show:
                plt.clf()
                plt.scatter(o[:, 0], o[:, 1], color='red', alpha=0.2)
                plt.scatter(sx, sy, color='orange', s=100)
                plt.scatter(gx, gy, color='darkred', s=100)
                plt.scatter(rx, ry, color='grey')
                fig.canvas.draw()
                time.sleep(1)
         
        waypoints = []
        for i in range(self.num_agents):
            path = reference_paths[i]
            path_extend = np.zeros(shape=(max_time, 3), dtype=np.float32)
            path_extend[:path.shape[0]] = path
            path_extend[path.shape[0]:] = path[-1]
            waypoints.append(path_extend[np.newaxis])
        waypoints = np.concatenate(waypoints, axis=0)
        self.waypoints = np.transpose(waypoints, [1, 0, 2])
        self.max_time = max_time
        self.steps = min(self.max_steps, max_time)


class Maze(Cityscape):

    def __init__(self, num_agents=64, max_steps=12, show=False):

        scale = np.sqrt(max(1.0, num_agents / 64.0))
        area_size = 20 * scale

        Cityscape.__init__(self, num_agents, area_size, max_steps, show)
        self.OBSTACLES = np.array([
            [3, 3, 6, 4], [3, 4, 4, 16], [3, 17, 6, 16],
            [8, 7, 12, 8], [8, 13, 12, 12],
            [11, 3, 12, 8], [11, 17, 12, 12],
            [10, 3, 17, 4], [10, 17, 17, 16],
            [14, 9, 16, 11], [16, 7, 17, 13]
        ]) * scale
        self.external_traj = False

    def write(self, data_path, num_episodes):
        start_points = []
        end_points = []
        for _ in range(num_episodes):
            self.reset()
            start_points.append(self.start_points[np.newaxis])
            end_points.append(self.end_points[np.newaxis])
        write_dict = {}
        write_dict['start_points'] = np.concatenate(start_points, axis=0)
        write_dict['end_points'] = np.concatenate(end_points, axis=0)
        obstacles = np.array(self.OBSTACLES, dtype=np.float32)
        obstacles = np.concatenate(
            [obstacles, 6 * np.ones_like(obstacles[:, :1])], axis=1)
        write_dict['obstacles'] = obstacles
        pickle.dump(write_dict, open(data_path, 'wb'))

    def write_trajectory(self, data_path, traj):
        write_dict = {}
        write_dict['trajectory'] = traj
        write_dict['start_points'] = np.array([t[0, :, :3] for t in traj])
        write_dict['end_points'] = np.array([t[-1, :, :3] for t in traj])
        obstacles = np.array(self.OBSTACLES, dtype=np.float32)
        obstacles = np.concatenate(
            [obstacles, 6 * np.ones_like(obstacles[:, :1])], axis=1)
        write_dict['obstacles'] = obstacles
        pickle.dump(write_dict, open(data_path, 'wb'))

    def read(self, traj_dir):
        traj_files = os.listdir(traj_dir)
        self.episodes = []
        for traj_file in traj_files:
            self.episodes.append(
                load(open(os.path.join(traj_dir, traj_file)), Loader=Loader))
        self.external_traj = True

    def reset(self):
        if self.external_traj is False:
            super().reset()
            return
        self.init_obstacles()
        self.init_broadlines()
        episode = np.random.choice(self.episodes)
        agents = list(episode.keys())[:self.num_agents]
        self.start_points = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.end_points = np.zeros((self.num_agents, 3), dtype=np.float32)
        max_time = 0
        for i, agent in enumerate(agents):
            episode_agent = np.array(episode[agent])
            self.start_points[i] = episode_agent[0, 1:]
            self.end_points[i] = episode_agent[-1, 1:]
            max_time = max(max_time, episode_agent[-1, 0])
        max_time = int(np.rint(max_time))

        waypoints = []
        for agent in agents:
            path = np.array(episode[agent])
            ts, path = path[:, 0], path[:, 1:]
            ts_extend = np.arange(max_time)

            xs = np.interp(ts_extend, ts, path[:, 0])[:, np.newaxis]
            ys = np.interp(ts_extend, ts, path[:, 1])[:, np.newaxis]
            zs = np.interp(ts_extend, ts, path[:, 2])[:, np.newaxis]

            path_extend = np.concatenate([xs, ys, zs], axis=1)
            waypoints.append(path_extend[np.newaxis])
        waypoints = np.concatenate(waypoints, axis=0)
        self.waypoints = np.transpose(waypoints, [1, 0, 2])
        self.steps = min(self.max_steps, max_time)


def quadrotor_dynamics_np(s, u):
    dsdt = s.dot(np.array(config.A_MAT).T) + u.dot(np.array(config.B_MAT).T)
    return dsdt


def quadrotor_dynamics_tf(s, u):
    A = tf.constant(np.array(config.A_MAT).T, dtype=tf.float32)
    B = tf.constant(np.array(config.B_MAT).T, dtype=tf.float32)
    dsdt = tf.matmul(s, A) + tf.matmul(u, B)
    return dsdt


def quadrotor_controller_np(s, s_ref):
    u = (s_ref - s).dot(np.array(config.K_MAT).T)
    return u


def quadrotor_controller_tf(s, s_ref):
    K = tf.constant(np.array(config.K_MAT).T, dtype=tf.float32)
    u = tf.matmul(s_ref - s, K)
    return u


def network_cbf(x, r, indices=None):
    """ Control barrier function as a neural network.
    Args:
        x (N, N, 8): The state difference of N agents.
        r (float): The radius of the dangerous zone.
        indices (N, K): The indices of K nearest agents of each agent.
    Returns:
        h (N, K, 1): The CBF of N agents with K neighbouring agents.
        mask (N, K, 1): The mask of agents within the observation radius.
        indices (N, K): The indices of K nearest agents of each agent.
    """
    d_norm = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :3]) + 1e-4, axis=2))
    x = tf.concat([x,
                   tf.expand_dims(tf.eye(tf.shape(x)[0]), 2),
                   tf.expand_dims(d_norm - r, 2)], axis=2)
    x, indices = remove_distant_agents(x=x, k=config.TOP_K, indices=indices)
    dist = tf.sqrt(
        tf.reduce_sum(tf.square(x[:, :, :2]) + 1e-4, axis=2, keepdims=True))
    mask = tf.cast(tf.less_equal(dist, config.OBS_RADIUS), tf.float32)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_1',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=128,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_2',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_3',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=1,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='cbf/conv_4',
                                 activation_fn=None)
    h = x * mask
    return h, mask, indices


def network_action(s, s_ref, obs_radius=1.0, indices=None):
    """ Controller as a neural network.
    Args:
        s (N, 8): The current state of N agents.
        s_ref (N, 8): The reference location, velocity and acceleration.
        obs_radius (float): The observation radius.
        indices (N, K): The indices of K nearest agents of each agent.
    Returns:
        u (N, 2): The control action.
    """
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    x = tf.concat([x,
                   tf.expand_dims(tf.eye(tf.shape(x)[0]), 2)], axis=2)
    x, _ = remove_distant_agents(x=x, k=config.TOP_K, indices=indices)
    dist = tf.norm(x[:, :, :3], axis=2, keepdims=True)
    mask = tf.cast(tf.less(dist, obs_radius), tf.float32)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=64,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='action/conv_1',
                                 activation_fn=tf.nn.relu)
    x = tf.contrib.layers.conv1d(inputs=x,
                                 num_outputs=128,
                                 kernel_size=1,
                                 reuse=tf.AUTO_REUSE,
                                 scope='action/conv_2',
                                 activation_fn=tf.nn.relu)
    x = tf.reduce_max(x * mask, axis=1)
    x = tf.concat([x, s - s_ref], axis=1)
    x = tf.contrib.layers.fully_connected(inputs=x,
                                          num_outputs=64,
                                          reuse=tf.AUTO_REUSE,
                                          scope='action/fc_1',
                                          activation_fn=tf.nn.relu)
    x = tf.contrib.layers.fully_connected(inputs=x,
                                          num_outputs=128,
                                          reuse=tf.AUTO_REUSE,
                                          scope='action/fc_2',
                                          activation_fn=tf.nn.relu)
    x = tf.contrib.layers.fully_connected(inputs=x,
                                          num_outputs=64,
                                          reuse=tf.AUTO_REUSE,
                                          scope='action/fc_3',
                                          activation_fn=tf.nn.relu)
    x = tf.contrib.layers.fully_connected(inputs=x,
                                          num_outputs=3,
                                          reuse=tf.AUTO_REUSE,
                                          scope='action/fc_4',
                                          activation_fn=None)
    u_ref = quadrotor_controller_tf(s, s_ref)
    u = x + u_ref
    return u


def loss_barrier(h, s, indices=None, eps=[5e-2, 1e-3]):
    """ Build the loss function for the control barrier functions.
    Args:
        h (N, N, 1): The control barrier function.
        s (N, 8): The current state of N agents.
        indices (N, K): The indices of K nearest agents of each agent.
        eps (2, ): The margin factors.
    Returns:
        loss_dang (float): The barrier loss for dangerous states.
        loss_safe (float): The barrier loss for safe sates.
        acc_dang (float): The accuracy of h(dangerous states) <= 0.
        acc_safe (float): The accuracy of h(safe states) >= 0.
    """
    h_reshape = tf.reshape(h, [-1])
    dang_mask = compute_dangerous_mask(
        s, r=config.DIST_MIN_THRES, indices=indices)
    dang_mask_reshape = tf.reshape(dang_mask, [-1])
    safe_mask = compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    safe_mask_reshape = tf.reshape(safe_mask, [-1])

    dang_h = tf.boolean_mask(h_reshape, dang_mask_reshape)
    safe_h = tf.boolean_mask(h_reshape, safe_mask_reshape)

    num_dang = tf.cast(tf.shape(dang_h)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_h)[0], tf.float32)

    loss_dang = tf.reduce_sum(
        tf.math.maximum(dang_h + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe = tf.reduce_sum(
        tf.math.maximum(-safe_h + eps[1], 0)) / (1e-5 + num_safe)

    acc_dang = tf.reduce_sum(tf.cast(
        tf.less_equal(dang_h, 0), tf.float32)) / (1e-5 + num_dang)
    acc_safe = tf.reduce_sum(tf.cast(
        tf.greater_equal(safe_h, 0), tf.float32)) / (1e-5 + num_safe)

    acc_dang = tf.cond(
        tf.greater(num_dang, 0), lambda: acc_dang, lambda: -tf.constant(1.0))
    acc_safe = tf.cond(
        tf.greater(num_safe, 0), lambda: acc_safe, lambda: -tf.constant(1.0))

    return loss_dang, loss_safe, acc_dang, acc_safe


def loss_derivatives(s, u, h, x, indices=None, eps=[8e-2, 0, 3e-2]):
    """ Build the loss function for the derivatives of the CBF.
    Args:
        s (N, 8): The current state of N agents.
        u (N, 2): The control action.
        h (N, N, 1): The control barrier function.
        x (N, N, 8): The state difference of N agents.
        indices (N, K): The indices of K nearest agents of each agent.
        eps (3, ): The margin factors.
    Returns:
        loss_dang_deriv (float): The derivative loss of dangerous states.
        loss_safe_deriv (float): The derivative loss of safe states.
        loss_medium_deriv (float): The derivative loss of medium states.
        acc_dang_deriv (float): The derivative accuracy of dangerous states.
        acc_safe_deriv (float): The derivative accuracy of safe states.
        acc_medium_deriv (float): The derivative accuracy of medium states.
    """
    dsdt = quadrotor_dynamics_tf(s, u)
    s_next = s + dsdt * config.TIME_STEP

    x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
    h_next, mask_next, _ = network_cbf(
        x=x_next, r=config.DIST_MIN_THRES, indices=indices)

    deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h

    deriv_reshape = tf.reshape(deriv, [-1])
    dang_mask = compute_dangerous_mask(
        s, r=config.DIST_MIN_THRES, indices=indices)
    dang_mask_reshape = tf.reshape(dang_mask, [-1])
    safe_mask = compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    safe_mask_reshape = tf.reshape(safe_mask, [-1])
    medium_mask_reshape = tf.logical_not(
        tf.logical_or(dang_mask_reshape, safe_mask_reshape))

    dang_deriv = tf.boolean_mask(deriv_reshape, dang_mask_reshape)
    safe_deriv = tf.boolean_mask(deriv_reshape, safe_mask_reshape)
    medium_deriv = tf.boolean_mask(deriv_reshape, medium_mask_reshape)

    num_dang = tf.cast(tf.shape(dang_deriv)[0], tf.float32)
    num_safe = tf.cast(tf.shape(safe_deriv)[0], tf.float32)
    num_medium = tf.cast(tf.shape(medium_deriv)[0], tf.float32)

    loss_dang_deriv = tf.reduce_sum(
        tf.math.maximum(-dang_deriv + eps[0], 0)) / (1e-5 + num_dang)
    loss_safe_deriv = tf.reduce_sum(
        tf.math.maximum(-safe_deriv + eps[1], 0)) / (1e-5 + num_safe)
    loss_medium_deriv = tf.reduce_sum(
        tf.math.maximum(-medium_deriv + eps[2], 0)) / (1e-5 + num_medium)

    acc_dang_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(dang_deriv, 0), tf.float32)) / (1e-5 + num_dang)
    acc_safe_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(safe_deriv, 0), tf.float32)) / (1e-5 + num_safe)
    acc_medium_deriv = tf.reduce_sum(tf.cast(
        tf.greater_equal(medium_deriv, 0), tf.float32)) / (1e-5 + num_medium)

    acc_dang_deriv = tf.cond(
        tf.greater(num_dang, 0), lambda: acc_dang_deriv, lambda: -tf.constant(1.0))
    acc_safe_deriv = tf.cond(
        tf.greater(num_safe, 0), lambda: acc_safe_deriv, lambda: -tf.constant(1.0))
    acc_medium_deriv = tf.cond(
        tf.greater(num_medium, 0), lambda: acc_medium_deriv, lambda: -tf.constant(1.0))

    return (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv,
            acc_dang_deriv, acc_safe_deriv, acc_medium_deriv)


def loss_actions(s, u, s_ref, indices):
    """ Build the loss function for control actions.
    Args:
        s (N, 8): The current state of N agents.
        u (N, 2): The control action.
        z_ref (N, 6): The reference trajectory.
        indices (N, K): The indices of K nearest agents of each agent.
    Returns:
        loss (float): The loss function for control actions.
    """
    u_ref = quadrotor_controller_tf(s, s_ref)
    loss = tf.minimum(tf.abs(u - u_ref), (u - u_ref)**2)
    safe_mask = compute_safe_mask(s, config.DIST_SAFE, indices)
    safe_mask = tf.reduce_mean(tf.cast(safe_mask, tf.float32), axis=1)
    safe_mask = tf.cast(tf.equal(safe_mask, 1), tf.float32)
    loss = tf.reduce_sum(loss * safe_mask) / (1e-4 + tf.reduce_sum(safe_mask))
    return loss


def compute_dangerous_mask(s, r, indices=None):
    """ Identify the agents within the dangerous radius.
    Args:
        s (N, 8): The current state of N agents.
        r (float): The dangerous radius.
        indices (N, K): The indices of K nearest agents of each agent.
    Returns:
        mask (N, K): 1 for agents inside the dangerous radius and 0 otherwise.
    """
    s_diff = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    s_diff = tf.concat(
        [s_diff, tf.expand_dims(tf.eye(tf.shape(s)[0]), 2)], axis=2)
    s_diff, _ = remove_distant_agents(s_diff, config.TOP_K, indices)
    z_diff, eye = s_diff[:, :, :3], s_diff[:, :, -1:]
    z_diff = tf.norm(z_diff, axis=2, keepdims=True)
    mask = tf.logical_and(tf.less(z_diff, r), tf.equal(eye, 0))
    return mask


def compute_safe_mask(s, r, indices=None):
    """ Identify the agents outside the safe radius.
    Args:
        s (N, 8): The current state of N agents.
        r (float): The safe radius.
        indices (N, K): The indices of K nearest agents of each agent.
    Returns:
        mask (N, K): 1 for agents outside the safe radius and 0 otherwise.
    """
    s_diff = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    s_diff = tf.concat(
        [s_diff, tf.expand_dims(tf.eye(tf.shape(s)[0]), 2)], axis=2)
    s_diff, _ = remove_distant_agents(s_diff, config.TOP_K, indices)
    z_diff, eye = s_diff[:, :, :3], s_diff[:, :, -1:]
    z_diff = tf.norm(z_diff, axis=2, keepdims=True)
    mask = tf.logical_or(tf.greater(z_diff, r), tf.equal(eye, 1))
    return mask


def dangerous_mask_np(s, r):
    """ Identify the agents within the dangerous radius.
    Args:
        s (N, 8): The current state of N agents.
        r (float): The dangerous radius.
    Returns:
        mask (N, N): 1 for agents inside the dangerous radius and 0 otherwise.
    """
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    s_diff = np.linalg.norm(s_diff[:, :, :3], axis=2, keepdims=False)
    eye = np.eye(s_diff.shape[0])
    mask = np.logical_and(s_diff < r, eye == 0)
    return mask


def remove_distant_agents(x, k, indices=None):
    """ Remove the distant agents.
    Args:
        x (N, N, C): The state difference of N agents.
        k (int): The K nearest agents to keep.
    Returns:
        x (N, K, C): The K nearest agents.
        indices (N, K): The indices of K nearest agents of each agent.
    """
    n, _, c = x.get_shape().as_list()
    if n <= k:
        return x, False
    d_norm = tf.sqrt(tf.reduce_sum(tf.square(x[:, :, :3]) + 1e-6, axis=2))
    if indices is not None:
        x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
        return x, indices
    _, indices = tf.nn.top_k(-d_norm, k=k)
    row_indices = tf.expand_dims(
        tf.range(tf.shape(indices)[0]), 1) * tf.ones_like(indices)
    row_indices = tf.reshape(row_indices, [-1, 1])
    column_indices = tf.reshape(indices, [-1, 1])
    indices = tf.concat([row_indices, column_indices], axis=1)
    x = tf.reshape(tf.gather_nd(x, indices), [n, k, c])
    return x, indices

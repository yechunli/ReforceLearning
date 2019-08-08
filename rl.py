import gym
env = gym.make("CartPole-v0")
import tensorflow as tf
import numpy as np
n_input = 4
n_hidden = 4
n_output = 1
class NN():
    def __init__(self):
        self.x = tf.placeholder(shape=[1,4], dtype=tf.float32)
        self.gradient_placeholders = []
        self.model()
    def initial(self):
        weight1 = tf.Variable(tf.truncated_normal(shape=[4,4], dtype=tf.float32, stddev=0.1), name='w1')
        bias1 = tf.Variable(tf.zeros(shape=[4], dtype=tf.float32), name='b1')
        weight2 = tf.Variable(tf.truncated_normal(shape=[4,1], dtype=tf.float32, stddev=0.1), name='w1')
        bias2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32), name='b1')
        return weight1, bias1, weight2, bias2
    def model(self):
        gradient_placeholders = []
        grads_and_vars_feed = []
        w1, b1, w2, b2 = self.initial()
        hidden = tf.nn.relu(tf.add(tf.matmul(self.x, w1), b1))
        logit = (tf.add(tf.matmul(hidden, w2), b2))
        #self.output = output = tf.nn.softmax(logit)
        outputs = tf.nn.sigmoid(logit)
        output = tf.concat(axis=1, values=[outputs, 1-outputs])
        action = tf.multinomial(tf.log(output), num_samples=1)
        self.action = action
        
        label = 1. - tf.to_float(action)
        #label = [y, 1-y]
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
        optimizer = tf.train.AdamOptimizer(0.01)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        gradients = [grad for grad, variable in grads_and_vars]
        self.gradients = gradients
        for grad, variable in grads_and_vars:
            self.gradient_placeholder = tf.placeholder(shape=grad.get_shape(),dtype=tf.float32)
            self.gradient_placeholders.append(self.gradient_placeholder)
            gradient_placeholders.append(self.gradient_placeholder)
            grads_and_vars_feed.append((self.gradient_placeholder, variable))
        self.train_op = optimizer.apply_gradients(grads_and_vars_feed)
        
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards
    
def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
    
n_iterations = 250
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_rate =0.95
network = NN()
feed_dict1 = {}
save_path = "F:\python_project\model\RL"
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([network.action, network.gradients], 
                                                    feed_dict={network.x:obs.reshape(1, 4)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        for var_index, gradient_placeholder in enumerate(network.gradient_placeholders):
            mean_gradients = np.mean([reward*all_gradients[game_index][step][var_index] 
                                     for game_index, rewards in enumerate(all_rewards) 
                                     for step,reward in enumerate(rewards)], axis=0)
            feed_dict1[gradient_placeholder] = mean_gradients
        sess.run(network.train_op, feed_dict=feed_dict1)
        if iteration % save_iterations == 0:
            saver.save(sess, save_path)
            print(len(current_rewards))

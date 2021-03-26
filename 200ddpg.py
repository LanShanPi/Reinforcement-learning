
# dense(
#     inputs,
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer=None,
#     bias_initializer=tf.zeros_initializer(),
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     trainable=True,
#     name=None,
#     reuse=None
# )

# inputs: 输入数据，2维tensor.
# units: 该层的神经单元结点数。
# activation: 激活函数.
# use_bias: Boolean型，是否使用偏置项.
# kernel_initializer: 卷积核的初始化器.
# bias_initializer: 偏置项的初始化器，默认初始化为0.
# kernel_regularizer: 卷积核化的正则化，可选.
# bias_regularizer: 偏置项的正则化，可选.
# activity_regularizer: 输出的正则化函数.
# trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
# name: 层的名字.
# reuse: Boolean型, 是否重复使用参数.

# get_variable(
#     name,
#     shape=None,
#     dtype=None,
#     initializer=None,
#     regularizer=None,
#     trainable=True,
#     collections=None,
#     caching_device=None,
#     partitioner=None,
#     validate_shape=True,
#     use_resource=None,
#     custom_getter=None
# )


# name：新变量或现有变量的名称.
# shape：新变量或现有变量的形状.
# dtype：新变量或现有变量的类型(默认为 DT_FLOAT).
# initializer：创建变量的初始化器.
# regularizer：一个函数(张量 - >张量或无)；将其应用于新创建的变量的结果将被添加到集合 tf.GraphKeys.REGULARIZATION_LOSSES 中,并可用于正则化.
# trainable：如果为 True,还将变量添加到图形集合：GraphKeys.TRAINABLE_VARIABLES.
# collections：要将变量添加到其中的图形集合键的列表.默认为 [GraphKeys.LOCAL_VARIABLES].
# caching_device：可选的设备字符串或函数,描述变量应该被缓存以读取的位置.默认为变量的设备,如果不是 None,则在其他设备上进行缓存.典型的用法的在使用该变量的操作所在的设备上进行缓存,通过 Switch 和其他条件语句来复制重复数据删除.
# partitioner：(可选)可调用性,它接受要创建的变量的完全定义的 TensorShape 和 dtype,并且返回每个坐标轴的分区列表(当前只能对一个坐标轴进行分区).
# validate_shape：如果为假,则允许使用未知形状的值初始化变量.如果为真,则默认情况下,initial_value 的形状必须是已知的.
# use_resource：如果为假,则创建一个常规变量.如果为真,则创建一个实验性的 ResourceVariable,而不是具有明确定义的语义.默认为假(稍后将更改为真).
# custom_getter：可调用的,将第一个参数作为真正的 getter,并允许覆盖内部的 get_variable 方法.custom_getter 的签名应该符合这种方法,
# 但最经得起未来考验的版本将允许更改：def custom_getter(getter, *args, **kwargs).还允许直接访问所有 get_variable 参数：def custom_getter(getter, name, *args, **kwargs).
# 创建具有修改的名称的变量的简单标识自定义 getter 是：python def custom_getter(getter, name, *args, **kwargs): return getter(name + '_suffix', *args, **kwargs)




import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import gym
import time


LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32



###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params) # 优化评估策略网络

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params) # 优化评估动作网络


        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]  # 这句代码表示给self.S赋值，然后搭建并运行网络self.a
        # s[np.newaxis, :] # 在s上增加一个维度   a = np.array([1,2,3])[:,np.newaxis]：维度变为（3，1），a = np.array([1,2,3])[np.newaxis,:] ：维度变为（1，3）


    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE) # 在区间为[0，MEMORY_CAPACITY)的范围内随机挑选size个整数
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))  # 将括号里的元素合并成一个数组，无论元素是一个数还是列表，元组。。。
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0] # 由于动作是连续的，因此定义动作空间大小为1，这样有利于操作
a_bound = env.action_space.high  # 动作的限制范围
print(s_dim,a_dim,a_bound)
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
MAX_EPISODES = 2000
MAX_EP_STEPS = 200
RENDER = False
for episode in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        a = ddpg.choose_action(s)
        # print(a)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration 产生一个均值为a方差为var的数,可以为小数也可以为整数，并用clip函数将这个数限制在（-2，2）之间，若大于2，则该数为2，小于-2则该数为-2
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= 0.9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
    if episode % 100 == 0:
      total_reward = 0
      for i in range(10):
        state = env.reset()
        for j in range(MAX_EP_STEPS):
          env.render()
          action = ddpg.choose_action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/300
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
print('Running time: ', time.time() - t1)
'''
训练minist的神经网络，识别率达99.0%以上
'''
import tensorflow as tf
import input_data
#filename_t_image  = "E:\\study\\DL\\Set\\train-images.idx3-ubyte"
#filename_t_label  = "E:\\study\\DL\\Set\\train-labels.idx1-ubyte"
#filename_test_image  = "E:\\study\\DL\\Set\\t10k-images.idx3-ubyte"
#filename_test_label  = "E:\\study\\DL\\Set\\t10k-labels.idx1-ubyte"

#权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#偏置初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#卷积 1步长（stride size）,0边距(padding size)
#padding = 'SAME'要对边界进行0填充
#padding = 'VALID'不考虑
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化
#ksize 池化窗口的大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

t = input_data.DP()
t.read_train_image(input_data.filename_t_image)
t.read_train_lable(input_data.filename_t_label)
t.read_test_image(input_data.filename_test_image)
t.read_test_lable(input_data.filename_test_label)

#能够在运行图的时候，插入一些计算图
sess = tf.InteractiveSession()
# sess = tf.Session() 需要在启动session之前构建整个计算图，然后启动该计算图

'''
占位符
'''
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])#占位符，其具体值由某一次具体计算决定


'''
第一层卷积
'''
#-1表示缺省
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
第二层卷积
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
密集连接层
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
DROUP OUT
'''
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
输出层
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
'''
类别预测和损失函数
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))#损失函数为交叉熵


'''
训练模型&模型评估
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch_xs,batch_ys = t.next_batch_image()
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 0.5})
test_xs, test_ys = t.test_image_label()
print("test accuracy %g"%accuracy.eval(feed_dict={x:test_xs, y_:test_ys, keep_prob: 1.0}))

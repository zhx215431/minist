import tensorflow as tf
import input_data


t = input_data.DP()
t.read_train_image(input_data.filename_t_image)
t.read_train_lable(input_data.filename_t_label)
t.read_test_image(input_data.filename_test_image)
t.read_test_lable(input_data.filename_test_label)

x = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#0.01为学习速率

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range (1000):
    batch_xs, batch_ys = t.next_batch_image()
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
test_xs,test_ys = t.test_image_label()
print(sess.run(accuracy, feed_dict={x: test_xs, y_:test_ys}))

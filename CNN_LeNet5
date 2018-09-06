#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist =input_data.read_data_sets("./MNIST_data/", one_hot=True)

#Parameters
learning_rate = 0.01
training_iters = 10001
batch_size = 100
display_step = 1
#Network Parameters
width_input=28
height_input=28
# n_input = 784  #MNIST data input(image shape = [28,28])
n_classes = 10  #MNIST total classes (0-9digits)
dropout = 0.75  # probability to keep units

#tf Graph input
x = tf.placeholder(tf.float32,[None,width_input*height_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)  #drop(keep probability)

#Create model
# def conv2d(X_image,w,s,b):
#     return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input=X_image,filter=w,strides=s,padding='SAME'),b))
# 
def max_pooling(image,k):
    return tf.nn.avg_pool(image, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

strides = {
    'sc1':[1,1,1,1],
    'sc2':[1,1,1,1]
    #strides=[b,h,w,c]:
    #b表示在样本上的步长默认为1，也就是每一个样本都会进行运算。
    #h表示在高度上的默认移动步长为1，这个可以自己设定，根据网络的结构合理调节。
    #w表示在宽度上的默认移动步长为1，这个同上可以自己设定。
    #c表示在通道上的默认移动步长为1，这个表示每一个通道都会进行运算。
    }

weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,6])),
    'wc2':tf.Variable(tf.random_normal([5,5,6,16])),
    'wd1':tf.Variable(tf.random_normal([5*5*16,120])),
    'wd2':tf.Variable(tf.random_normal([120,84])),
    'out':tf.Variable(tf.random_normal([84,n_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([6])),
    'bc2':tf.Variable(tf.random_normal([16])),
    'bd1':tf.Variable(tf.random_normal([120])),
    'bd2':tf.Variable(tf.random_normal([84])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def conv_net(_X,_width_input,_height_input,_weights,_strides,_biases,_dropout):
    
    #将输入1维向量形式转为2维向量形式
    lay1_input = tf.reshape(_X,[-1,_width_input,_height_input,1])
    
    
    #Layer 1
    lay1_input=lay1_input
    lay1_conv=tf.nn.conv2d(input=lay1_input,filter=_weights['wc1'],strides=_strides['sc1'],padding='SAME')
    lay1_actv=tf.nn.sigmoid(tf.nn.bias_add(lay1_conv,_biases['bc1']))
    lay1_pool=max_pooling(lay1_actv, k=2)
    #lay1_drop=tf.nn.dropout(lay1_pool,keep_prob=_dropout)
    lay1_output=lay1_pool
    
    
    #Layer 2
    lay2_input=lay1_output
    lay2_conv=tf.nn.conv2d(input=lay2_input,filter=_weights['wc2'],strides=_strides['sc2'],padding='VALID')
    lay2_actv=tf.nn.sigmoid(tf.nn.bias_add(lay2_conv,_biases['bc2']))
    lay2_pool=max_pooling(lay2_actv, k=2)
    #lay2_drop=tf.nn.dropout(lay2_pool,keep_prob=_dropout) 
    lay2_output=lay2_pool   
    
    
    #将2维向量形式转为1维向量形式
    lay3_input = tf.reshape(lay2_output,[-1,_weights['wd1'].get_shape().as_list()[0]])
    
    #lay3: Fully Connected
    lay3_input=lay3_input
    lay3_fccd=tf.matmul(lay3_input,_weights['wd1'])
    lay3_actv=tf.nn.sigmoid(tf.add(lay3_fccd,_biases['bd1']))
    #lay3_drop=tf.nn.dropout(lay3_actv,_dropout)
    lay3_output=lay3_actv
    
    #lay4: Fully Connected
    lay4_input=lay3_output
    lay4_fccd=tf.matmul(lay4_input,_weights['wd2'])
    lay4_actv=tf.nn.sigmoid(tf.add(lay4_fccd,_biases['bd2']))
    #lay4_drop=tf.nn.dropout(lay4_actv,_dropout)
    lay4_output=lay4_actv
    
    #lay5: output
    lay5_input=lay4_output
    lay5_output = tf.add(tf.matmul(lay5_input,_weights['out']),_biases['out'])

    
    print(lay5_output)
    return lay5_output

#Construct model
pred = conv_net(x,width_input,height_input, weights, strides,biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size<training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict = {x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step %display_step==0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))

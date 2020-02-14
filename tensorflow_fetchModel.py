import tensorflow as tf
saver=tf.train.Saver()
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'./Models_par/-200')
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc=accuracy.eval(feed_dict={x:test_images[0:100],y_:one_hot_test[:100],keep_prob:1.0})
    print(acc)
    
    
    
  #铭记 变量名要与之前保存的一致
  
  
import tensorflow as tf
reader = tf.train.NewCheckpointReader('./model.ckpt')
dic = reader.get_variable_to_shape_map()
print(dic)
#查看有哪些变量名  以及具体的值

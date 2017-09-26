import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from deformable_convolution import *
from img_utils import imsave
import ops

# 搭建U-net
class UNet(object):
    
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        
        # 记录在第几层使用DCN
        if self.conf.add_dcn == True:
            self.insertdcn = self.conf.dcn_location
        else:
            self.insertdcn = -1
         
        # 设置一些需要的参数
        self.data_format = 'NHWC'
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [conf.batch, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]
        
        # 设置一些保存模型需要的文件夹
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
            
        # 配置网络
        self.configure_networks()
        
        # 用于记录summary
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    # 配置网络的基本参数
    def configure_networks(self):
        
        # 搭建网络
        self.build_network()
        
        # 选择优化器和学习率
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        
        # 用于保存模型和summary
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    # 搭建网络
    def build_network(self):
        
        # 设置X\Y的容器
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(tf.int64, self.output_shape, name='annotations')
        expand_annotations = tf.expand_dims(self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(expand_annotations, axis=[self.channel_axis],name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        
        # 计算预测出来的Y
        self.predictions = self.inference(self.inputs)
        
        # 选择cross_entropy损失函数
        losses = tf.losses.softmax_cross_entropy(one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        
        # 计算两个评价指标
        self.decoded_predictions = tf.argmax(self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(self.annotations, self.decoded_predictions, name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        weights = tf.cast(tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(self.annotations, self.decoded_predictions, self.conf.class_num,
            weights, name='m_iou/m_ious')

    # 用来配置保存summary
    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.image(name+'/input', self.inputs, max_outputs=100))
        summarys.append(tf.summary.image(name + '/annotation', tf.cast(tf.expand_dims(
                self.annotations, -1), tf.float32), max_outputs=100))
        summarys.append(tf.summary.image(name + '/prediction', tf.cast(tf.expand_dims(
                self.decoded_predictions, -1), tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    # 用于预测Y，搭建真正的网络结构
    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        
        # 搭建下采样down网络
        for layer_index in range(self.conf.network_depth-1):
            
            # 记录是否是第一层
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
           
            if layer_index == self.insertdcn:
                outputs = self.construct_down_block(outputs, name, down_outputs, first=is_first, DCN=True)
            else:
                outputs = self.construct_down_block(outputs, name, down_outputs, first=is_first, DCN=False) 
            print("down ",layer_index," shape ", outputs.get_shape()) 
            
        # 搭建下采样层顶层
        outputs = self.construct_bottom_block(outputs, 'bottom')
        print("bottom shape",outputs.get_shape())
        
        # 搭建上采样up网络
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            
            # 记录是否是最后一层
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_up_block(outputs, down_inputs, name, final=is_final)
            print("up ",layer_index," shape ",outputs.get_shape())
        return outputs

    # 下采样层
    def construct_down_block(self, inputs, name, down_outputs, first=False, DCN=False):
        
        # 计算本层需要输出的filters深度数目
        num_outputs = self.conf.start_channel_num if first else 2*inputs.shape[self.channel_axis].value
            
        if DCN == True:
            conv1 = ops.deform_conv2d(inputs, num_outputs, self.conv_size, name+'/deformconv1')
            conv2 = ops.deform_conv2d(conv1, num_outputs, self.conv_size, name+'/deformconv2')
        else:
            conv1 = self.down_conv_func()(inputs, num_outputs, self.conv_size, name+'/conv1')
            conv2 = self.down_conv_func()(conv1, num_outputs, self.conv_size, name+'/conv2')
            
        down_outputs.append(conv2)
        pool = ops.pool2d(conv2, self.pool_size, name+'/pool')
        return pool

    # 顶层，这一层shape不变
    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = self.down_conv_func()(inputs, 2*num_outputs, self.conv_size, name+'/conv1')
        conv2 = self.down_conv_func()(conv1, num_outputs, self.conv_size, name+'/conv2')
        return conv2

    # 上采样层
    def construct_up_block(self, inputs, down_inputs, name, final = False):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(inputs, num_outputs, self.conv_size, name+'/conv1')
        conv1 = tf.concat([conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.up_conv_func()(conv1, num_outputs, self.conv_size, name+'/conv2')
       
        # 计算本层需要输出的filters深度数目
        num_outputs = self.conf.class_num if final else num_outputs/2
        conv3 = self.up_conv_func()(conv2, num_outputs, self.conv_size, name+'/conv3')
        return conv3

    # 得取自己定义的卷积和反卷积函数
    def down_conv_func(self):
        return getattr(ops, self.conf.down_conv_name)
    
    def up_conv_func(self):
        return getattr(ops, self.conf.up_conv_name)
    
    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    # 保存summary
    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    # 训练
    def train(self):
        
        # 有时可以从以训练好的model开始训练
        if self.conf.reload_epoch > 0:
            self.reload(self.conf.reload_epoch)
            
        # 读取数据
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)
        
        # 记录loss
        valid_loss_list = []
        train_loss_list = []
        
        for epoch_num in range(self.conf.max_epoch):
            if epoch_num % self.conf.test_step == 1:
                inputs, annotations = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations}
                loss, summary = self.sess.run([self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
               
                print(epoch_num, '----testing loss', loss)
                print(epoch_num)
                
                # 记录验证集上的loss
                valid_loss_list.append(loss)
                np.save(self.conf.record_dir+"valid_loss.npy",np.array(valid_loss_list))
            elif epoch_num % self.conf.summary_step == 1:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations}
                loss, _, summary = self.sess.run([self.loss_op, self.train_op, self.train_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num)
                print(epoch_num)
                
                # 记录训练集上的loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
            else:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations}
                loss,_ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
                
                print(epoch_num)
                
                # 记录训练集上的loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
        
            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)
        
    # 测试
    def test(self,model_i):
        print('---->testing ', model_i)
        
        # 加载模型
        if model_i > 0:
            self.reload(model_i)
        else:
            print("please set a reasonable test_epoch")
            return
        
        # 读取数据，注意是False，代表不是在训练
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data,False)
        self.sess.run(tf.local_variables_initializer())
       
        # 记录测试参数
        losses = []
        accuracies = []
        m_ious = []
        while True:
            inputs, annotations = valid_reader.next_batch(self.conf.batch)
           
            # 终止条件：当取出的batch不够个数了就break
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)          
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            
            # 其实是每一个batch上计算一次指标，最后求均值
            
        return np.mean(losses),np.mean(accuracies),m_ious[-1]

    # 预测
    def predict(self):
        print('---->predicting ', self.conf.test_epoch)
        
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        
        # 读取数据
        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())
        predictions = []
        losses = []
        accuracies = []
        m_ious = []
     
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            
            # 终止条件
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
            # 记录指标
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            # 记录预测值
            predictions.append(self.sess.run(self.decoded_predictions, feed_dict=feed_dict))
      
        print('----->saving predictions')
        print(np.shape(predictions))
        
        for index, prediction in enumerate(predictions):
            
            # 下面的程序用于输出一通道的预测值，测试时需要观察的
            #print(prediction.shape)
            #print(index)
            #np.save("pred",np.array(prediction))
            
            # 把一通道的预测值保存为三通道图片，这是自己写的函数
            for i in range(prediction.shape[0]):
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i)+'.png')
                
        # 验证和测试的时候，指标都是返回的全体上的均值
        return np.mean(losses),np.mean(accuracies),m_ious[-1]

    # 保存函数
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    # 用于加载模型
    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

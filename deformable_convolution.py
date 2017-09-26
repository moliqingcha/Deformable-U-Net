
# coding: utf-8


import numpy as np
import tensorflow as tf

class DCN(object):
    
    """
    初始化函数
    只有两个参数：
    input_shape：输入的input feature map的shape,是一个1*4的list
    kernel_size：卷积核的尺寸，是一个1*2的list
    """
    def __init__(self, input_shape, kernel_size):
        
        # 定义deform_conv的kernel size
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0]*kernel_size[1]
        
        # 定义input feature map的shape参数
        self.num_batch = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.num_channels = input_shape[3]
        
        self.extend_scope = 7.0
    
    
    """
    函数_coordinate_map(self, offset_field)用于生成3W*3H的coordinate map
    输入：offset field
    输出：3W*3H的coordinate map
    """
    def _coordinate_map(self, offset_field, name):
        with tf.variable_scope(name+"/_coordinate_map"):
            
            # offset矩阵
            x_offset, y_offset = tf.split(tf.reshape(offset_field, [self.num_batch, self.height, self.width, 2, self.num_points]),2,3)
            x_offset = tf.squeeze(x_offset)
            y_offset = tf.squeeze(y_offset)

            # 中心点坐标矩阵
            x_center = tf.reshape(tf.tile(tf.range(self.width),[self.height]),[self.height*self.width,-1])
            x_center = tf.tile(x_center,[1,self.num_points])
            x_center = tf.reshape(x_center,[self.height,self.width,self.num_points])
            x_center = tf.tile(tf.expand_dims(x_center, 0), [self.num_batch,1,1,1])

            y_center = tf.tile(tf.range(self.height),[self.width])
            y_center = tf.transpose(tf.reshape(y_center, [self.width,self.height]))
            y_center = tf.reshape(y_center,[self.height*self.width,-1])
            y_center = tf.tile(y_center,[1,self.num_points])
            y_center = tf.reshape(y_center,[self.height,self.width,self.num_points])
            y_center = tf.tile(tf.expand_dims(y_center, 0), [self.num_batch,1,1,1])
        
            x_center = tf.cast(x_center,"float32")
            y_center = tf.cast(y_center,"float32")

            # regular grid R矩阵
            x = tf.linspace(-(self.kernel_size[0]-1)/2, (self.kernel_size[0]-1)/2, self.kernel_size[0])
            y = tf.linspace(-(self.kernel_size[1]-1)/2, (self.kernel_size[1]-1)/2, self.kernel_size[1])
            x,y = tf.meshgrid(x,y)
            x_spread = tf.transpose(tf.reshape(x,(-1,1)))
            y_spread = tf.transpose(tf.reshape(y,(-1,1)))
            x_grid = tf.tile(x_spread,[1,self.height*self.width])
            x_grid = tf.reshape(x_grid, [self.height, self.width, self.num_points])
            y_grid = tf.tile(y_spread,[1,self.height*self.width])
            y_grid = tf.reshape(y_grid, [self.height, self.width, self.num_points])
            x_grid = tf.tile(tf.expand_dims(x_grid, 0), [self.num_batch,1,1,1])
            y_grid = tf.tile(tf.expand_dims(y_grid, 0), [self.num_batch,1,1,1])
            

            # 计算得到X,Y
            x = tf.add_n([x_center, x_grid, tf.multiply(self.extend_scope, x_offset)])
            y = tf.add_n([y_center, y_grid, tf.multiply(self.extend_scope, y_offset)])

            # 将N*H*W*num_points转换为N*3H*3W
            x_new = tf.reshape(x,[self.num_batch,self.height,self.width,self.kernel_size[0],self.kernel_size[1]])
            x_new = tf.squeeze(tf.split(x_new,self.height,1))
            x_new = tf.squeeze(tf.split(x_new,self.kernel_size[0],3))
            x_new = tf.reshape(tf.split(x_new,self.height,1),[self.kernel_size[0]*self.height,self.num_batch,self.kernel_size[1]*self.width])
            x_new = tf.squeeze(tf.split(x_new,self.num_batch,1))

            y_new = tf.reshape(y,[self.num_batch,self.height,self.width,self.kernel_size[0],self.kernel_size[1]])
            y_new = tf.squeeze(tf.split(y_new,self.height,1))
            y_new = tf.squeeze(tf.split(y_new,self.kernel_size[0],3))
            y_new = tf.reshape(tf.split(y_new,self.height,1),[self.kernel_size[0]*self.height,self.num_batch,self.kernel_size[1]*self.width])
            y_new = tf.squeeze(tf.split(y_new,self.num_batch,1))
          
            return x_new, y_new
    
    """
    函数_bilinear_interpolate(self, input_feature, coordinate_map)用于进行双线性插值
    输入：input feature map；coordinate map
    输出：3W*3H*C1的deformed feature map
    """
    def _bilinear_interpolate(self, input_feature, x, y, name):
        with tf.variable_scope(name+"/_bilinear_interpolate"):
        
            # 一维数据
            x = tf.reshape(x, [-1])
            y = tf.reshape(y, [-1])

            # 数据类型转换
            x = tf.cast(x, "float32")
            y = tf.cast(y, "float32")
            zero = tf.zeros([], dtype="int32")
            max_x = tf.cast(self.width-1, "int32")
            max_y = tf.cast(self.height-1, "int32")

            # 找到四个格点
            x0 = tf.cast(tf.floor(x), "int32")
            x1 = x0+1
            y0 = tf.cast(tf.floor(y), "int32")
            y1 = y0+1

            # 截断feature map以外的点
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            # 把input_feature和coordinate X和Y都转换为二维，方便从中抽取值
            input_feature_flat = tf.reshape(input_feature, tf.stack([-1, self.num_channels]))
          
            dimension_2 = self.width
            dimension_1 = self.width*self.height
            base = tf.range(self.num_batch)*dimension_1
            repeat = tf.transpose(tf.expand_dims(tf.ones(shape=(tf.stack([self.num_points*self.height*self.width,]))),1),[1,0])
            repeat = tf.cast(repeat,"int32")
            base = tf.matmul(tf.reshape(base,(-1,1)),repeat)
            base = tf.reshape(base, [-1])
            base_y0 = base+y0*dimension_2
            base_y1 = base+y1*dimension_2
            index_a = base_y0+x0
            index_b = base_y1+x0
            index_c = base_y0+x1
            index_d = base_y1+x1

            # 计算四个格点的value
            value_a = tf.gather(input_feature_flat, index_a)
            value_b = tf.gather(input_feature_flat, index_b)
            value_c = tf.gather(input_feature_flat, index_c)
            value_d = tf.gather(input_feature_flat, index_d)

            # 计算四个面积
            x0_float = tf.cast(x0, "float32")
            x1_float = tf.cast(x1, "float32")
            y0_float = tf.cast(y0, "float32")
            y1_float = tf.cast(y1, "float32")
            area_a = tf.expand_dims(((x1_float-x)*(y1_float-y)),1)
            area_b = tf.expand_dims(((x1_float-x)*(y-y0_float)),1)
            area_c = tf.expand_dims(((x-x0_float)*(y1_float-y)),1)
            area_d = tf.expand_dims(((x-x0_float)*(y-y0_float)),1)

            # 相乘相加
            outputs = tf.add_n([value_a*area_a, value_b*area_b, value_c*area_c, value_d*area_d])
            outputs = tf.reshape(outputs, [self.num_batch, self.kernel_size[0]*self.height, self.kernel_size[1]*self.width, self.num_channels])

        
            return outputs
        
       
    
    """
    函数deform_conv(self, inputs)用于进行deformable convolution操作
    输入：input feature map
    输出：output feature map
    """
    def deform_conv(self, inputs, offset, name, **kwargs):
        with tf.variable_scope(name+"/DeformedFeature"):
            x, y = self._coordinate_map(offset, name)
            deformed_feature = self._bilinear_interpolate(inputs, x, y, name)
            return deformed_feature







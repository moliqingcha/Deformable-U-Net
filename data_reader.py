
import h5py
import numpy as np

# 用来读取H5格式的数据
class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'], data_file['Y']
        self.gen_indexes()
    
    # 用于生成初始的index列表
    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0
    
    # 生成每一次训练、测试需要的batch
    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        
        # 有时候train的样本量可能不被batch整除，就需要重新初始化index列表啦
        # cur_indexes必须是严格递增的（无重复）,直接重新开始就行
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            self.cur_index = batch_size
            cur_indexes = list(self.indexes[:batch_size])
        if len(cur_indexes)==0 and not self.is_train:
            cur_indexes = [0]
        cur_indexes.sort()
        return self.images[cur_indexes], self.labels[cur_indexes]

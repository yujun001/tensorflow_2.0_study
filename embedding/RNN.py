import tensorflow as tf
import numpy as np

class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        # 建立char_indices 的map映射
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # get total index list of the raw_text
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        # 序列seq_length:40,  batch_size: 50
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            # 随机抽取seq_length 为40的序列, 循环50次
            seq.append(self.text[index:index+seq_length])
            # 每个随机序列seq对应的下一个char,50个依次存入列表list
            next_char.append(self.text[index+seq_length])
        # 维度信息 [batch_size, seq_length], [num_batch]
        return np.array(seq), np.array(next_char)


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)
        # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])

if __name__ =='__main__':

    # num_batches = 1000
    num_batches = 100
    seq_length = 40
    batch_size = 50
    learning_rate = 1e-3

    # __init__ 初始化数据获取
    data_loader = DataLoader()
    print("data_loader.chars++++++++++++", data_loader.chars)
    print("the len of data_loader", len(data_loader.chars))

    # data_loader.chars; 去重且排序后的字符串 57个
    model = RNN(num_chars=len(data_loader.chars), 
                batch_size=batch_size, 
                seq_length=seq_length)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # num_batches = 1000次
    # 每个batch: seq长度为40+next_char长度1; 共50个batch
    # 所有batch 中的seq 拼合为list;  next_char 拼合为list, 长度50 size
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(seq_length, batch_size)
        # print(X, y)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)   # 交叉熵计算损失
            loss = tf.reduce_mean(loss)                                                       # 累计损失求和
            print("batch %d: loss %f" % (batch_index, loss.numpy()))                          # batch_index, 代表多少轮训练
        grads = tape.gradient(loss, model.variables)                                          #
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))                 # 损失函数的寻优计算(梯度下降计算)

    # batches =1, 序列长度 40个;  即[1,40],[1]  数据维度统计
    X_, _ = data_loader.get_batch(seq_length, 1)
    print(X_)
    print(_)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = model.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print("\n")
"""
NeuralLog多文件日志异常检测示例

演示如何使用多文件数据加载功能处理包含多个独立日志文件的数据集。
"""
import os
import sys

sys.path.append("../")

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.nlp import optimization
from sklearn.utils import shuffle

from neurallog.models import NeuralLog
from neurallog.data_loader_multi_file import load_multi_file_logs
from neurallog.utils import classification_report

# 配置参数
data_root = "../data/logs"  # 数据根目录
embed_dim = 768  # 嵌入维度（BERT输出维度）
max_len = 75  # 序列最大长度


class BatchGenerator(Sequence):
    """
    批处理生成器
    
    用于将日志序列批量输入模型，处理变长序列的填充和截断。
    """
    def __init__(self, X, Y, batch_size, max_len=max_len, embed_dim=embed_dim):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.max_len = max_len
        self.embed_dim = embed_dim

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # 获取当前batch的数据
        batch_x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        batch_y = self.Y[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.Y))]
        
        # 初始化batch数组
        X = np.zeros((len(batch_x), self.max_len, self.embed_dim))
        Y = np.zeros((len(batch_x), 2))
        
        # 处理每个序列
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            # 转换为numpy数组
            x = np.array(x)
            
            # 如果序列长度超过max_len，截取最后max_len个元素
            if len(x) > self.max_len:
                x = x[-self.max_len:]
            
            # 如果序列长度小于max_len，在左侧填充0（保持时间顺序）
            if len(x) < self.max_len:
                pad_width = ((self.max_len - len(x), 0), (0, 0))
                x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
            
            X[i] = x
            # 将标签转换为one-hot编码
            Y[i] = [1, 0] if y == 0 else [0, 1]
        
        return X, Y[:, 0]  # 返回稀疏标签


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, 
                    batch_size, epoch_num, model_name=None):
    """
    训练模型
    
    参数:
        training_generator: 训练数据生成器
        validate_generator: 验证数据生成器
        num_train_samples: 训练样本数
        num_val_samples: 验证样本数
        batch_size: 批大小
        epoch_num: 训练轮数
        model_name: 模型保存路径
    """
    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw'
    )

    loss_object = SparseCategoricalCrossentropy()

    # 创建模型
    model = NeuralLog(embed_dim, ff_dim=2048, max_len=max_len, num_heads=12, dropout=0.1)

    model.compile(loss=loss_object, metrics=['accuracy'], optimizer=optimizer)

    print(model.summary())

    # 设置回调函数
    callbacks_list = []
    if model_name:
        checkpoint = ModelCheckpoint(
            model_name,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=True
        )
        callbacks_list.append(checkpoint)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )
    callbacks_list.append(early_stop)

    # 训练模型
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_train_samples / batch_size),
        epochs=epoch_num,
        verbose=1,
        validation_data=validate_generator,
        validation_steps=int(num_val_samples / batch_size),
        workers=16,
        max_queue_size=32,
        callbacks=callbacks_list,
        shuffle=True
    )
    return model


def train(X, Y, epoch_num, batch_size, model_file=None):
    """
    训练函数
    
    参数:
        X: 训练特征（序列列表）
        Y: 训练标签
        epoch_num: 训练轮数
        batch_size: 批大小
        model_file: 模型保存路径
    """
    X, Y = shuffle(X, Y)
    n_samples = len(X)
    
    # 90%用于训练，10%用于验证
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator = BatchGenerator(train_x, train_y, batch_size)
    validate_generator = BatchGenerator(val_x, val_y, batch_size)
    num_train_samples = len(train_x)
    num_val_samples = len(val_x)

    print(f"训练样本数: {num_train_samples} - 验证样本数: {num_val_samples}")

    model = train_generator(
        training_generator, validate_generator, 
        num_train_samples, num_val_samples, 
        batch_size, epoch_num, model_name=model_file
    )

    return model


def test_model(model, x, y, batch_size):
    """
    测试模型
    
    参数:
        model: 训练好的模型
        x: 测试特征
        y: 测试标签
        batch_size: 批大小
    """
    x, y = shuffle(x, y)
    # 确保batch大小对齐
    x = x[:len(x) // batch_size * batch_size]
    y = y[:len(y) // batch_size * batch_size]
    
    test_loader = BatchGenerator(x, y, batch_size)
    prediction = model.predict_generator(
        test_loader, 
        steps=(len(x) // batch_size), 
        workers=16, 
        max_queue_size=32,
        verbose=1
    )
    prediction = np.argmax(prediction, axis=1)
    y = y[:len(prediction)]
    report = classification_report(np.array(y), prediction)
    print(report)


if __name__ == '__main__':
    # 加载多文件日志数据
    print("开始加载数据...")
    (x_tr, y_tr), (x_te, y_te) = load_multi_file_logs(
        data_root=data_root,
        pattern="*log*.txt",  # 匹配所有包含"log"的txt文件
        train_ratio=0.8,
        windows_size=20,
        step_size=20,  # 固定窗口，不重叠
        e_type='bert',
        mode='balance',  # 对训练集进行类别平衡
        # file_train_ratio=0.8  # 可选：按文件级别分割
    )

    # 训练模型
    print("\n开始训练模型...")
    model = train(x_tr, y_tr, epoch_num=10, batch_size=256, model_file="multi_file_transformer.hdf5")
    
    # 测试模型
    print("\n开始测试模型...")
    test_model(model, x_te, y_te, batch_size=1024)


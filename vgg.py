import tensorflow as tf
import os
import matplotlib.pyplot as plt


# 初始化参数
batch_size = 256
epochs = 200  # 迭代次数
num_classes = 10  # 分类数


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def vgg_11(kn_size, lr, decay, drop):

#   opt = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)

    k_size = kn_size
    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, input_shape=(32, 32, 3), kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
#    16 * 16 * 64

    model.add(tf.keras.layers.Conv2D(128, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
#    8 * 8 * 128

    model.add(tf.keras.layers.Conv2D(256, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
#    model.add(tf.keras.layers.Dropout(drop))
#    8 * 8 * 256

    model.add(tf.keras.layers.Conv2D(256, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()
                                     ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
#    4 * 4 * 256

    model.add(tf.keras.layers.Conv2D(512, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(drop))
#    4 * 4 * 512

    model.add(tf.keras.layers.Conv2D(512, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
#    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(drop))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=k_size, activation='relu', padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(decay),
                                     kernel_initializer=tf.keras.initializers.he_normal()))
#    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
#    1 * 1 * 512

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(decay)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Dense(4096, use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(decay)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def load_data():
    # data loading
    (x_train, y_train), (x_test, y_test) =tf.keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # data preprocessing
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)


def vgg_init(lr, decay):
    opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)


def show_train_history(train_acc, test_acc):
    plt.plot(tr_history.history[train_acc])
    plt.plot(tr_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("1.png")
    plt.show()


def tfb(save_path):
    log_filepath = save_path
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)

    return tb_cb


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 数据预处理
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_train /= 255.
    x_test = x_test.astype('float32')
    x_test /= 255.

    '''    
        x_train = x_train / 255
        x_test = x_test / 255
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
        # data preprocessing
        x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
        x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
        x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
        x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
        x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
        x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)
    '''

    model = vgg_11((3, 2), lr=0.01, decay=0.000001, drop=0.3)

    tbcb = tf.keras.callbacks.TensorBoard(
               './logs_3_2_decay0.000001',
               histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
               batch_size=32,     # 用多大量的数据计算直方图
               write_graph=True,  # 是否存储网络结构图
               write_grads=True,  # 是否可视化梯度直方图
               write_images=True,  # 是否可视化参数
               embeddings_freq=0,
               embeddings_layer_names=None,
               embeddings_metadata=None)

    data_generate = tf.keras.preprocessing.image.ImageDataGenerator(
                                        featurewise_center=False,  # 将输入数据的均值设置为0
                                        samplewise_center=False,  # 将每个样本的均值设置为0
                                        featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
                                        samplewise_std_normalization=False,  # 将每个输出除以其标准差
                                        zca_epsilon=1e-6,  # ZCA白化的epsilon值，默认为1e-6
                                        zca_whitening=False,  # 是否应用ZCA白化
                                        rotation_range=0.,  # 随机旋转的度数范围，输入为整数
                                        width_shift_range=0.1,  # 左右平移，输入为浮点数，大于1时输出为像素值
                                        height_shift_range=0.1,  # 上下平移，输入为浮点数，大于1时输出为像素值
                                        shear_range=0.,  # 剪切强度，输入为浮点数
                                        zoom_range=0.,  # 随机缩放，输入为浮点数
                                        channel_shift_range=0.,  # 随机通道转换范围，输入为浮点数
                                        fill_mode='nearest',  # 输入边界以外点的填充方式，还有constant,reflect,wrap三种填充方式
                                        cval=0.,  # 用于填充的值，当fill_mode='constant'时生效
                                        horizontal_flip=True,  # 随机水平翻转
                                        vertical_flip=False,  # 随机垂直翻转
                                        rescale=None,  # 重随放因子，为None或0时不进行缩放
                                        preprocessing_function=None,  # 应用于每个输入的函数
                                        data_format=None,  # 图像数据格式，默认为channels_last
                                        validation_split=0.0)

    tr_history = model.fit_generator(data_generate.flow(x_train, y_train, batch_size), steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs, callbacks=tbcb, validation_data=(x_test, y_test))

    show_train_history('accuracy', 'val_accuracy')
    show_train_history('loss', 'val_loss')

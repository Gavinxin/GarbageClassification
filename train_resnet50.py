# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical

def acc_top5(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

# train_dir = os.environ['IMAGE_TRAIN_INPUT_PATH']
# MODEL_PATH = os.environ['MODEL_INFERENCE_PATH']
train_dir = "./wget_img/"
MODEL_PATH =""
def eachFile(filepath):  # 将目录内的文件名放入列表中
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        out.append(allDir)
    return out
dic = {'奶粉': '26', '纸箱': '75', '胶水': '79', '吹风机': '16', '塑料玩具': '20', '椅子': '53', '充电器': '7', '塑料袋': '23', '纸尿裤': '73', '牙刷': '62', '剃须刀': '11', '辣椒': '90', '土豆': '17', '瓶盖': '66', '一次性塑料手套': '1', '抹布': '38', '杏核': '49', '充电线': '10', '塑料盖子': '22', '干电池': '28', '烟盒': '61', '中性笔': '4', '旧镜子': '46', '充电宝': '8', '鼠标': '99', '水彩笔': '55', '蒜皮': '85', '旧玩偶': '45', '退热贴': '92', '废弃食用油': '30', '青椒': '96', '口服液瓶': '15', '一次性纸杯': '3', '纽扣': '76', '指甲油瓶子': '40', '插座': '42', '充电电池': '9', '塑料桶': '19', '袜子': '89', '电视机': '67', '护手霜': '35', '手表': '32', '红豆': '72', '衣架': '88', '消毒液瓶': '60', '医用棉签': '14', '扫把': '34', '海绵': '59', '塑料包装': '18', '菜刀': '81', '蛋_蛋壳': '87', '剪刀': '12', '暖宝宝贴': '47', '纸巾_卷纸_抽纸': '74', '糖果': '71', '铅笔屑': '94', '头饰': '25', '泡沫盒子': '57', '打火机': '33', '杀虫剂': '48', '毛毯': '54', '自行车': '80', '耳机': '77', '信封': '6', '酸奶盒': '93', '作业本': '5', '拖把': '39', '外卖餐盒': '24', '水龙头': '56', '旧帽子': '44', '蒜头': '84', '白糖_盐': '69', '蚊香': '86', '快递盒': '31', '胶带': '78', '菜板': '82', '抱枕': '37', '洗面奶瓶': '58', '空调机': '70', '废弃衣服': '29', '面膜': '97', '香烟': '98', '无纺布手提袋': '43', 'PET塑料瓶': '0', '姜': '27', '护肤品玻璃罐': '36', '过期化妆品': '91', '陶瓷碗碟': '95', '化妆品瓶': '13', '棉签': '52', '指甲钳': '41', '牛奶盒': '65', '牙签': '63', '塑料盆': '21', '葡萄干': '83', '果皮': '51', '牙膏皮': '64', '一次性筷子': '2', '电风扇': '68', '杯子': '50'}

def get_class_index(class_index_filename):
    dic = {}
    f =open(class_index_filename)
    for i in f:
        temp = i.split(" ")
        dic[temp[0]] = temp[1].strip('\n')
    return dic

def get_data(resize=True, data_format=None,t=''):  # 从文件夹中获取图像数据
    pic_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    #data_format = conv_utils.normalize_data_format(data_format)
    for pic_dir in pic_dir_set:
        label = int(dic[pic_dir])
        if not os.path.isdir(os.path.join(pic_dir_data, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(pic_dir_data, pic_dir))
        for pic_name in pic_set[:200]:
            if not os.path.isfile(os.path.join(pic_dir_data, pic_dir, pic_name)):
                continue
            # img = cv2.imdecode(np.fromfile(os.path.join(pic_dir_data, pic_dir, pic_name), dtype=np.uint8), -1)
            # image_raw_data = tf.gfile.FastGFile(os.path.join(pic_dir_data, pic_dir, pic_name), 'r').read()
            # img_data = tf.image.decode_jpeg(image_raw_data)

                
            img = cv2.imread(os.path.join(pic_dir_data, pic_dir, pic_name))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, :, :3]

            img = preprocess_input(np.expand_dims(cv2.resize(img, (224, 224)), 0), mode='tf')
            
            # https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L116
            # normalize
            #  x /= 127.5
            #  x -= 1.
                        
            X_train.append(img)
            y_train.append(label)
            
            # y_train.append(np.expand_dims(to_categorical(label, num_classes=100),0))
    X_train = np.concatenate(X_train, axis=0)
    # y_train = np.concatenate(y_train, axis=0)
    y_train = np.array(y_train)
    print(X_train.shape, y_train.shape)
    return (X_train, y_train)


def main():
    global Width, Height,pic_dir_data,class_index_filename
    Width = 224
    Height = 224
    num_classes = 100  # 数据集100
    pic_dir_data = train_dir
    export_path = 'SavedModel'
    #print export_path
    Width = 224
    Height = 224
    #class_index_filename = os.path.join(os.getcwd()+"class_index.txt")
    (X_train, y_train) = get_data(data_format='channels_last')
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    num_classes = 100
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))

    base_model = tf.keras.applications.ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    # x = tf.keras.layers.BatchNormalization(name='bn_fc_01')(x)
    # x = tf.keras.layers.GlobalAveragePooling2D(name='Global_average_pool')(x)
    # x = tf.keras.layers.Flatten(name='flatten')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    #end = tf.keras.layers.Zeros()
    resnet50 = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    resnet50.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', acc_top5])
    resnet50.fit(X_train, y_train, epochs=5, batch_size=15, validation_split=0.25)
    # resnet50.summary()
    #tf.keras.backend.set_learning_phase(0)
    tf.keras.experimental.export_saved_model(resnet50, export_path)


if __name__ == '__main__':
    main()
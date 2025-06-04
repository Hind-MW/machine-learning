[若需要完整数据集以及代码请点击以下链接](https://mbd.pub/o/bread/aJaVmJ9s)
# 基于CNN的多种类蝴蝶图像分类🦋  
基于卷积神经网络对6499+2786张图像，75种不同类别的蝴蝶进行可视化分析、模型训练及分类展示  
## 导入库

```python
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r"Your `PyDataset` class should call `super().__init__\(\*\*kwargs\)`")
```
## 数据分析及可视化

```python
df = pd.read_csv("/home/mw/input/btfl7333/btfl/btfl/Training_set.csv")
df.head(10)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9eac194d2c54e2386ffe11c6dafeb34.png)

```python
print("查看数据信息")
print(df.describe())
print("查看空值")
print(df.isnull().sum())
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/141c9904ae20470bb1870c10f8263465.png)
**查看各个类别包含的数据量**

```python
labelcounts = df['label'].value_counts().sort_index()
plt.figure(figsize=(14, 8))
sns.barplot(x=labelcounts.index, y=labelcounts.values, palette='viridis')
plt.title('蝴蝶类型数目详细信息')
plt.xlabel('蝴蝶类型')
plt.ylabel('类别数量')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8dc8c6850bff40558bc08da217d0f5ce.png)
**随机查看部分图片及其对应的标签**

```python
image_dir = "/home/mw/input/btfl7333/btfl/btfl/train"
sample_images = df.sample(12, random_state=43)
fig, axes = plt.subplots(4, 3, figsize=(15, 15))

for i, (index, row) in enumerate(sample_images.iterrows()):
    img_path = os.path.join(image_dir, row['filename'])
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  
    
    ax = axes[i // 3, i % 3]
    ax.imshow(img_array)
    ax.set_title(f"类别: {row['label']}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b375a5e56b0d475ba00b3bc28047e8a3.png)
## 数据预处理  
为图像分类任务准备训练和验证数据  
使用train_test_split将数据集按照80%的比例划分为训练集 (train_df) 和验证集 (val_df)。  
创建训练集的数据生成器，对训练数据进行数据增强，同时将标签转换为独热编码形式  
创建验证集的数据生成器，对测试数据进行像素归一化

```python
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255, # 将像素值归一化到 [0, 1] 范围
    rotation_range=40, # 随机旋转图片，范围为0到40度
    width_shift_range=0.2, # 随机水平和垂直平移图片，范围为20%
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2, # 随机缩放图片
    horizontal_flip=True,
    fill_mode='nearest' # 在变换时填充空白区域，使用最近邻插值
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical' # 将标签转换为独热编码形式
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1aa2cfa42d3e4374832ad724e0c124b6.png)
## 展示部分处理后的数据  
上一步已经对标签进行了编码

```python
images, labels = next(train_generator)

# 设置绘图参数
plt.figure(figsize=(12, 8))

# 显示前10张图片及其标签
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(images[i])  # 显示图片
    plt.title(f'Label: {labels[i]}')  # 显示标签
    plt.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2d001bf5d3464104bcdbb13dbf82a9e0.png)
## 构建模型  
构建的是卷积神经网络CNN的模型，如下  
输入层: 形状为 (150, 150, 3) 的图像输入。  
卷积层 1: 32 个卷积核，尺寸为 (3, 3)，激活函数为 ReLU。  
池化层 1: 最大池化层，池化窗口为 (2, 2)。  
卷积层 2: 64 个卷积核，尺寸为 (3, 3)，激活函数为 ReLU。  
池化层 2: 最大池化层，池化窗口为 (2, 2)。  
卷积层 3: 128 个卷积核，尺寸为 (3, 3)，激活函数为 ReLU。  
池化层 3: 最大池化层，池化窗口为 (2, 2)。  
展平层: 将多维特征图展平为一维。  
全连接层 1: 128 个节点，激活函数为 ReLU。  
dropout 层: 以减少过拟合，丢弃率为 0.5。  
全连接层 2（输出层）: 节点数与类别数相同，激活函数为 softmax

```python
# 获取类别数量
num_classes = len(train_generator.class_indices)

# 构建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # 使用 num_classes

```

```python
model.summary()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df023111d65f43c2aac5a0441ba8cfcc.png)

```python
# 编译模型
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

```python
# 训练模型
history = model.fit(train_generator, 
                    steps_per_epoch=train_generator.n // train_generator.batch_size, 
                    validation_data=val_generator, 
                    validation_steps=val_generator.n // val_generator.batch_size, 
                    epochs=40)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f0bd49a1424447d837f87a1bdc84d56.png)
## 模型评估

```python
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e0dd291064d44ba8ef893efd0faf0d7.png)

```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc4aa0f5e04f42a9b387c18e00e4292c.png)

```python
# 保存模型
model.save('butterfly_classifier.h5')

```
## 使用模型进行预测展示

```python
# 加载之前保存的模型
model = load_model('butterfly_classifier.h5')

val_images, val_labels = next(val_generator)

# 进行预测
predictions = model.predict(val_images)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(val_labels, axis=1)

# 获取类别映射
class_indices = val_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}

# 定义显示图像的函数
def display_images(images, true_labels, pred_labels, class_names, num_images=9):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        true_label = class_names[int(true_labels[i])]
        pred_label = class_names[int(pred_labels[i])]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 调用显示函数
display_images(val_images, true_labels, pred_labels, class_names, num_images=9)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/da72942d1d754b69bda308d3709495d2.png)
# 总结  
这次这个基于cnn的图像分类，获得了高于 70% 的准确率。可以加载我保存好的模型进行预测试试，感兴趣的还可以继续调参训练
```python
# 若需要完整数据集以及代码请点击以下链接
https://mbd.pub/o/bread/aJaVmJ9s
```

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Tải dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0  # Chuẩn hóa

# Xây dựng mô hình
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5, verbose=0)

# Đánh giá

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}")

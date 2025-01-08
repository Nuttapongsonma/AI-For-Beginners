# Import libraries
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# โหลดข้อมูล
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับขนาดข้อมูล (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer สำหรับ regression
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ฝึกโมเดล
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# ประเมินโมเดลด้วยชุดทดสอบ
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# ทำนายผลลัพธ์
predictions = model.predict(X_test)

# แสดงผลลัพธ์การทำนาย
print("Predictions:", predictions.flatten()[:5])  # แสดง 5 ผลลัพธ์แรก
print("Actual values:", y_test[:5])  # แสดง 5 ค่าจริงแรก

# พล็อตกราฟ loss และ validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# 1. XOR Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 1, 1, 0])  # Outputs (labels)

# 2. Define MLP Model with Regularization and Dropout
model = Sequential([
    Dense(4, input_dim=2, activation='relu', kernel_regularizer=l2(0.01), name='Hidden_Layer'),  # L2 Regularization
    Dropout(0.2),  # Dropout layer to reduce overfitting
    Dense(1, activation='sigmoid', name='Output_Layer')  # Output layer
])

# 3. Compile and Train Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=5000, batch_size=4, verbose=0)  # Train the model

# 4. Generate Decision Boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict the output for each grid point
predictions = model.predict(grid_points).reshape(xx.shape)

# 5. Plot Decision Boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], colors=['red', 'blue'], alpha=0.5)

# Plot original data points
for i, (x1, x2) in enumerate(X):
    color = 'blue' if y[i] == 1 else 'red'
    plt.scatter(x1, x2, color=color, edgecolor='k', s=100, label=f'Class {y[i]}' if i < 2 else '')

plt.title("Decision Boundary of MLP for XOR Problem with Regularization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

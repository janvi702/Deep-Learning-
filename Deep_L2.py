# Import required libraries
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset
iris = load_iris()

X = iris.data      # Input features
y = iris.target    # Output classes

# 2. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

# 3. Normalize the input data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the Multilayer Perceptron (MLP) model
model = tf.keras.Sequential([
    
    # Input layer
    tf.keras.layers.Input(shape=(4,)),
    
    # Hidden layer 1
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Hidden layer 2
    tf.keras.layers.Dense(8, activation='relu'),
    
    # Output layer: 3 classes
    tf.keras.layers.Dense(3, activation='softmax')
])

# 5. Compile the model
model.compile(
    optimizer='adam', #adaptive momentum estimation- Weight and bias
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the model
model.fit( #ready to training
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
# 7. Evaluate the trained model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test Accuracy Percentage:", accuracy * 100, "%")

# 8. Classify some test samples
predictions = model.predict(X_test)

predicted_classes = predictions.argmax(axis=1)

print("\nActual Classes:   ", y_test[:10])
print("Predicted Classes:", predicted_classes[:10])

# Display output class names
print("\nClass Names:")
print("0 = Setosa")
print("1 = Versicolor")
print("2 = Virginica")
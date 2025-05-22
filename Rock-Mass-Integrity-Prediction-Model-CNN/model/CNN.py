import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Concatenate

# Load text and image data
text_file_path = ''
image_file_path = ''

# Read text features
text_df = pd.read_excel(text_file_path, header=1)
# Read image features
image_df = pd.read_excel(image_file_path, header=0)

# Data cleaning
text_numeric_cols = text_df.columns[2:]
text_df[text_numeric_cols] = text_df[text_numeric_cols].apply(pd.to_numeric, errors='coerce')
text_df = text_df.dropna()

image_numeric_cols = image_df.columns[2:]
image_df[image_numeric_cols] = image_df[image_numeric_cols].apply(pd.to_numeric, errors='coerce')
image_df = image_df.dropna()

# Extract features
X_text = text_df.iloc[:, 2:].values
# Labels range [0, 4]
y_text = text_df.iloc[:, 1].values - 1

X_image = image_df.iloc[:, 2:].values
# Labels range [0, 4]
y_image = image_df.iloc[:, 1].values - 1

# Split datasets independently
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)
X_train_image, X_test_image, y_train_image, y_test_image = train_test_split(X_image, y_image, test_size=0.2,
                                                                            random_state=42)

# Align the number of samples for training sets
min_samples = min(X_train_text.shape[0], X_train_image.shape[0])
X_train_text = X_train_text[:min_samples]
X_train_image = X_train_image[:min_samples]
y_train_text = y_train_text[:min_samples]

# Align the number of samples for test sets
min_test_samples = min(X_test_text.shape[0], X_test_image.shape[0])
X_test_text = X_test_text[:min_test_samples]
X_test_image = X_test_image[:min_test_samples]
y_test_text = y_test_text[:min_test_samples]

# Feature scaling
scaler_text = StandardScaler()
X_train_text = scaler_text.fit_transform(X_train_text)
X_test_text = scaler_text.transform(X_test_text)

scaler_image = StandardScaler()
X_train_image = scaler_image.fit_transform(X_train_image)
X_test_image = scaler_image.transform(X_test_image)

# Reshape image features for CNN input
X_train_image = np.expand_dims(X_train_image, axis=2)
X_test_image = np.expand_dims(X_test_image, axis=2)

# Define model inputs
input_text = Input(shape=(X_train_text.shape[1],), name='text_input')
input_image = Input(shape=(X_train_image.shape[1], 1), name='image_input')

# Text feature branch
dense_text = Dense(128, activation='relu')(input_text)
dropout_text = Dropout(0.5)(dense_text)

# Image feature CNN branch
conv1 = Conv1D(filters=50, kernel_size=3, activation='relu')(input_image)
pool1 = MaxPooling1D(pool_size=2)(conv1)
drop1 = Dropout(0.5)(pool1)
conv2 = Conv1D(filters=100, kernel_size=3, activation='relu')(drop1)
pool2 = MaxPooling1D(pool_size=2)(conv2)
drop2 = Dropout(0.5)(pool2)
flat_image = Flatten()(drop2)

# Merge branches
merged = Concatenate()([dropout_text, flat_image])

# Fully connected classification head
dense_merged = Dense(128, activation='relu')(merged)
dropout_merged = Dropout(0.5)(dense_merged)
output = Dense(5, activation='softmax')(dropout_merged)

# Build model
model = Model(inputs=[input_text, input_image], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
start_train_time = time.time()
history = model.fit(
    [X_train_text, X_train_image],
    y_train_text,
    epochs=20,
    batch_size=32,
    validation_data=([X_test_text, X_test_image], y_test_text)
)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"Training time: {train_time:.2f} seconds")

# Save training history
history_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})
history_df.to_excel('combined_training_history.xlsx', index=False)

# Plot accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('combined_accuracy_curve.png')
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('combined_loss_curve.png')
plt.show()

# Prediction on test set
start_test_time = time.time()
y_pred = model.predict([X_test_text, X_test_image])
end_test_time = time.time()
test_time = end_test_time - start_test_time
print(f"Prediction time: {test_time:.2f} seconds")

# Convert predictions to class indices
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
report = classification_report(y_test_text, y_pred_classes, target_names=[f'Class {i + 1}' for i in range(5)])
print(report)

# Save classification report to Excel
report_df = pd.DataFrame(
    classification_report(y_test_text, y_pred_classes, target_names=[f'Class {i + 1}' for i in range(5)],
                          output_dict=True)).transpose()
report_df.to_excel('classification_report.xlsx', index=True)


# Permutation-based feature importance
def permutation_importance(model, X_test, y_test, baseline_accuracy):
    feature_importance = []
    for i in range(X_test.shape[1]):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])  # Shuffle i-th feature
        y_pred_permuted = model.predict([X_test_permuted, X_test_image])
        y_pred_permuted_classes = np.argmax(y_pred_permuted, axis=1)
        permuted_accuracy = accuracy_score(y_test, y_pred_permuted_classes)
        importance = baseline_accuracy - permuted_accuracy
        feature_importance.append(importance)
    return feature_importance


# Compute feature importance
baseline_accuracy = accuracy_score(y_test_text, y_pred_classes)
text_feature_importance = permutation_importance(model, X_test_text, y_test_text, baseline_accuracy)

# Display top 20 important features
importance_df = pd.DataFrame({'Feature': text_numeric_cols, 'Importance': text_feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
print(importance_df)

# Save feature importance to Excel
importance_df.to_excel('top_20_feature_importance.xlsx', index=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.grid(True)
plt.savefig('top_20_feature_importance.png')
plt.show()


# Sensitivity analysis with feature perturbation
def sensitivity_analysis(model, X_test, y_test, perturbation=0.05):
    sensitivity_scores = []
    baseline_accuracy = accuracy_score(y_test, np.argmax(model.predict([X_test, X_test_image]), axis=1))

    for i in range(X_test.shape[1]):
        # Positive perturbation
        X_test_perturbed = X_test.copy()
        X_test_perturbed[:, i] += X_test[:, i] * perturbation
        accuracy_perturbed = accuracy_score(y_test, np.argmax(model.predict([X_test_perturbed, X_test_image]), axis=1))
        positive_sensitivity = abs(baseline_accuracy - accuracy_perturbed)

        # Negative perturbation
        X_test_perturbed[:, i] -= 2 * X_test[:, i] * perturbation  # Restore then perturb negatively
        accuracy_perturbed = accuracy_score(y_test, np.argmax(model.predict([X_test_perturbed, X_test_image]), axis=1))
        negative_sensitivity = abs(baseline_accuracy - accuracy_perturbed)

        # Average sensitivity
        sensitivity = (positive_sensitivity + negative_sensitivity) / 2
        sensitivity_scores.append(sensitivity)

    return sensitivity_scores


# Perform sensitivity analysis with higher perturbation
text_sensitivity_scores = sensitivity_analysis(model, X_test_text, y_test_text, perturbation=0.30)

# Display top 20 sensitive features
sensitivity_df = pd.DataFrame({'Feature': text_numeric_cols, 'Sensitivity': text_sensitivity_scores})
sensitivity_df = sensitivity_df.sort_values(by='Sensitivity', ascending=False).head(20)
print(sensitivity_df)

# Save sensitivity analysis results to Excel
sensitivity_df.to_excel('top_20_feature_sensitivity.xlsx', index=False)

# Plot sensitivity scores
plt.figure(figsize=(10, 6))
plt.barh(sensitivity_df['Feature'], sensitivity_df['Sensitivity'])
plt.title('Top 20 Feature Sensitivity')
plt.xlabel('Sensitivity')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.grid(True)
plt.savefig('top_20_feature_sensitivity.png')
plt.show()

# Final message
print("All analysis results (feature importance and sensitivity analysis) have been saved as Excel and image files.")

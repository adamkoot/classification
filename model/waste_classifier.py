import numpy as np
import os
import random
from PIL import Image
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        """
        Initialize neural network with given architecture
        
        Args:
            input_size: Number of input features
            hidden_layers: List of integers representing neurons in each hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases using Xavier/Glorot initialization
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers) - 1):
            # Xavier initialization for weights
            limit = np.sqrt(6.0 / (self.layers[i] + self.layers[i + 1]))
            w = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i + 1]))
            self.weights.append(w)
            
            # Initialize biases to small positive values
            b = np.zeros((1, self.layers[i + 1]))
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            activations: List of activations for each layer
        """
        activations = [X]
        current_input = X
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self.relu(z)
            activations.append(current_input)
        
        # Output layer with softmax
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z_output)
        activations.append(output)
        
        return activations
    
    def backward_pass(self, X, y, activations):
        """
        Backward propagation to compute gradients
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            activations: Activations from forward pass
        """
        m = X.shape[0]  # batch size
        
        # Compute output layer error
        output_error = activations[-1] - y
        
        # Initialize gradients
        weight_gradients = []
        bias_gradients = []
        
        # Output layer gradients
        weight_grad = np.dot(activations[-2].T, output_error) / m
        bias_grad = np.mean(output_error, axis=0, keepdims=True)
        weight_gradients.append(weight_grad)
        bias_gradients.append(bias_grad)
        
        # Backpropagate through hidden layers
        current_error = output_error
        for i in range(len(self.weights) - 2, -1, -1):
            # Compute error for current layer
            current_error = np.dot(current_error, self.weights[i + 1].T)
            current_error = current_error * self.relu_derivative(activations[i + 1])
            
            # Compute gradients
            weight_grad = np.dot(activations[i].T, current_error) / m
            bias_grad = np.mean(current_error, axis=0, keepdims=True)
            
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)
        
        # Reverse gradients to match layer order
        weight_gradients.reverse()
        bias_gradients.reverse()
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute categorical cross-entropy loss"""
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Size of each training batch
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i + batch_size]
                batch_y = y_train_shuffled[i:i + batch_size]
                
                # Forward pass
                activations = self.forward_pass(batch_X)
                
                # Backward pass
                weight_grads, bias_grads = self.backward_pass(batch_X, batch_y, activations)
                
                # Update parameters
                self.update_parameters(weight_grads, bias_grads)
            
            # Compute epoch metrics
            train_pred = self.predict_proba(X_train)
            val_pred = self.predict_proba(X_val)
            
            train_loss = self.compute_loss(y_train, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)
            
            train_acc = self.compute_accuracy(y_train, train_pred)
            val_acc = self.compute_accuracy(y_val, val_pred)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("-" * 50)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        activations = self.forward_pass(X)
        return activations[-1]
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy from one-hot encoded labels and predictions"""
        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(true_labels == pred_labels)

class WasteClassifier:
    def __init__(self, image_size=(64, 64)):
        """
        Waste classification system
        
        Args:
            image_size: Tuple of (width, height) for image preprocessing
        """
        self.image_size = image_size
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
        # Bin mapping for waste disposal
        self.bin_mapping = {
            'cardboard': 'Recycling Bin (Paper/Cardboard)',
            'glass': 'Recycling Bin (Glass)',
            'metal': 'Recycling Bin (Metal/Cans)',
            'paper': 'Recycling Bin (Paper/Cardboard)',
            'plastic': 'Recycling Bin (Plastic)',
            'trash': 'General Waste Bin'
        }
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array.flatten()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """
        Load dataset from directory structure
        
        Args:
            data_dir: Path to data directory containing class subdirectories
            
        Returns:
            X: Feature matrix
            y: One-hot encoded labels
            filenames: List of loaded filenames
        """
        X = []
        y = []
        filenames = []
        
        print("Loading dataset...")
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            class_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Loading {len(class_files)} images from {class_name}...")
            
            for filename in class_files:
                image_path = os.path.join(class_dir, filename)
                img_features = self.load_image(image_path)
                
                if img_features is not None:
                    X.append(img_features)
                    # One-hot encode the label
                    label = np.zeros(len(self.classes))
                    label[self.class_to_idx[class_name]] = 1
                    y.append(label)
                    filenames.append(f"{class_name}/{filename}")
        
        return np.array(X), np.array(y), filenames
    
    def normalize_data(self, X_train, X_val=None, X_test=None):
        """Normalize features using training data statistics"""
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        
        results = [X_train_norm]
        
        if X_val is not None:
            X_val_norm = (X_val - self.scaler_mean) / self.scaler_std
            results.append(X_val_norm)
        
        if X_test is not None:
            X_test_norm = (X_test - self.scaler_mean) / self.scaler_std
            results.append(X_test_norm)
        
        return results if len(results) > 1 else results[0]
    
    def train_model(self, data_dir, validation_split=0.2, test_split=0.1):
        """
        Train the waste classification model
        
        Args:
            data_dir: Path to data directory
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        # Load dataset
        X, y, filenames = self.load_dataset(data_dir)
        
        if len(X) == 0:
            raise ValueError("No images loaded. Check your data directory structure.")
        
        print(f"Loaded {len(X)} images with {X.shape[1]} features each")
        print(f"Class distribution: {np.sum(y, axis=0)}")
        
        # Split data
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_split)
        val_size = int(len(X) * validation_split)
        train_size = len(X) - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Normalize data
        X_train_norm, X_val_norm, X_test_norm = self.normalize_data(X_train, X_val, X_test)
        
        # Initialize model with optimal configuration from analysis
        input_size = X_train_norm.shape[1]
        hidden_layers = [64, 32]  # Best configuration from your experiments
        output_size = len(self.classes)
        
        self.model = NeuralNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            learning_rate=0.01
        )
        
        # Train model
        print("Starting training...")
        history = self.model.train(
            X_train_norm, y_train,
            X_val_norm, y_val,
            epochs=50,  # From optimal configuration
            batch_size=32  # From optimal configuration
        )
        
        # Evaluate on test set
        test_pred = self.model.predict_proba(X_test_norm)
        test_accuracy = self.model.compute_accuracy(y_test, test_pred)
        test_loss = self.model.compute_loss(y_test, test_pred)
        
        print(f"\nFinal Test Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return history
    
    def classify_image(self, image_path):
        """
        Classify a single image and recommend disposal bin
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Load and preprocess image
        img_features = self.load_image(image_path)
        if img_features is None:
            return {"error": "Could not load image"}
        
        # Normalize using training statistics
        img_normalized = (img_features - self.scaler_mean) / self.scaler_std
        img_normalized = img_normalized.reshape(1, -1)
        
        # Make prediction
        probabilities = self.model.predict_proba(img_normalized)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Get bin recommendation
        recommended_bin = self.bin_mapping[predicted_class]
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "recommended_bin": recommended_bin,
            "class_probabilities": {
                self.classes[i]: float(probabilities[i]) 
                for i in range(len(self.classes))
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'weights': self.model.weights,
            'biases': self.model.biases,
            'layers': self.model.layers,
            'learning_rate': self.model.learning_rate,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'bin_mapping': self.bin_mapping,
            'image_size': self.image_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model architecture
        self.model = NeuralNetwork(
            input_size=model_data['layers'][0],
            hidden_layers=model_data['layers'][1:-1],
            output_size=model_data['layers'][-1],
            learning_rate=model_data['learning_rate']
        )
        
        # Restore weights and biases
        self.model.weights = model_data['weights']
        self.model.biases = model_data['biases']
        self.model.layers = model_data['layers']
        
        # Restore preprocessing parameters
        self.scaler_mean = model_data['scaler_mean']
        self.scaler_std = model_data['scaler_std']
        
        # Restore class information
        self.classes = model_data['classes']
        self.class_to_idx = model_data['class_to_idx']
        self.bin_mapping = model_data['bin_mapping']
        self.image_size = model_data['image_size']
        
        print(f"Model loaded from {filepath}")

# Example usage and complete workflow
if __name__ == "__main__":
    # Initialize classifier
    classifier = WasteClassifier(image_size=(64, 64))
    
    # Step 1: Train the model on your dataset
    print("Step 1: Training the model...")
    try:
        history = classifier.train_model('data/', validation_split=0.2, test_split=0.1)
        
        # Plot training history (optional - requires matplotlib)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history['train_losses'], label='Train Loss')
        # plt.plot(history['val_losses'], label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # 
        # plt.subplot(1, 2, 2)
        # plt.plot(history['train_accuracies'], label='Train Accuracy')
        # plt.plot(history['val_accuracies'], label='Validation Accuracy')
        # plt.title('Training and Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure your data directory structure is correct:")
        print("data/")
        print("├── cardboard/")
        print("│   ├── cardboard1.jpg")
        print("│   └── ...")
        print("├── glass/")
        print("├── metal/")
        print("├── paper/")
        print("├── plastic/")
        print("└── trash/")
    
    # Step 2: Save the trained model
    print("\nStep 2: Saving the model...")
    try:
        classifier.save_model('waste_classifier_model.pkl')
    except Exception as e:
        print(f"Could not save model: {e}")
    
    # Step 3: Load and test the model
    print("\nStep 3: Loading and testing the model...")
    try:
        # You can load the model in a new session
        # classifier_new = WasteClassifier()
        # classifier_new.load_model('waste_classifier_model.pkl')
        
        # Test on a new image
        test_image_path = 'path/to/test_image.jpg'  # Replace with actual path
        
        if os.path.exists(test_image_path):
            result = classifier.classify_image(test_image_path)
            
            print(f"\n🔍 Classification Results:")
            print(f"📋 Predicted class: {result['predicted_class'].upper()}")
            print(f"🎯 Confidence: {result['confidence']:.2%}")
            print(f"🗑️  Recommended bin: {result['recommended_bin']}")
            
            print(f"\n📊 All class probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"   {class_name:>10}: {prob:.2%}")
                
        else:
            print("Test image not found. Replace 'path/to/test_image.jpg' with actual image path")
            
    except Exception as e:
        print(f"Testing failed: {e}")

# Additional utility functions
def batch_classify_images(classifier, image_directory):
    """
    Classify all images in a directory
    
    Args:
        classifier: Trained WasteClassifier instance
        image_directory: Directory containing images to classify
    """
    if not os.path.exists(image_directory):
        print(f"Directory {image_directory} not found")
        return
    
    image_files = [f for f in os.listdir(image_directory) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Classifying {len(image_files)} images...")
    results = []
    
    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        try:
            result = classifier.classify_image(image_path)
            result['filename'] = filename
            results.append(result)
            
            print(f"{filename:>20} → {result['predicted_class']:>10} "
                  f"({result['confidence']:.2%}) → {result['recommended_bin']}")
                  
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return results

def evaluate_model_performance(classifier, test_data_dir):
    """
    Evaluate model performance on a test dataset
    
    Args:
        classifier: Trained WasteClassifier instance
        test_data_dir: Directory with same structure as training data
    """
    print("Evaluating model performance...")
    
    # Load test data
    X_test, y_test, filenames = classifier.load_dataset(test_data_dir)
    
    if len(X_test) == 0:
        print("No test data found")
        return
    
    # Normalize test data
    X_test_norm = (X_test - classifier.scaler_mean) / classifier.scaler_std
    
    # Make predictions
    predictions = classifier.model.predict_proba(X_test_norm)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    print("\nPer-class Performance:")
    for i, class_name in enumerate(classifier.classes):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(f"{class_name:>10}: {class_accuracy:.4f} "
                  f"({np.sum(class_mask)} samples)")
    
    # Confusion matrix (simple version)
    print("\nConfusion Matrix (True vs Predicted):")
    print("Rows: True classes, Columns: Predicted classes")
    confusion = np.zeros((len(classifier.classes), len(classifier.classes)), dtype=int)
    
    for true_idx, pred_idx in zip(true_classes, predicted_classes):
        confusion[true_idx, pred_idx] += 1
    
    # Print header
    print(f"{'':>12}", end="")
    for class_name in classifier.classes:
        print(f"{class_name[:8]:>8}", end="")
    print()
    
    # Print matrix
    for i, class_name in enumerate(classifier.classes):
        print(f"{class_name[:10]:>10}: ", end="")
        for j in range(len(classifier.classes)):
            print(f"{confusion[i, j]:>8}", end="")
        print()

# Quick start guide
def print_quick_start_guide():
    """Print usage instructions"""
    print("""
    🚀 WASTE CLASSIFIER QUICK START GUIDE
    =====================================
    
    1. PREPARE YOUR DATA:
       Organize images in this structure:
       data/
       ├── cardboard/    # Cardboard waste images
       ├── glass/        # Glass waste images  
       ├── metal/        # Metal waste images
       ├── paper/        # Paper waste images
       ├── plastic/      # Plastic waste images
       └── trash/        # General trash images
    
    2. TRAIN THE MODEL:
       classifier = WasteClassifier()
       classifier.train_model('data/')
       classifier.save_model('my_model.pkl')
    
    3. USE THE MODEL:
       classifier.load_model('my_model.pkl')
       result = classifier.classify_image('test_image.jpg')
       print(result['recommended_bin'])
    
    4. WASTE BIN MAPPING:
       • Cardboard → Recycling Bin (Paper/Cardboard)
       • Glass     → Recycling Bin (Glass)
       • Metal     → Recycling Bin (Metal/Cans)
       • Paper     → Recycling Bin (Paper/Cardboard)
       • Plastic   → Recycling Bin (Plastic)
       • Trash     → General Waste Bin
    
    💡 TIP: Start with 64x64 pixel images for faster training!
    """)

# Uncomment to see the quick start guide
# print_quick_start_guide()
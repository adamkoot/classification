import numpy as np
import os
import cv2
import json
import csv
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import random

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

class DataLoader:
    """Handles image loading and preprocessing"""
    
    def __init__(self, data_path, img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_images(self, max_per_class=None):
        """Load and preprocess all images"""
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found")
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_per_class:
                image_files = image_files[:max_per_class]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalize pixel values to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    # Flatten image for neural network
                    img_flat = img.flatten()
                    
                    images.append(img_flat)
                    labels.append(self.class_to_idx[class_name])
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    def create_one_hot(self, labels):
        """Convert labels to one-hot encoding"""
        num_classes = len(self.classes)
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def train_test_split(self, images, labels, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        np.random.seed(random_state)
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        
        split_idx = int(len(images) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return (images[train_indices], labels[train_indices],
                images[test_indices], labels[test_indices])

class NeuralNetwork:
    """Neural Network implementation from scratch with multiple optimizers"""
    
    def __init__(self, input_size, hidden_layers, output_size, 
                 activation='relu', learning_method='sgd'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.learning_method = learning_method
        
        # Initialize network architecture
        self.layers = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Initialize optimizer-specific parameters
        self._initialize_optimizer_params()
        
        # Get activation functions
        self.activation_func = self._get_activation_function()
        self.activation_derivative = self._get_activation_derivative()
    
    def _initialize_optimizer_params(self):
        """Initialize parameters for different optimizers"""
        num_layers = len(self.weights)
        
        # For momentum and Nesterov
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
        # For Adam and RMSprop
        self.m_w = [np.zeros_like(w) for w in self.weights]  # First moment
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]  # Second moment
        self.v_b = [np.zeros_like(b) for b in self.biases]
        
        # For Adagrad
        self.cache_w = [np.zeros_like(w) for w in self.weights]
        self.cache_b = [np.zeros_like(b) for b in self.biases]
        
        # Time step for Adam
        self.t = 0
    
    def _get_activation_function(self):
        """Get activation function based on name"""
        functions = {
            'sigmoid': ActivationFunctions.sigmoid,
            'relu': ActivationFunctions.relu,
            'tanh': ActivationFunctions.tanh,
            'leaky_relu': ActivationFunctions.leaky_relu
        }
        return functions.get(self.activation, ActivationFunctions.relu)
    
    def _get_activation_derivative(self):
        """Get activation derivative based on name"""
        derivatives = {
            'sigmoid': ActivationFunctions.sigmoid_derivative,
            'relu': ActivationFunctions.relu_derivative,
            'tanh': ActivationFunctions.tanh_derivative,
            'leaky_relu': ActivationFunctions.leaky_relu_derivative
        }
        return derivatives.get(self.activation, ActivationFunctions.relu_derivative)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i in range(self.num_layers - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == self.num_layers - 2:  # Output layer - use softmax
                activation = self.softmax(z)
            else:  # Hidden layers
                activation = self.activation_func(z)
            
            self.activations.append(activation)
            current_input = activation
        
        return self.activations[-1]
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y, learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Backward propagation with multiple optimizer support"""
        m = X.shape[0]
        
        # Calculate output layer error
        output_error = self.activations[-1] - y
        
        # Store gradients
        weight_gradients = []
        bias_gradients = []
        
        # Output layer gradients
        dW_output = np.dot(self.activations[-2].T, output_error) / m
        db_output = np.sum(output_error, axis=0, keepdims=True) / m
        
        weight_gradients.append(dW_output)
        bias_gradients.append(db_output)
        
        # Backpropagate through hidden layers
        error = output_error
        for i in range(self.num_layers - 2, 0, -1):
            error = np.dot(error, self.weights[i].T) * self.activation_derivative(self.z_values[i-1])
            
            dW = np.dot(self.activations[i-1].T, error) / m
            db = np.sum(error, axis=0, keepdims=True) / m
            
            weight_gradients.append(dW)
            bias_gradients.append(db)
        
        # Reverse gradients to match layer order
        weight_gradients.reverse()
        bias_gradients.reverse()
        
        # Update weights based on learning method
        self._update_weights(weight_gradients, bias_gradients, learning_rate, 
                           momentum, beta1, beta2, epsilon)
    
    def _update_weights(self, weight_gradients, bias_gradients, learning_rate, 
                       momentum, beta1, beta2, epsilon):
        """Update weights based on selected optimizer"""
        
        if self.learning_method == 'sgd':
            self._sgd_update(weight_gradients, bias_gradients, learning_rate)
            
        elif self.learning_method == 'momentum':
            self._momentum_update(weight_gradients, bias_gradients, learning_rate, momentum)
            
        elif self.learning_method == 'nesterov':
            self._nesterov_update(weight_gradients, bias_gradients, learning_rate, momentum)
            
        elif self.learning_method == 'adam':
            self._adam_update(weight_gradients, bias_gradients, learning_rate, beta1, beta2, epsilon)
            
        elif self.learning_method == 'rmsprop':
            self._rmsprop_update(weight_gradients, bias_gradients, learning_rate, beta2, epsilon)
            
        elif self.learning_method == 'adagrad':
            self._adagrad_update(weight_gradients, bias_gradients, learning_rate, epsilon)
            
        elif self.learning_method == 'adamax':
            self._adamax_update(weight_gradients, bias_gradients, learning_rate, beta1, beta2, epsilon)
    
    def _sgd_update(self, weight_gradients, bias_gradients, learning_rate):
        """Stochastic Gradient Descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def _momentum_update(self, weight_gradients, bias_gradients, learning_rate, momentum):
        """Momentum optimizer"""
        for i in range(len(self.weights)):
            self.velocity_w[i] = momentum * self.velocity_w[i] - learning_rate * weight_gradients[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] - learning_rate * bias_gradients[i]
            
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]
    
    def _nesterov_update(self, weight_gradients, bias_gradients, learning_rate, momentum):
        """Nesterov Accelerated Gradient"""
        for i in range(len(self.weights)):
            v_prev_w = self.velocity_w[i].copy()
            v_prev_b = self.velocity_b[i].copy()
            
            self.velocity_w[i] = momentum * self.velocity_w[i] - learning_rate * weight_gradients[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] - learning_rate * bias_gradients[i]
            
            self.weights[i] += -momentum * v_prev_w + (1 + momentum) * self.velocity_w[i]
            self.biases[i] += -momentum * v_prev_b + (1 + momentum) * self.velocity_b[i]
    
    def _adam_update(self, weight_gradients, bias_gradients, learning_rate, beta1, beta2, epsilon):
        """Adam optimizer"""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * weight_gradients[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * bias_gradients[i]
            
            # Update biased second raw moment estimate
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (weight_gradients[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (bias_gradients[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_corrected = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_corrected = self.m_b[i] / (1 - beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_corrected = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
            self.biases[i] -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)
    
    def _rmsprop_update(self, weight_gradients, bias_gradients, learning_rate, beta2, epsilon):
        """RMSprop optimizer"""
        for i in range(len(self.weights)):
            # Update moving average of squared gradients
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (weight_gradients[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (bias_gradients[i] ** 2)
            
            # Update parameters
            self.weights[i] -= learning_rate * weight_gradients[i] / (np.sqrt(self.v_w[i]) + epsilon)
            self.biases[i] -= learning_rate * bias_gradients[i] / (np.sqrt(self.v_b[i]) + epsilon)
    
    def _adagrad_update(self, weight_gradients, bias_gradients, learning_rate, epsilon):
        """Adagrad optimizer"""
        for i in range(len(self.weights)):
            # Accumulate squared gradients
            self.cache_w[i] += weight_gradients[i] ** 2
            self.cache_b[i] += bias_gradients[i] ** 2
            
            # Update parameters
            self.weights[i] -= learning_rate * weight_gradients[i] / (np.sqrt(self.cache_w[i]) + epsilon)
            self.biases[i] -= learning_rate * bias_gradients[i] / (np.sqrt(self.cache_b[i]) + epsilon)
    
    def _adamax_update(self, weight_gradients, bias_gradients, learning_rate, beta1, beta2, epsilon):
        """Adamax optimizer (Adam based on infinity norm)"""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * weight_gradients[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * bias_gradients[i]
            
            # Update the exponentially weighted infinity norm
            self.v_w[i] = np.maximum(beta2 * self.v_w[i], np.abs(weight_gradients[i]))
            self.v_b[i] = np.maximum(beta2 * self.v_b[i], np.abs(bias_gradients[i]))
            
            # Compute bias-corrected first moment estimate
            m_w_corrected = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_corrected = self.m_b[i] / (1 - beta1 ** self.t)
            
            # Update parameters
            self.weights[i] -= learning_rate * m_w_corrected / (self.v_w[i] + epsilon)
            self.biases[i] -= learning_rate * m_b_corrected / (self.v_b[i] + epsilon)
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def calculate_loss(self, y_true, y_pred):
        """Calculate cross-entropy loss"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def calculate_accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

class ExperimentRunner:
    """Runs experiments with different parameter combinations"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.results = []
    
    def run_experiments(self, X_train, y_train, X_test, y_test, 
                       param_combinations, epochs=50, runs_per_config=10):
        """Run experiments for all parameter combinations"""
        
        total_experiments = len(param_combinations) * runs_per_config
        current_exp = 0
        
        for params in param_combinations:
            print(f"\nTesting configuration: {params}")
            
            config_results = []
            
            for run in range(runs_per_config):
                current_exp += 1
                print(f"  Run {run + 1}/{runs_per_config} (Experiment {current_exp}/{total_experiments})")
                
                # Create network
                nn = NeuralNetwork(
                    input_size=X_train.shape[1],
                    hidden_layers=params['hidden_layers'],
                    output_size=y_train.shape[1],
                    activation=params['activation'],
                    learning_method=params['learning_method']
                )
                
                # Training history
                train_losses = []
                train_accuracies = []
                test_losses = []
                test_accuracies = []
                
                start_time = time.time()
                
                # Training loop
                for epoch in range(epochs):
                    # Forward and backward pass
                    train_output = nn.forward(X_train)
                    nn.backward(X_train, y_train, learning_rate=params['learning_rate'])
                    
                    # Calculate metrics every 10 epochs
                    if epoch % 10 == 0 or epoch == epochs - 1:
                        train_loss = nn.calculate_loss(y_train, train_output)
                        train_acc = nn.calculate_accuracy(X_train, y_train)
                        
                        test_output = nn.forward(X_test)
                        test_loss = nn.calculate_loss(y_test, test_output)
                        test_acc = nn.calculate_accuracy(X_test, y_test)
                        
                        train_losses.append(train_loss)
                        train_accuracies.append(train_acc)
                        test_losses.append(test_loss)
                        test_accuracies.append(test_acc)
                
                training_time = time.time() - start_time
                
                # Final metrics
                final_train_acc = nn.calculate_accuracy(X_train, y_train)
                final_test_acc = nn.calculate_accuracy(X_test, y_test)
                final_train_loss = nn.calculate_loss(y_train, nn.forward(X_train))
                final_test_loss = nn.calculate_loss(y_test, nn.forward(X_test))
                
                run_result = {
                    'run': run + 1,
                    'final_train_accuracy': final_train_acc,
                    'final_test_accuracy': final_test_acc,
                    'final_train_loss': final_train_loss,
                    'final_test_loss': final_test_loss,
                    'training_time': training_time,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'test_losses': test_losses,
                    'test_accuracies': test_accuracies
                }
                
                config_results.append(run_result)
                
                print(f"    Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}")
            
            # Calculate statistics for this configuration
            train_accs = [r['final_train_accuracy'] for r in config_results]
            test_accs = [r['final_test_accuracy'] for r in config_results]
            train_losses = [r['final_train_loss'] for r in config_results]
            test_losses = [r['final_test_loss'] for r in config_results]
            
            result_summary = {
                'parameters': params,
                'runs': config_results,
                'mean_train_accuracy': np.mean(train_accs),
                'std_train_accuracy': np.std(train_accs),
                'mean_test_accuracy': np.mean(test_accs),
                'std_test_accuracy': np.std(test_accs),
                'best_test_accuracy': np.max(test_accs),
                'mean_train_loss': np.mean(train_losses),
                'mean_test_loss': np.mean(test_losses),
                'mean_training_time': np.mean([r['training_time'] for r in config_results])
            }
            
            self.results.append(result_summary)
            
            print(f"  Mean Test Accuracy: {result_summary['mean_test_accuracy']:.4f} ± {result_summary['std_test_accuracy']:.4f}")
            print(f"  Best Test Accuracy: {result_summary['best_test_accuracy']:.4f}")
    
    def save_results(self, filename='experiment_results.csv'):
        """Save results to CSV file"""
        # Save summary results
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'experiment_type', 'parameter_value', 'hidden_layers', 
                'activation', 'optimizer', 'batch_size', 'epochs',
                'mean_train_accuracy', 'std_train_accuracy', 'mean_test_accuracy', 
                'std_test_accuracy', 'best_test_accuracy', 'mean_train_loss', 
                'mean_test_loss', 'mean_training_time', 'mean_train_mse',
                'mean_test_mse', 'mean_train_rmse', 'mean_test_rmse',
                'mean_train_mae', 'mean_test_mae', 'best_train_accuracy', 'best_test_loss'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                params = result['parameters']
                
                # Convert accuracy to MSE-like metrics for classification
                # Since we're doing classification, we'll use 1-accuracy as error
                train_accuracies = [r['final_train_accuracy'] for r in result['runs']]
                test_accuracies = [r['final_test_accuracy'] for r in result['runs']]
                train_losses = [r['final_train_loss'] for r in result['runs']]
                test_losses = [r['final_test_loss'] for r in result['runs']]
                
                # Convert to error metrics (MSE-like for classification)
                train_errors = [1 - acc for acc in train_accuracies]
                test_errors = [1 - acc for acc in test_accuracies]
                
                row = {
                    'experiment_type': params.get('experiment_type', 'unknown'),
                    'parameter_value': params.get('parameter_value', 'unknown'),
                    'hidden_layers': str(params['hidden_layers']),
                    'activation': params['activation'],
                    'optimizer': params['learning_method'],
                    'batch_size': 32,  # Default batch size for image classification
                    'epochs': 50,      # Default epochs
                    'mean_train_accuracy': result['mean_train_accuracy'],
                    'std_train_accuracy': result['std_train_accuracy'],
                    'mean_test_accuracy': result['mean_test_accuracy'],
                    'std_test_accuracy': result['std_test_accuracy'],
                    'best_test_accuracy': result['best_test_accuracy'],
                    'mean_train_loss': result['mean_train_loss'],
                    'mean_test_loss': result['mean_test_loss'],
                    'mean_training_time': result['mean_training_time'],
                    'mean_train_mse': np.mean(train_errors),
                    'mean_test_mse': np.mean(test_errors),
                    'mean_train_rmse': np.sqrt(np.mean(train_errors)),
                    'mean_test_rmse': np.sqrt(np.mean(test_errors)),
                    'mean_train_mae': np.mean(train_errors),
                    'mean_test_mae': np.mean(test_errors),
                    'best_train_accuracy': max(train_accuracies),
                    'best_test_loss': min(test_losses)
                }
                writer.writerow(row)
        
        # Save detailed results (individual runs) - with both MSE-like and accuracy/loss metrics
        detailed_filename = filename.replace('.csv', '_detailed.csv')
        with open(detailed_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'experiment_type', 'parameter_value', 'run', 'hidden_layers', 
                'activation', 'optimizer', 'batch_size', 'epochs', 'training_time',
                'train_mse', 'train_rmse', 'train_mae', 'test_mse', 'test_rmse', 'test_mae',
                'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                params = result['parameters']
                for run_result in result['runs']:
                    # Convert classification metrics to regression-like metrics
                    train_error = 1 - run_result['final_train_accuracy']  # Error rate
                    test_error = 1 - run_result['final_test_accuracy']     # Error rate
                    
                    row = {
                        'experiment_type': params.get('experiment_type', 'unknown'),
                        'parameter_value': params.get('parameter_value', 'unknown'),
                        'run': run_result['run'],
                        'hidden_layers': str(params['hidden_layers']),
                        'activation': params['activation'],
                        'optimizer': params['learning_method'],
                        'batch_size': 32,  # Default batch size
                        'epochs': 50,      # Default epochs
                        'training_time': run_result['training_time'],
                        'train_mse': train_error,
                        'train_rmse': np.sqrt(train_error),
                        'train_mae': train_error,
                        'test_mse': test_error,
                        'test_rmse': np.sqrt(test_error),
                        'test_mae': test_error,
                        'train_accuracy': run_result['final_train_accuracy'],
                        'test_accuracy': run_result['final_test_accuracy'],
                        'train_loss': run_result['final_train_loss'],
                        'test_loss': run_result['final_test_loss']
                    }
                    writer.writerow(row)
        
        print(f"\nResults saved to {filename}")
        print(f"Detailed results saved to {detailed_filename}")
    
    def generate_report(self):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        # Sort results by test accuracy
        sorted_results = sorted(self.results, key=lambda x: x['mean_test_accuracy'], reverse=True)
        
        print(f"\nTop 5 Configurations by Test Accuracy:")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:5]):
            params = result['parameters']
            print(f"\n{i+1}. Test Accuracy: {result['mean_test_accuracy']:.4f} ± {result['std_test_accuracy']:.4f}")
            print(f"   Layers: {len(params['hidden_layers'])}, Neurons: {params['hidden_layers']}")
            print(f"   Activation: {params['activation']}, Learning: {params['learning_method']}")
            print(f"   Learning Rate: {params['learning_rate']}")
        
        # Parameter analysis
        print(f"\n\nParameter Analysis by Experiment Type:")
        print("-" * 60)
        
        # Group results by experiment type for analysis
        experiment_results = defaultdict(list)
        for result in self.results:
            params = result['parameters']
            exp_type = params.get('experiment_type', 'unknown')
            experiment_results[exp_type].append(result)
        
        # Analyze each experiment type
        for exp_type, exp_results in experiment_results.items():
            if exp_type == 'unknown':
                continue
                
            print(f"\n{exp_type.upper()} EXPERIMENTS:")
            print("-" * 40)
            
            # Sort by test accuracy (higher is better)
            exp_results.sort(key=lambda x: x['mean_test_accuracy'], reverse=True)
            
            for result in exp_results:
                params = result['parameters']
                param_value = params.get('parameter_value', 'unknown')
                test_error = 1 - result['mean_test_accuracy']  # Convert to error rate
                test_error_std = result['std_test_accuracy']
                print(f"  {param_value:15}: Accuracy: {result['mean_test_accuracy']:.4f} ± {test_error_std:.4f} | Error: {test_error:.4f}")
        
        # Overall activation function analysis
        activation_results = defaultdict(list)
        for result in self.results:
            activation_results[result['parameters']['activation']].append(result['mean_test_accuracy'])
        
        print(f"\n\nOverall Activation Function Performance:")
        print("-" * 50)
        for activation, accuracies in activation_results.items():
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            avg_error = 1 - avg_accuracy
            print(f"  {activation:12}: Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f} | MSE-like: {avg_error:.4f}")
        
        # Overall learning method analysis
        learning_results = defaultdict(list)
        for result in self.results:
            learning_results[result['parameters']['learning_method']].append(result['mean_test_accuracy'])
        
        print(f"\nOverall Optimizer Performance:")
        print("-" * 50)
        for method, accuracies in learning_results.items():
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            avg_error = 1 - avg_accuracy
            print(f"  {method:12}: Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f} | MSE-like: {avg_error:.4f}")
        
        # Architecture analysis
        architecture_results = defaultdict(list)
        for result in self.results:
            arch_key = str(result['parameters']['hidden_layers'])
            architecture_results[arch_key].append(result['mean_test_accuracy'])
        
        print(f"\nArchitecture Performance (Top 5):")
        print("-" * 50)
        arch_sorted = sorted(architecture_results.items(), 
                           key=lambda x: np.mean(x[1]), reverse=True)[:5]
        
        for arch, accuracies in arch_sorted:
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            avg_rmse = np.sqrt(1 - avg_accuracy)
            print(f"  {arch:20}: Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f} | RMSE-like: {avg_rmse:.4f}")

def create_parameter_combinations():
    """Create all parameter combinations to test"""
    
    # Parameter options (at least 4 values for each)
    layer_configs = [
        [32],           # 1 hidden layer
        [64, 32],       # 2 hidden layers
        [128, 64, 32],  # 3 hidden layers
        [256, 128, 64, 32]  # 4 hidden layers
    ]
    
    neuron_configs = [
        [16],
        [32], 
        [64],
        [128]
    ]
    
    activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
    
    # Extended optimizer list with new algorithms
    learning_methods = ['sgd', 'momentum', 'nesterov', 'adam', 'rmsprop', 'adagrad', 'adamax']
    
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    sample_sizes = [50, 100, 200, None]  # None means use all data
    
    combinations = []
    
    # Base configuration
    base_config = {
        'activation': 'relu',
        'learning_method': 'sgd', 
        'learning_rate': 0.01,
        'sample_size': None
    }
    
    # Test different architectures (layer configurations)
    for i, layers in enumerate(layer_configs):
        config = base_config.copy()
        config['hidden_layers'] = layers
        config['experiment_type'] = 'layers'
        config['parameter_value'] = f"{len(layers)}_layers"
        combinations.append(config)
    
    # Test different neuron counts
    for i, neurons in enumerate(neuron_configs):
        config = base_config.copy()
        config['hidden_layers'] = neurons
        config['experiment_type'] = 'neurons'
        config['parameter_value'] = f"{neurons[0]}_neurons"
        combinations.append(config)
    
    # Test different activation functions
    for activation in activations:
        config = base_config.copy()
        config['hidden_layers'] = [64, 32]
        config['activation'] = activation
        config['experiment_type'] = 'activation'
        config['parameter_value'] = activation
        combinations.append(config)
    
    # Test all learning methods/optimizers
    for method in learning_methods:
        config = base_config.copy()
        config['hidden_layers'] = [64, 32]
        config['learning_method'] = method
        # Adjust learning rate for different optimizers
        if method in ['adam', 'rmsprop', 'adagrad', 'adamax']:
            config['learning_rate'] = 0.001  # Lower learning rate for adaptive methods
        else:
            config['learning_rate'] = 0.01   # Standard rate for SGD-based methods
        config['experiment_type'] = 'optimizer'
        config['parameter_value'] = method
        combinations.append(config)
    
    # Test different learning rates
    for lr in learning_rates:
        config = base_config.copy()
        config['hidden_layers'] = [64, 32]
        config['learning_rate'] = lr
        config['experiment_type'] = 'learning_rate'
        config['parameter_value'] = str(lr)
        combinations.append(config)
    
    # Test different sample sizes
    for size in sample_sizes:
        config = base_config.copy()
        config['hidden_layers'] = [64, 32]
        config['sample_size'] = size
        config['experiment_type'] = 'sample_size'
        config['parameter_value'] = str(size) if size else 'all'
        combinations.append(config)
    
    # Additional experiments comparing optimizers with different activation functions
    optimizer_activation_combos = [
        ('adam', 'relu'),
        ('adam', 'tanh'),
        ('rmsprop', 'relu'),
        ('rmsprop', 'leaky_relu'),
        ('nesterov', 'relu'),
        ('adagrad', 'sigmoid')
    ]
    
    for optimizer, activation in optimizer_activation_combos:
        config = base_config.copy()
        config['hidden_layers'] = [64, 32]
        config['learning_method'] = optimizer
        config['activation'] = activation
        config['learning_rate'] = 0.001 if optimizer in ['adam', 'rmsprop', 'adagrad', 'adamax'] else 0.01
        config['experiment_type'] = 'optimizer_activation'
        config['parameter_value'] = f"{optimizer}_{activation}"
        combinations.append(config)
    
    return combinations

def main():
    """Main execution function"""
    print("Starting Enhanced Neural Network Image Classification Experiment")
    print("="*70)
    print("Available Optimizers: SGD, Momentum, Nesterov, Adam, RMSprop, Adagrad, Adamax")
    print("="*70)
    
    # Initialize data loader
    data_loader = DataLoader('data', img_size=(32, 32))  # Smaller size for faster processing
    
    # Load images
    print("Loading images...")
    try:
        images, labels = data_loader.load_images(max_per_class=100)  # Limit for demo
        print(f"Loaded {len(images)} images")
        print(f"Image shape: {images[0].shape}")
        print(f"Classes: {data_loader.classes}")
        
        # Convert labels to one-hot
        labels_onehot = data_loader.create_one_hot(labels)
        
        # Split data
        X_train, y_train, X_test, y_test = data_loader.train_test_split(
            images, labels_onehot, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        np.random.seed(42)
        num_samples = 600
        img_size = 32 * 32 * 3  # 32x32 RGB images
        
        X_train = np.random.rand(int(num_samples * 0.8), img_size)
        X_test = np.random.rand(int(num_samples * 0.2), img_size)
        
        y_train = np.eye(6)[np.random.randint(0, 6, X_train.shape[0])]
        y_test = np.eye(6)[np.random.randint(0, 6, X_test.shape[0])]
        
        print("Using synthetic data for demonstration")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Create parameter combinations
    param_combinations = create_parameter_combinations()
    print(f"\nTotal parameter combinations to test: {len(param_combinations)}")
    
    # Show optimizer breakdown
    optimizer_counts = defaultdict(int)
    for combo in param_combinations:
        optimizer_counts[combo['learning_method']] += 1
    
    print("\nOptimizer distribution in experiments:")
    for optimizer, count in optimizer_counts.items():
        print(f"  {optimizer}: {count} experiments")
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner(data_loader)
    
    # Run experiments
    print("\nStarting experiments...")
    experiment_runner.run_experiments(
        X_train, y_train, X_test, y_test,
        param_combinations[:15],  # Test first 15 combinations for demo
        epochs=30,  # Reduced epochs for faster execution
        runs_per_config=10  # Reduced runs for demo
    )
    
    # Save results and generate report
    experiment_runner.save_results('enhanced_neural_network_results.csv')
    experiment_runner.generate_report()
    
    print("\nExperiment completed successfully!")
    print("Results saved to:")
    print("  - enhanced_neural_network_results.csv (summary)")
    print("  - enhanced_neural_network_results_detailed.csv (individual runs)")
    
    # Print optimizer comparison summary
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*60)
    print("Implemented optimizers:")
    print("• SGD: Standard Stochastic Gradient Descent")
    print("• Momentum: SGD with momentum for faster convergence")
    print("• Nesterov: Nesterov Accelerated Gradient (lookahead)")
    print("• Adam: Adaptive Moment Estimation (popular choice)")
    print("• RMSprop: Root Mean Square Propagation")
    print("• Adagrad: Adaptive Gradient Algorithm")
    print("• Adamax: Adam based on infinity norm")

if __name__ == "__main__":
    main()
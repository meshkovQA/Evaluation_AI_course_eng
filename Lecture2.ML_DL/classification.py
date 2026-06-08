import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, top_k_accuracy_score
from scipy.stats import entropy as scipy_entropy
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Configure the compute device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. LOADING AND PREPARING THE DATA (CIFAR-10)
# ============================================================================


def load_cifar10_data(batch_size=128, val_split=0.1):
    """
    Load and prepare the CIFAR-10 dataset with a train/validation/test split.

    CIFAR-10 contains 60,000 color 32x32 images in 10 classes:
    - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    Args:
        batch_size: batch size
        val_split: fraction of the training data to use for validation (0.1 = 10%)
    """

    # Define transforms for the training set (with augmentation)
    transform_train = transforms.Compose([
        # RandomHorizontalFlip: randomly flips the image horizontally (e.g. airplane left/right).
        transforms.RandomHorizontalFlip(p=0.5),
        # RandomCrop: randomly crops a portion of the image and adds padding (to imitate different angles).
        transforms.RandomCrop(32, padding=4),
        # ToTensor: converts the image from PIL/NumPy to a PyTorch tensor.
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))  # Normalization
    ])

    # Transforms for the validation and test sets (without augmentation)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load the raw data
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val_test
    )

    # Split the training set into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Create a generator for a reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Build a separate validation dataset without augmentation
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_val_test
    )

    # Get the indices for the validation set
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Dataset sizes:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, classes


# def load_mnist_data(batch_size=128, val_split=0.1):
#     """
#     Load and prepare the MNIST dataset with a train/validation/test split.
#
#     MNIST contains 70,000 grayscale 28x28 images of handwritten digits:
#     - digits from 0 to 9
#
#     Args:
#         batch_size: batch size
#         val_split: fraction of the training data to use for validation (0.1 = 10%)
#     """
#
#     # Training set transforms (with augmentation)
#     transform_train = transforms.Compose([
#         # Random horizontal flip (less relevant for digits)
#         transforms.RandomHorizontalFlip(p=0.1),  # Reduced probability
#         # Random crop with padding
#         transforms.RandomCrop(28, padding=2),     # Smaller padding for 28x28
#         # Random rotation (useful for digits)
#         transforms.RandomRotation(degrees=10),
#         transforms.ToTensor(),                    # Convert to tensor
#         #  normalization for MNIST
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     # Validation/test set transforms (without augmentation)
#     transform_val_test = transforms.Compose([
#         transforms.ToTensor(),
#         #  normalization for MNIST
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#
#     # Load MNIST instead of CIFAR-10
#     full_train_dataset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform_train
#     )
#     test_dataset = torchvision.datasets.MNIST(
#         root='./data', train=False, download=True, transform=transform_val_test
#     )
#
#     # Split the training set into train and validation (UNCHANGED)
#     train_size = int((1 - val_split) * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size
#
#     # Reproducible split generator
#     generator = torch.Generator().manual_seed(42)
#     train_dataset, val_dataset_temp = torch.utils.data.random_split(
#         full_train_dataset, [train_size, val_size], generator=generator
#     )
#
#     # Separate validation dataset without augmentation
#     val_dataset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=False, transform=transform_val_test
#     )
#
#     # Get indices for the validation set
#     val_indices = val_dataset_temp.indices
#     val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
#
#     # Build data loaders (UNCHANGED)
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     # MNIST class names
#     classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
#
#     print(f"Dataset sizes:")
#     print(f"  - Training: {len(train_dataset)} samples")
#     print(f"  - Validation: {len(val_dataset)} samples")
#     print(f"  - Test: {len(test_dataset)} samples")
#
#     return train_loader, val_loader, test_loader, classes

# ============================================================================
# 2. NEURAL NETWORK ARCHITECTURE DEFINITION
# ============================================================================


class SimpleCNN_CIFAR(nn.Module):
    """
    A simple convolutional neural network for CIFAR classification
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN_CIFAR, self).__init__()

        # First layer accepts 3 channels instead of 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 32x32x32 -> 32x32x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 16x16x64 -> 16x16x128

        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        #  Fully connected layers adapted to the 8x8x128 shape
        # 8x8x128 -> 256 (fewer neurons)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)           # 256 -> 64
        self.fc3 = nn.Linear(64, num_classes)   # 64 -> 10 (number of classes)

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.pool(self.relu(self.conv1(x)))  # 28x28x32 -> 14x14x32
        x = self.pool(self.relu(self.conv2(x)))  # 14x14x64 -> 7x7x64
        x = self.relu(self.conv3(x))             # 7x7x128 (NO pooling)

        # Flatten into a 1D vector
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (7*7*128)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer (logits)

        return x


# class VerySimpleCNN(nn.Module):
#     """
#     A very simple model for comparison
#     """
#
#     def __init__(self, num_classes=10):
#         super(VerySimpleCNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 16, 5)  # 28x28 -> 24x24
#         self.pool = nn.MaxPool2d(2, 2)    # 24x24 -> 12x12
#         self.conv2 = nn.Conv2d(16, 32, 5)  # 12x12 -> 8x8
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 4 * 4)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class SimpleCNN(nn.Module):
#     """
#     A simple convolutional neural network for classifying CIFAR-10 images
#     """
#
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         # 1) input channels = 1
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # 28x28 -> 28x28
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 14x14 -> 14x14
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 7x7   -> 7x7
#         # 3x3   -> 3x3 (after 3 pools)
#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#
#         # 2) size after three pools on 28x28 is 3x3
#         self.fc1 = nn.Linear(256 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         # Convolutional layers with activation and pooling
#         x = self.pool(self.relu(self.conv1(x)))  # 32x32x32 -> 16x16x32
#         x = self.pool(self.relu(self.conv2(x)))  # 16x16x64 -> 8x8x64
#         x = self.pool(self.relu(self.conv3(x)))  # 8x8x128 -> 4x4x128
#         x = self.relu(self.conv4(x))             # 4x4x256
#
#         # Flatten into a 1D vector
#         x = x.view(x.size(0), -1)  # Flatten: batch_size x (4*4*256)
#
#         # Fully connected layers
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)  # Output layer (logits)
#
#         return x

# ============================================================================
# 3. TRAINING AND EVALUATION FUNCTIONS
# ============================================================================


def validate_model(model, val_loader, criterion, device):
    """
    Validate the model (evaluate on the validation set)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the model with validation and metric tracking
    """
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # For early stopping
    best_val_accuracy = 0.0
    patience = 3  # Number of epochs without improvement
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ====================================================================
        # TRAINING
        # ====================================================================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Training metrics for this epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ====================================================================
        # VALIDATION
        # ====================================================================
        val_loss, val_accuracy = validate_model(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch metrics
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(
            f'  Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(
            f'  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

        # Check for improvement (for early stopping)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            # You could save the best model here
            print(
                f'  ✅ New best validation accuracy: {best_val_accuracy:.2f}%')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'  ⏹️ Early stopping: {patience} epochs without improvement')
                break

        print()

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }


def evaluate_model_with_metrics(model, test_loader, device, classes):
    """
    Comprehensive model evaluation with various metrics
    """
    model.eval()

    # Lists to store results
    all_predictions = []
    all_labels = []
    all_softmax_probs = []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Get the logits (raw outputs)
            logits = model(inputs)

            # Apply softmax to get probabilities
            softmax_probs = torch.softmax(logits, dim=1)

            # Get the predictions
            _, predicted = torch.max(logits, 1)

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_softmax_probs.extend(softmax_probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # Convert to numpy arrays for convenience
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_softmax_probs = np.array(all_softmax_probs)
    all_logits = np.array(all_logits)

    return all_predictions, all_labels, all_softmax_probs, all_logits

# ============================================================================
# 4. METRIC COMPUTATION FUNCTIONS
# ============================================================================


def calculate_confidence_scores(softmax_probs):
    """
    Compute confidence scores as the maximum probability
    """
    confidence_scores = np.max(softmax_probs, axis=1)
    return confidence_scores


def calculate_top_k_accuracy(softmax_probs, true_labels, k=5):
    """
    Compute Top-K accuracy
    """
    acc = top_k_accuracy_score(true_labels, softmax_probs, k=k)
    return acc * 100.0


def calculate_entropy(softmax_probs):
    """
    Compute entropy as a measure of uncertainty.
    H = -sum(p_i * log(p_i))
    """
    # Adding a small value to avoid log(0)
    ent = scipy_entropy(softmax_probs, axis=1)
    return ent


def calculate_calibration_error(softmax_probs, predictions, true_labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    Measures how well the model's confidence matches its actual accuracy.
    """
    probs_t = torch.from_numpy(softmax_probs).float()      # shape: [N, C]
    targets_t = torch.from_numpy(true_labels).long()       # shape: [N]

    # ECE with L1 norm (|conf - acc|), as before
    ece_metric = MulticlassCalibrationError(
        num_classes=softmax_probs.shape[1],
        n_bins=n_bins,
        norm='l1'
    )
    ece = float(ece_metric(probs_t, targets_t).item())

    # bin_data is not needed - return an empty list to keep the signature stable
    return ece, []

# ============================================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================================


def plot_training_history(training_history):
    """
    Plot training and validation curves (with range bands)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- unified style
    FIGSIZE = (12, 5)
    GRID_ALPHA = 0.3
    BAND_ALPHA = 0.15
    LINEWIDTH = 2

    train_accuracies = np.asarray(
        training_history['train_accuracies'], dtype=float)
    val_accuracies = np.asarray(
        training_history['val_accuracies'],   dtype=float)
    train_losses = np.asarray(
        training_history['train_losses'],     dtype=float)
    val_losses = np.asarray(training_history['val_losses'],       dtype=float)
    epochs = np.arange(1, len(train_accuracies) + 1)

    def _smooth(x, k=3):
        if len(x) < k:
            return x
        w = np.ones(k)/k
        y = np.convolve(x, w, mode='valid')
        pad = (len(x) - len(y)) // 2
        return np.pad(y, (pad, len(x)-len(y)-pad), mode='edge')

    def _roll_minmax(x, k=5):
        if k < 2 or len(x) < k:
            return x, x
        from collections import deque
        dmin, dmax, qmin, qmax, xs = [], [], deque(), deque(), x.tolist()
        for i, v in enumerate(xs):
            while qmin and xs[qmin[-1]] >= v:
                qmin.pop()
            qmin.append(i)
            while qmax and xs[qmax[-1]] <= v:
                qmax.pop()
            qmax.append(i)
            if qmin[0] <= i-k:
                qmin.popleft()
            if qmax[0] <= i-k:
                qmax.popleft()
            dmin.append(xs[qmin[0]])
            dmax.append(xs[qmax[0]])
        return np.array(dmin), np.array(dmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # --- Accuracy
    tr_s = _smooth(train_accuracies, k=3)
    va_s = _smooth(val_accuracies,   k=3)
    ax1.plot(epochs, tr_s, linewidth=LINEWIDTH, label='Training')
    ax1.plot(epochs, va_s, linewidth=LINEWIDTH, label='Validation')
    tr_lo, tr_hi = _roll_minmax(train_accuracies, k=5)
    va_lo, va_hi = _roll_minmax(val_accuracies,   k=5)
    ax1.fill_between(epochs, tr_lo, tr_hi, alpha=BAND_ALPHA)
    ax1.fill_between(epochs, va_lo, va_hi, alpha=BAND_ALPHA)
    ax1.set_title('Accuracy per epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc='lower right')

    # --- Loss
    tl_s = _smooth(train_losses, k=3)
    vl_s = _smooth(val_losses,   k=3)
    ax2.plot(epochs, tl_s, linewidth=LINEWIDTH, label='Training')
    ax2.plot(epochs, vl_s, linewidth=LINEWIDTH, label='Validation')
    tl_lo, tl_hi = _roll_minmax(train_losses, k=5)
    vl_lo, vl_hi = _roll_minmax(val_losses,   k=5)
    ax2.fill_between(epochs, tl_lo, tl_hi, alpha=BAND_ALPHA)
    ax2.fill_between(epochs, vl_lo, vl_hi, alpha=BAND_ALPHA)
    ax2.set_title('Loss per epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- Overfitting analysis (unchanged)
    print("\n🔍 OVERFITTING ANALYSIS:")
    final_train_acc = float(train_accuracies[-1])
    final_val_acc = float(val_accuracies[-1])
    gap = final_train_acc - final_val_acc
    print(f"Final training accuracy: {final_train_acc:.2f}%")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Gap (Train - Val): {gap:.2f}%")
    if gap < 3:
        print("✅ Virtually no overfitting")
    elif gap < 8:
        print("⚠️ Mild overfitting")
    elif gap < 15:
        print("🔶 Moderate overfitting")
    else:
        print("❌ Strong overfitting")
    return gap


def plot_confusion_matrix(true_labels, predictions, classes):
    """
    Plot the confusion matrix (percentages + absolute counts, normalized by true class)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    FIGSIZE = (8, 6)
    GRID_ALPHA = 0.3

    cm_abs = confusion_matrix(true_labels, predictions)
    with np.errstate(invalid='ignore'):
        cm = cm_abs / cm_abs.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm)

    plt.figure(figsize=FIGSIZE)
    ax = sns.heatmap(cm, annot=False, cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     vmin=0.0, vmax=1.0, cbar_kws={"label": "Share per true class"})
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]*100:.1f}%\n({cm_abs[i, j]})"
            ax.text(j+0.5, i+0.5, txt, ha='center',
                    va='center', fontsize=9, color='black')

    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.grid(False)  # heatmap has its own grid
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. MAIN FUNCTION - COMPLETE PIPELINE
# ============================================================================


def main():
    """
    Main function demonstrating the full training and evaluation pipeline
    """
    print("=" * 80)
    print("PRACTICE: DEEP LEARNING CLASSIFICATION AND METRICS")
    print("=" * 80)

    # 1. Load the data
    print("\n1. Loading the CIFAR-10 data...")
    train_loader, val_loader, test_loader, classes = load_cifar10_data(
        batch_size=128, val_split=0.1)
    print(f"Classes: {classes}")

    # 2. Build the model
    print("\n2. Building the model...")
    model = SimpleCNN_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count the parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # 3. Train the model
    print("\n3. Training the model with validation...")
    print("=" * 60)
    training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10
    )

    print(f"\n🎯 TRAINING SUMMARY:")
    print(
        f"Best validation accuracy: {training_history['best_val_accuracy']:.2f}%")

    # 4. Final evaluation on the test set
    print("\n4. Final evaluation on the test set...")
    print("⚠️  IMPORTANT: The test set is used ONLY for the final evaluation!")
    print("🔒 This set was NOT used during training or validation")
    predictions, true_labels, softmax_probs, logits = evaluate_model_with_metrics(
        model, test_loader, device, classes
    )

    # Basic accuracy on the test set
    test_accuracy = accuracy_score(true_labels, predictions) * 100
    val_accuracy = training_history['best_val_accuracy']

    print(f"\n📊 VALIDATION VS TEST COMPARISON:")
    print(f"Best validation accuracy: {val_accuracy:.2f}%")
    print(f"Final test accuracy:      {test_accuracy:.2f}%")

    accuracy_diff = abs(val_accuracy - test_accuracy)
    if accuracy_diff < 2:
        print(f"✅ Excellent agreement (difference: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 5:
        print(f"✅ Good agreement (difference: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 10:
        print(f"⚠️  Moderate discrepancy (difference: {accuracy_diff:.2f}%)")
    else:
        print(f"❌ Significant discrepancy (difference: {accuracy_diff:.2f}%)")
        print("   Possible causes: overfitting on the validation set, data distribution shift")

    # ========================================================================
    # 5. METRIC COMPUTATION AND ANALYSIS
    # ========================================================================

    print("\n" + "=" * 50)
    print("FINAL METRICS ON THE TEST SET")
    print("=" * 50)
    print("🔬 All metrics below are computed on the independent test set")

    # 5.1 Basic accuracy (Top-1 Accuracy)
    basic_accuracy = accuracy_score(true_labels, predictions) * 100
    print(f"\n📊 BASIC METRICS:")
    print(f"Top-1 Accuracy: {basic_accuracy:.2f}%")

    # 5.2 Confidence Scores
    confidence_scores = calculate_confidence_scores(softmax_probs)
    avg_confidence = np.mean(confidence_scores)
    print(f"Average Confidence Score: {avg_confidence:.3f}")
    print(f"Min. Confidence Score: {np.min(confidence_scores):.3f}")
    print(f"Max. Confidence Score: {np.max(confidence_scores):.3f}")

    # 5.3 Top-K Accuracy
    print(f"\n🎯 TOP-K ACCURACY:")
    for k in [1, 3, 5]:
        top_k_acc = calculate_top_k_accuracy(softmax_probs, true_labels, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}%")

    # 5.4 Entropy
    entropy_values = calculate_entropy(softmax_probs)
    avg_entropy = np.mean(entropy_values)
    print(f"\n🔀 ENTROPY (uncertainty):")
    print(f"Average entropy: {avg_entropy:.3f}")
    print(f"Min. entropy: {np.min(entropy_values):.3f}")
    print(f"Max. entropy: {np.max(entropy_values):.3f}")
    print(f"Standard deviation: {np.std(entropy_values):.3f}")

    # 5.5 Calibration
    ece, bin_data = calculate_calibration_error(
        softmax_probs, predictions, true_labels)
    print(f"\n⚖️ CALIBRATION:")
    print(
        f"Expected Calibration Error (ECE), the difference between stated confidence and actual accuracy: {ece:.3f}")
    print(f"How to interpret: if the model says 'I'm 80% sure', it is right in about 76-80% of those cases (almost perfect).")
    print("ECE interpretation:")
    print("  - 0.0-0.05: Excellently calibrated")
    print("  - 0.05-0.1: Well calibrated")
    print("  - 0.1-0.2: Moderately calibrated")
    print("  - >0.2: Poorly calibrated")

    # 5.6 Per-class analysis
    print(f"\n📋 DETAILED PER-CLASS REPORT:")
    class_report = classification_report(true_labels, predictions,
                                         target_names=classes, digits=3)
    print(class_report)

    # ========================================================================
    # 6. VISUALIZATION OF RESULTS
    # ========================================================================

    print("\n" + "=" * 50)
    print("VISUALIZATION OF RESULTS")
    print("=" * 50)

    # Training plot with validation
    overfitting_gap = plot_training_history(training_history)

    # Confusion matrix
    plot_confusion_matrix(true_labels, predictions, classes)

# ============================================================================
# 8. ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()

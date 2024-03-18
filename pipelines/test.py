import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from components.data_loader2 import data_loaders
from sklearn.metrics import confusion_matrix, accuracy_score
from model import PointTransformerModel
import warnings
warnings.filterwarnings("ignore")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model.eval()

def test(model, device, test_loader):
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in test_loader:
            inputs,labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predicted_labels, true_labels

def compute_class_accuracy(predicted_labels, true_labels, num_classes):
    class_accuracies = {}
    for cls in range(num_classes):
        cls_indices = np.where(np.array(true_labels) == cls)[0]
        cls_correct = np.sum(np.array(predicted_labels)[cls_indices] == cls)
        cls_accuracy = cls_correct / len(cls_indices) if len(cls_indices) > 0 else 0
        class_accuracies[cls] = cls_accuracy
    return class_accuracies

def generate_confusion_matrix(predicted_labels, true_labels, num_classes):
    return confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))

def plot_confusion_matrix(conf_matrix, num_classes):
    plt.figure(figsize=(30, 30))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    
if __name__=='__main__':
    batch_size =16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model_path = "best/best_model.pt"
    model = PointTransformerModel(input_dim=6).to(device)
    model = load_model(model, model_path)
    train_loader, test_loader = data_loaders(ROOT_DIR="./data/modelnet40_normal_resampled/",batch_size=batch_size)
    predicted_labels, true_labels = test(model, device, test_loader)
    class_accuracies = compute_class_accuracy(predicted_labels, true_labels, num_classes=40)
    conf_matrix = generate_confusion_matrix(predicted_labels, true_labels, num_classes=40)
    overall_accuracy = accuracy_score(true_labels, predicted_labels)

    # Print results
    print("Overall accuracy:", overall_accuracy)
    print("Class accuracies:", class_accuracies)
    plot_confusion_matrix(conf_matrix, num_classes=40)
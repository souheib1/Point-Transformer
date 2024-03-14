import torch
import time
from tqdm import tqdm
from components.loss import basic_loss
from model import PointTransformerModel
from components.data_loader import data_loaders
import matplotlib.pyplot as plt

def train(model, device, train_loader, test_loader=None, epochs=250, model_path='models/best_model.pt'):
    if model is None:
        model = PointTransformerModel(input_dim=3).to(device)
    if train_loader is None:
        train_loader, test_loader = data_loaders(batch_size=16)   
    
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_test_acc = 0
    model_path = model_path

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in tqdm(range(epochs)): 
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = basic_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accs.append(100. * correct_train / total_train)

        scheduler.step()

        model.eval()
        epoch_test_loss = 0.0
        correct_test = 0
        total_test = 0

        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs = model(inputs)
                    loss = basic_loss(outputs, labels)
                    epoch_test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_test += labels.size(0)
                    correct_test += predicted.eq(labels).sum().item()

            test_losses.append(epoch_test_loss / len(test_loader))
            test_accs.append(100. * correct_test / total_test)
            
            if test_accs[-1] > best_test_acc:
                best_test_acc = test_accs[-1]
                torch.save(model.state_dict(), model_path)

        print('Epoch: %d, Train Loss: %.3f, Train Acc: %.3f, Test Loss: %.3f, Test Acc: %.3f' % (
            epoch + 1, train_losses[-1], train_accs[-1], test_losses[-1], test_accs[-1]))

    print("Best test accuracy: ", best_test_acc)

    # Plotting train/test loss and train/test accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train/Test Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train/Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./loss_accuracy_plot.png')
    plt.show()
    
if __name__ == "__main__":
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    batch_size = 16
    print("batch_size=",batch_size)
    model = PointTransformerModel(input_dim=3).to(device)
    t0 = time.time()
    train(model, device, train_loader=None, test_loader=None, epochs=250, model_path='models/best_model.pt')
    print('training time',((time.time()-t0)//60),' minutes' )
from model import AlexNet
import torch
import torch.nn as nn
import torch.optim as optim


def cal(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    # argmax():Returns the indices of the maximum value of all elements in the input tensor.
    correct = (preds == labels).sum().item()
    # item():Returns the value of this tensor as a standard Python number.
    return correct


class NetSolver:
    def __init__(self, class_number=1000, epoch_print=10, model_save=True):
        self.class_number = class_number
        self.epoch_print = epoch_print
        self.model_save = model_save
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = AlexNet(class_number=self.class_number).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.train_losses = []
        self.train_acc = []
        self.valid_losses = []
        self.valid_acc = []
        self.best_acc = 0

    def train(self, trainsets, validsets, epochs, batch_size=64, learning_rate=0.01, momentum=0, weight_decay=0):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)
        for epoch in range(epochs):
            correct = 0
            total = 0
            train_loss = 0
            for i, (X, Y) in enumerate(trainsets):
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                correct += cal(outputs, Y)
                total += Y.size(0)

            train_acc = 100.*correct/total
            self.train_acc.append(train_acc)
            self.train_losses.append(train_loss)
            val_acc, val_loss = self.validate(validsets)
            self.valid_acc.append(val_acc)
            self.valid_losses.append(val_loss)
            if (epoch+1) % self.epoch_print == 0:
                print(
                    f'epoch {epoch}/{epochs}:train acc = {train_acc} , train_loss = {train_loss}')
            if train_acc > self.best_acc:
                self.best_acc = train_acc
                if self.model_save:
                    torch.save(self.model.state_dict(), './BestAlexModel.pt')
                    print('Saved best model,best_acc = %f' % self.best_acc)

    def validate(self, validsets):
        # Sets the module in evaluation mode.
        self.model.eval()
        val_losses = []
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (X, Y) in enumerate(validsets):
                X = X.to(self.device)
                Y = Y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, Y)
                val_losses.append(loss.item())
                correct += cal(outputs, Y)
                total += Y.size(0)
        # Sets the module in training mode.
        self.model.train()
        return (100*correct/total, sum(val_losses))


if __name__ == '__main__':
    a=NetSolver(class_number=10)


import shutil
epochs = 100
valid_loss_min = np.Inf
iter = 0
path = 'drive/Shared drives/Bangla-Handwritten/models'
train_loss_data, valid_loss_data = [], []
for e in range(epochs):
    running_loss = 0
    train_loss = 0.0
    valid_loss = 0.0
    total = 0
    correct = 0
    print("Epoch:", e+1)
    if torch.cuda.is_available():
      print('Started Running')
      for images, labels in train_loader:
          # Move input and label tensors to the default device
          images = images.to(device)
          labels = labels.to(device)

          #images  = Variable(images.view(-1, 28, 28))
          #labels = Variable(labels)
          # clear the gradients of all optimized variables
          optimizer.zero_grad()
          # forward pass: compute predicted outputs by passing inputs to the model
          log_ps = model(images)
          # calculate the loss
          loss = criterion(log_ps, labels)
          # backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()
          # perform a single optimization step (parameter update)
          optimizer.step()
          # update running training loss
          train_loss += loss.item() * images.size(0)
          
          
      for data, target in test_loader:
          #data, target = Variable(data.view(100, 1, 28, 28)).to(device), target.to(device)
          data = data.to(device)
          target = target.to(device)
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the loss
          loss_p = criterion(output, target)
          # update running validation loss
          valid_loss += loss_p.item() * data.size(0)
          # calculate accuracy
          proba = torch.exp(output)
          top_p, top_class = proba.topk(1, dim=1)
          equals = top_class == target.view(*top_class.shape)
          # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

          _, predicted = torch.max(output.data, 1)
          total += target.size(0)
          correct += (predicted == target).sum().item()

      # print training/validation statistics
      # calculate average loss over an epoch
      train_loss = train_loss / len(train_loader.dataset)
      valid_loss = valid_loss / len(test_loader.dataset)

      # calculate train loss and running loss
      train_loss_data.append(train_loss * 100)
      valid_loss_data.append(valid_loss * 100)

      accuracy = (correct / total) * 100

      print("\tTrain loss:{:.6f}..".format(train_loss),
            "\tValid Loss:{:.6f}..".format(valid_loss),
            "\tAccuracy: {:.4f}".format(accuracy))
    
      if valid_loss <= valid_loss_min:
              print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                  valid_loss_min,
                  valid_loss))
              torch.save(model.state_dict(), 'model_cnn_aug.pt')
              valid_loss_min = valid_loss
              shutil.copy('model_cnn_aug.pt', path)
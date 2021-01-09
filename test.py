transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
path_img = 'drive/Shared drives/Bangla-Handwritten/test/test34.png'

image = Image.open(path_img).convert('L')
plt.imshow(image)
image = transform(image).unsqueeze(0)
image = image.to(device)

output = model(image)
_, predicted = torch.max(output.data, 1)
print(predicted.item())

for images, labels in test_loader:
    break
    
fig, ax = plt.subplots(1, 5)
for i in range(5):
    images[i] = (255 - images[i]) / 255.
    ax[i].imshow(images[i].view(28, 28))

plt.show()

predictions = model(images.to(device))
predictions = torch.argmax(predictions, dim=1)
print('Predicted labels', predictions.cpu().numpy())
# Import the libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from clip import load, tokenize
import utils as ut
import os
from tqdm import tqdm

device = "cuda"
# Load the CLIP model and the device
model, preprocess = load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation
# # transforms.Resize(224,224),
# transform = transforms.Compose([    
#     transforms.ToTensor(),
#     preprocess
# ])

# Define the dataset class
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, texts, images, labels):
        self.texts = texts # a list of text strings
        self.images = images # a list of image paths
        self.labels = labels # a list of binary labels (0 or 1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Load and transform the image
        image = Image.open(self.images[index])
        # image = transform(image)
        image = preprocess(image)

        # Tokenize the text
        text = tokenize(self.texts[index], truncate=True)

        # Get the label
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return image, text, label

# Define the network class
class TextImageClassifier(nn.Module):
    def __init__(self):
        super(TextImageClassifier, self).__init__()
        # Get the embedding size from the CLIP model
        self.embedding_size = model.visual.output_dim

        # Define the feed forward network with 3 layers
        self.fc1 = nn.Linear(self.embedding_size * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2) # binary classification

    def forward(self, image, text):
        # Get the image and text embeddings from the CLIP model
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        # Concatenate the embeddings
        features = torch.cat([image_features, text_features], dim=1).to(torch.float32)

        # Pass through the feed forward network
        output = self.fc1(features)
        output = torch.relu(output)
        output = self.fc2(output)
        output = torch.relu(output)
        output = self.fc3(output)

        return output





# Define the training loop
def train(dataset_train,dataset_test, epochs):
    # Create a data loader
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train.texts), shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test.texts), shuffle=False)
    # Create an instance of the network
    classifier = TextImageClassifier().to("cuda")
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Loop over the epochs
    for epoch in range(epochs):
        classifier.train()
        # Initialize the loss and accuracy
        epoch_loss_train = 0.0
        epoch_acc_train = 0.0
        epoch_loss_test = 0.0
        epoch_acc_test = 0.0
        best_acc = 0.0

        # Loop over the batches
        with tqdm(loader_train, unit="batch") as toder:
            for batch, Data in enumerate(toder):
                image,text,label = Data
                toder.set_description(f"Epoch: {epoch}, Batch: {batch}  ")
                # Move the data to the device
                image = image.to(device)
                text = text.squeeze(1).to(device)
                label = label.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = classifier(image, text)

                # Compute the loss and accuracy
                loss = criterion(output, label)
                acc = (output.argmax(dim=1) == label).float().mean()

                # Backward pass and update
                loss.backward()
                optimizer.step()

                # Update the loss and accuracy
                epoch_loss_train += loss.item()
                epoch_acc_train += acc.item()

                toder.set_postfix_str(f"loss: {loss.item()} accuracy: {acc.item()}")
        # Evaluate the model 

        classifier.eval()
        with tqdm(loader_test, unit="batch") as toder_test:
            for batch,Data in enumerate(toder_test):
                image,text,label = Data
                toder_test.set_description(f"Batch: {batch}   ")
                image = image.to(device)
                text = text.squeeze(1).to(device)
                label = label.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.no_grad():
                    output = classifier(image, text)
                    # Compute the loss and accuracy
                    loss = criterion(output, label)
                    acc = (output.argmax(dim=1) == label).float().mean()           

                # Update the loss and accuracy
                epoch_loss_test += loss.item()
                epoch_acc_test += acc.item()
                
                toder_test.set_postfix_str(f"loss: {loss.item()} accuracy: {acc.item()}")
        # Print the average loss and accuracy for the epoch
        
        if (epoch_acc_test/len(loader_test)) > best_acc:
            print(f"Saving best model, epoch = {epoch}")
            best_acc = (epoch_acc_test/len(loader_test))
            torch.save(classifier.state_dict(), "./DL_A4/models/best_model.pt")

        print(f"Epoch {epoch+1}, Loss_train: {epoch_loss_train/len(loader_train):.4f}, Accuracy_train: {epoch_acc_train/len(loader_train):.4f}, Loss_test: {epoch_loss_test/len(loader_test):.4f}, Accuracy_test: {epoch_acc_test/len(loader_test):.4f}")

# Create some dummy data for testing (you should use your own data here)
if __name__ == "__main__":

    print(os.getcwd())
    file_train = "./DL_A4/Data/Texts/train.jsonl"
    file_test = "./DL_A4/Data/Texts/dev_unseen.jsonl"
    df_train = ut.JsonlToDf(file_train)
    df_test = ut.JsonlToDf(file_test)
    train_texts = df_train["text"].tolist()
    test_texts = df_test["text"].tolist()
    images_train = ["./DL_A4/Data/" + str(img) for img in df_train["img"].tolist()]
    images_test = ["./DL_A4/Data/" + str(img) for img in df_test["img"].tolist()]
    labels_train = df_train["label"].tolist()
    labels_test = df_test["label"].tolist()

    # Create a dataset object
    dataset_train = TextImageDataset(train_texts, images_train, labels_train)
    dataset_test = TextImageDataset(test_texts, images_test, labels_test)

    # Train the network for 10 epochs (you can change this as needed)
    train(dataset_train,dataset_test, 10) 

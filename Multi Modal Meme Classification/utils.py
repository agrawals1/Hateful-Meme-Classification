import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torch
from clip import load, tokenize
import torch.nn as nn
from tqdm import tqdm   
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report


device = 'cuda'
model, preprocess = load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

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
    
  

def Evaluate(classifier, dataset_test):
    classifier.to('cuda')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test.texts), shuffle=False)  
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
                y_pred = output.argmax(dim=1)
                # Compute the loss and accuracy
                loss = criterion(output, label)
                acc = (output.argmax(dim=1) == label).float().mean()
                target_names = ['Not-Hateful', 'Hateful']
                print(classification_report(y_true=label.to("cpu"), y_pred=y_pred.to("cpu"), target_names=target_names))           

            # Update the loss and accuracy
            # epoch_loss_test += loss.item()
            # epoch_acc_test += acc.item()
            
            toder_test.set_postfix_str(f"loss: {loss.item()} accuracy: {acc.item()}")

def JsonlToDf(file):
    # with open(file) as f:
    #     data = json.load(f)
    # df = pd.DataFrame(data)
    df = pd.read_json(file, lines=True)

    return df







    


if __name__ == "__main__":

    file_test = "./DL_A4/Data/Texts/dev_unseen.jsonl"
    df_test = JsonlToDf(file_test)
    test_texts = df_test["text"].tolist()
    
    labels_test = df_test["label"].tolist()
    images_test = ["./DL_A4/Data/" + str(img) for img in df_test["img"].tolist()]
    dataset_test = TextImageDataset(test_texts, images_test, labels_test) 
    classifier = TextImageClassifier()
    classifier.load_state_dict(torch.load("./DL_A4/models/best_model.pt"))
    classifier.eval() 
    Evaluate(classifier, dataset_test)







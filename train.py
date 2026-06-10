import io
import os
import argparse
import boto3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. CLOUD DATASET EXTRACTOR
class AWSSkinCancerDataset(Dataset):
    def __init__(self, bucket_name, metadata_key, transform=None):
        self.bucket_name = bucket_name
        self.transform = transform
        self.s3_client = boto3.client('s3')
        
        csv_obj = self.s3_client.get_object(Bucket=bucket_name, Key=metadata_key)
        self.df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        s3_image_key = f"raw-images/{image_id}.jpg"
        
        image_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_image_key)
        image_bytes = image_obj['Body'].read()
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, image

# 2. THE MODEL ARCHITECTURE
class SkinCancerAutoencoder(nn.Module):
    def __init__(self):
        super(SkinCancerAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# 3. TRAINING MANIFEST EXECUTOR
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bucket', type=str, default="skin-cancer-imaging-pipeline-ankush")
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    # Hardware acceleration check (SageMaker GPU pass-through)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training cluster initialized. Computational Core: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = AWSSkinCancerDataset(bucket_name=args.bucket, metadata_key="metadata/HAM10000_metadata.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SkinCancerAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}] complete. Average Structural Loss: {total_loss / len(dataloader):.6f}")

    # Save model artifacts for downstream SVM deployment
    torch.save(model.state_dict(), os.path.join(args.model_dir, "autoencoder.pth"))
    print("Model weight vectors successfully compiled and saved.")

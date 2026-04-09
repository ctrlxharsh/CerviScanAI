import torch
import torch.nn as nn
from fastai.vision.all import *
import timm

# 1. Setup & Hyperparameters
batch_size = 8 # Reduced slightly as fusion models are memory-intensive
epochs = 100
lr = 1e-4

path_img = Path("/home/pglabns04/Downloads/deep-cervical-cancer-master/sipakmed_formatted")

# 2. Loading Data
data = ImageDataLoaders.from_folder(
    path=path_img, 
    train='train',
    valid='val', 
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)],
    bs=batch_size
)

# 3. Defining the Fusion Model
class CervicalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. CNN Branch: DenseNet121
        # Using num_classes=0 removes the classification head to get raw features
        self.cnn_branch = timm.create_model('densenet121', pretrained=True, num_classes=0)
        self.cnn_features_dim = 1024 # Feature dimension for DenseNet121
        
        # 2. ViT Branch: ViT Tiny
        # Using num_classes=0 removes the head to get the global representation
        self.vit_branch = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.vit_features_dim = 192 # Feature dimension for ViT-Tiny
        
        # 3. Concatenated Dimension
        self.fusion_dim = self.cnn_features_dim + self.vit_features_dim
        
        # 4. MLP Classification Head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        cnn_feat = self.cnn_branch(x) # Output shape: [batch, 1024]
        vit_feat = self.vit_branch(x) # Output shape: [batch, 192]
        
        # Concatenate along dimension 1 (features)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        
        # Final prediction
        return self.mlp_head(combined)

# Initialize the model
model = CervicalFusionModel(num_classes=data.c)

# 4. Create Learner
# Since it's a custom model, we use the standard Learner class
learn = Learner(
    data, 
    model, 
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy, FBeta(beta=1, average="weighted")]
)

# 5. Training
save_loc = f'fusion_densenet_vit_sipakmed'
cbs = [SaveModelCallback(monitor='accuracy', fname=save_loc)]

# We use a lower LR for the backbone and train
learn.fit_one_cycle(epochs, lr_max=lr, cbs=cbs)

# 6. Evaluation
print("Validation Results:")
print(learn.validate())
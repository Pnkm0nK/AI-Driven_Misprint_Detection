from timm import create_model
import torch

if __name__ == "__main__":
    model_name = "resnet18"
    model = create_model(model_name, pretrained=True)
    model.eval()


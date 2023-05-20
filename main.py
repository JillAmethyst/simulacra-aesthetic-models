import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from CLIP import clip
import requests
from io import BytesIO

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

clip_model_name = 'ViT-B/16'
clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# 512 is embed dimension for ViT-B/16 CLIP
model = AestheticMeanPredictionLinearModel(512)
model.load_state_dict(
    torch.load("models/sac_public_2022_06_29_vit_b_16_linear.pth")
)
model = model.to(device)

app = FastAPI()

@app.get("/ai3/{score}")
async def main(score:str, path:str):
    # return {"score":score, "path":path}
    response = requests.get(path)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224,224))
    img = TF.to_tensor(img).to(device)
    img = normalize(img)
    clip_image_embed = F.normalize(
        clip_model.encode_image(img[None, ...]).float(),
        dim=-1)
    score = model(clip_image_embed)
    return {"score": score.item()}

if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, reload=True, debug=True)

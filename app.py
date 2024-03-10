import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='/clip/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin')
tokenizer = open_clip.get_tokenizer('ViT-L-14')

image = preprocess(Image.open("/images/lambo.jpg")).unsqueeze(0)
text = tokenizer(["a car", "a lamborghini", "a hot dog"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
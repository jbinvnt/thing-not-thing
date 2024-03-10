import torch
from PIL import Image
import open_clip
import gradio as gr

def is_thing(image, name_of_thing, other_class, threshold): # With the dfeault example, softmax is computed over classes so w/ just 1 class the output is always 1.00
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='/clip/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    image = preprocess(image).unsqueeze(0)
    text = tokenizer([name_of_thing, other_class])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    return ("Not " if text_probs.data.tolist()[0][0] < threshold else "") + name_of_thing + " (probability = {})".format(text_probs.data.tolist()[0][0])

demo = gr.Interface(fn=is_thing, inputs=[gr.Image(type="pil"), "text", "text", gr.Slider(0, 1, value=0.25)], outputs=["text", ])
demo.launch()
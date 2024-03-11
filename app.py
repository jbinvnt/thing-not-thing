import torch
from PIL import Image
import open_clip
import gradio as gr
import os

use_gpu = False
load_local_model = False

if "RUN_MODE" in os.environ:
    run_mode = os.environ["RUN_MODE"]
    use_gpu = "gpu" in run_mode
    load_local_model = "local" in run_mode

def is_thing(image, name_of_thing, threshold, output_similarity):
    model_path = "/clip/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin" if load_local_model else "laion2B_s32B_b82K"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=model_path)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    image = preprocess(image).unsqueeze(0)
    text = tokenizer([name_of_thing, ])

    with torch.no_grad(), torch.cuda.amp.autocast() if use_gpu else torch.cpu.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity_output = image_features @ text_features.T
    similarity = similarity_output.data.tolist()[0][0]
    return ("Not " if similarity < threshold else "") + name_of_thing + "." + (" Similarity = {}.".format(similarity) if output_similarity else "")

demo = gr.Interface(fn=is_thing, inputs=[gr.Image(type="pil"), gr.Textbox(label="Is this a...")], additional_inputs=[gr.Slider(0, 1, value=0.25), gr.Checkbox(label="Output the similarity score")], outputs=["text", ])
demo.launch()
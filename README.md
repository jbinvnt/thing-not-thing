# Thing/Not Thing
This is a zero-shot binary image classifier. Type in the name of an object and AI predicts whether your uploaded photo matches it.

## How it Works

This project uses [OpenCLIP](https://github.com/mlfoundations/open_clip), an open-source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020).

## Examples

![Hot Dog Detected](/readme-images/HotDog.png?raw=true)

![Hot Dog Not Detected](/readme-images/NotHotDog.png?raw=true)

## Try It Yourself!

This project is available as a web demo [here](https://huggingface.co/spaces/jbinvnt/thing-not-thing). But it will be slower than when the project is run locally on a GPU.

You can expand "Additional Inputs" to allow adjusting the cosine similarity threshold below which your photo is deemed `Not <object>.`

## Running Locally on a GPU

Tested on [Debian](https://debian.org).

### Requirements

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/) (and Docker Compose)
- An NVIDIA GPU with sufficient VRAM for your chosen ViT model (model size can be changed in [app.py](app.py))

### Startup

Create a `.env` file which points to the path where you downloaded the OpenCLIP model. Then run:

```bash
docker compose build
docker compose run torch
```

Finally, go to `http://localhost:7860` in your browser.
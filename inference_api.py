import base64
import io
from functools import lru_cache
from pathlib import Path

import hydra
import torch
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms as T
from optree import tree_map

from mode.models.mode_agent import MoDEAgent

app = FastAPI()


class StepRequest(BaseModel):
    image: str
    instr: str


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_mode_from_safetensor(
    filedir: Path,
    overwrite_cfg: dict = {},
):
    if not filedir.is_dir():
        raise ValueError(f"not valid file path: {str(filedir)}")
    
    config = get_config_from_dir(filedir)
    ckpt_path = filedir

    print(f"Loading model from {ckpt_path}")
    load_cfg = OmegaConf.create({**OmegaConf.to_object(config.model), **{"optimizer": None}, **overwrite_cfg})
    load_cfg["ckpt_path"] = str(ckpt_path)

    model = hydra.utils.instantiate(load_cfg)

    print(f"Finished loading model {ckpt_path}")
    return model


@lru_cache
def get_agent(device: str) -> MoDEAgent:
    model = load_mode_from_safetensor(
        Path("./MoDE_Pretrained"),
    )
    model.eval()
    model.freeze()
    model.to(device)

    return model


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/reset")
def model_reset():
    agent = get_agent("cuda")
    agent.reset()
    return {"message": "Done"}


@app.post("/step")
def model_step(step: StepRequest):
    agent = get_agent("cuda")

    image_bytes = base64.b64decode(step.image)
    image = Image.open(io.BytesIO(image_bytes), formats=["JPEG"])

    trans = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    image = trans(image)
    image = image[None, None, ...]

    obs = {
        "rgb_obs": {
            "rgb_static": image,
            "rgb_gripper": image,
        }
    }

    obs = tree_map(lambda x: x.to(agent.device), obs)

    with torch.no_grad():
        action = agent.step(
            obs,
            {"lang_text": step.instr},
        )
    return {"action": action.cpu().detach().numpy().tolist()}


def main():
    # Create a white image
    image = Image.new("RGB", (224, 224), (255, 255, 255))

    # Save to io.Bytes with JPEG format
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Encode to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    model_reset()

    res = model_step(
        StepRequest(
            image=image_base64,
            instr="Move the box to the left",
        )
    )
    print(res)


if __name__ == "__main__":
    main()

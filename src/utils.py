from __future__ import annotations

import asyncio
import base64
import io
from functools import wraps
from typing import Awaitable, Callable, Literal, TypeVar, Union

import numpy as np
import torch
from httpx import AsyncClient, get
from openai import AsyncOpenAI
from PIL import Image
from typing_extensions import ParamSpec, TypeAlias

T = TypeVar("T")
P = ParamSpec("P")

ai = AsyncOpenAI()


def asyncify(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


ImageStyle: TypeAlias = Literal["vivid", "natural"]
ImageSize: TypeAlias = Literal[
    "256x256", "512x512", "1024x1024", "1024x1728", "1728x1024"
]
ImageQuality: TypeAlias = Literal["hd", "standard"]


@asyncify
def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(image))  # type: ignore
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float16)
        / 255.0
    )


@asyncify
def parse_byte_str_to_image(data: Union[str, bytes]) -> Image.Image:
    if isinstance(data, str):
        if data.startswith("http"):
            image_content = get(data).content
        else:
            image_content = base64.b64decode(data)
    else:
        image_content = data
    return Image.open(io.BytesIO(image_content)).convert("RGB")


async def parse_image(
    image_data: Union[str, bytes], size: ImageSize, session: AsyncClient
) -> Image.Image:
    if isinstance(image_data, str):
        if image_data.startswith("http"):
            image_content = (await session.get(image_data)).content
        else:
            image_content = base64.b64decode(image_data)
    else:
        image_content = image_data
    image = Image.open(io.BytesIO(image_content)).convert("RGB")
    return image.resize(size=(int(size.split("x")[0]), int(size.split("x")[1])))


async def parse_mask(
    mask_data: Union[str, bytes], size: ImageSize, session: AsyncClient
) -> Image.Image:
    if isinstance(mask_data, str):
        if mask_data.startswith("http"):
            mask_content = (await session.get(mask_data)).content
        else:
            mask_content = base64.b64decode(mask_data)
    else:
        mask_content = mask_data
    mask_bounding_box = Image.open(io.BytesIO(mask_content)).convert("L")
    mask = Image.new("RGBA", mask_bounding_box.size, (255, 255, 255, 0))
    mask.paste(mask_bounding_box, (0, 0), mask_bounding_box)
    return mask.resize(size=(int(size.split("x")[0]), int(size.split("x")[1])))


async def refine_prompt(prompt: str, style: ImageStyle = "vivid") -> str:
    """
    Refine the given prompt for optimal Stable Diffusion image generation in the specified style (vivid or natural).
    Vivid: Generates hyper-real and dramatic images.
    Natural: Generates more natural and less hyper-real images.
    """
    instructions = f"""Refine this prompt for a more {style} styled Stable Diffusion image generation. Be more descriptive regarding the content. 
    For example: 
    'Colorful bird with turquoise beak in lush forest.' optimized to 'A vibrant and exotic bird in a lush forest setting, featuring a turquoise beak and adorned with an array of colorful feathers in red, purple, and green hues. The bird has intricate details, including beaded decorations around its neck and a feathered headdress. The background is a soft blur of green foliage, highlighting the bird's striking appearance. The image is highly detailed and vivid, with a focus on the intricate patterns and bright colors of the bird's plumage.'"""

    response = await ai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    content = response.choices[0].message.content
    assert isinstance(content, str)
    return content

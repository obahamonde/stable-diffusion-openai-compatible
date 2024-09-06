from __future__ import annotations

import asyncio
import base64
import io
from functools import wraps
from typing import Awaitable, Callable, Literal, TypeVar, Union

import numpy as np
import torch
from cachetools import TTLCache, cached
from httpx import AsyncClient, get
from openai import AsyncOpenAI
from PIL import Image
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
INSTRUCTIONS = """Analyze the user's input and generate a detailed, compelling prompt for Stable Diffusion. Capture the core intent of the request, whether it's for a logo, illustration, photograph, or any other type of image. Infer and include key details such as style, mood, color palette, lighting, and composition. If specific elements are not mentioned, use appropriate defaults that align with the overall concept. Incorporate relevant descriptors to enhance image quality and clarity. Aim to create a prompt that will produce a visually appealing result matching the user's vision. Be comprehensive yet concise, focusing on the most impactful aspects of the desired image. Avoid restrictive language and allow for creative interpretation within the bounds of the user's request. Craft the prompt to guide the AI towards generating an image that the user will find satisfying and engaging."""

ai = AsyncOpenAI()


def ttl_cache(func: Callable[P, T], *, maxsize: int = 128, ttl: int = 300):
    @cached(TTLCache[str, T](maxsize=maxsize, ttl=ttl))
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)

    return wrapper


def asyncify(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


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
    image_data: Union[str, bytes],
    size: Literal["256x256", "512x512", "1024x1024"],
    session: AsyncClient,
):
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
    mask_data: Union[str, bytes],
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
    session: AsyncClient,
):
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


async def refine_prompt(prompt: str) -> str:
    response = await ai.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    content = response.choices[0].message.content
    assert isinstance(content, str)
    return content

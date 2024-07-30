import os
from datetime import datetime
from typing import Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, computed_field
from typing_extensions import Required, TypeAlias, TypedDict

ImageStyle: TypeAlias = Literal["vivid", "natural"]
ImageSize: TypeAlias = Literal[
    "256x256", "512x512", "1024x1024", "1024x1728", "1728x1024"
]
ImageQuality: TypeAlias = Literal["hd", "standard"]
BUCKET_NAME = os.getenv("BUCKET_NAME")


class Request(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model: Literal["stabilityai/stable-diffusion-3-medium-diffusers"] = Field(
        default="stabilityai/stable-diffusion-3-medium-diffusers"
    )
    prompt: str = Field(default=...)
    quality: ImageQuality = Field(default="hd")
    response_format: Literal["url", "b64_json"] = Field(default="url")
    size: ImageSize = Field(
        default="1024x1024",
        description="The size of the generated images. Must be one of 256x256, 512x512, 1024x1024, 1792x1024, or 1024x1792.",
    )
    n: int = Field(
        default=1,
        description="The number of images to generate. Must be between 1 and 10",
    )
    style: ImageStyle = Field(
        default="vivid",
        description="The style of the generated images. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
    )
    image: Optional[Union[HttpUrl, str, bytes]] = Field(default=None)
    mask: Optional[Union[str, bytes]] = Field(default=None)
    strength: float = Field(default=0.8)
    guidance_scale: float = Field(default=7.5)
    negative_prompt: str = Field(
        default="blurry image, low quality image, body parts, deformed, grotesque, scary, gore"
    )

    @computed_field(return_type=int)
    def num_inference_steps(self):
        if self.quality == "hd":
            return 36
        return 18


class ImageUrl(TypedDict):
    url: Required[HttpUrl]
    refined_prompt: Optional[str]


class ImageB64(TypedDict):
    b64_json: Required[str]
    refined_prompt: Optional[str]


ImageGeneration: TypeAlias = Union[ImageUrl, ImageB64]


class Response(BaseModel):
    created: float = Field(default_factory=lambda: datetime.now().timestamp())
    data: list[ImageGeneration]
    revised_prompt: Optional[str] = Field(default=None)

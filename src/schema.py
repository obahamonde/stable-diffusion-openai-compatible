import os
from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl
from typing_extensions import Required, TypeAlias, TypedDict

BUCKET_NAME = os.getenv("BUCKET_NAME", "terabytes")


class ImageGenerationParams(BaseModel):
    prompt: str = Field(
        ...,
        max_length=4000,
        description="A text description of the desired image(s). Max length is 1000 characters for dall-e-2 and 4000 for dall-e-3.",
    )
    model: Literal["dall-e-2", "dall-e-3"] = Field(
        default="dall-e-2", description="The model to use for image generation."
    )
    n: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="The number of images to generate. Must be between 1 and 10. For dall-e-3, only n=1 is supported.",
    )
    quality: Optional[Literal["standard", "hd"]] = Field(
        default="standard",
        description="The quality of the image. 'hd' creates images with finer details.",
    )
    response_format: Optional[Literal["url", "b64_json"]] = Field(
        default="url",
        description="The format in which the generated images are returned. URLs are valid for 60 minutes.",
    )
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    ] = Field(
        default="1024x1024",
        description="The size of the generated images. '256x256', '512x512', '1024x1024' for dall-e-2. '1024x1024', '1792x1024', '1024x1792'.",
    )
    style: Optional[Literal["vivid", "natural"]] = Field(
        default="vivid",
        description="The style of the generated images. Only supported for dall-e-3.",
    )


class ImageUrl(TypedDict):
    url: Required[HttpUrl]
    refined_prompt: Optional[str]


class ImageB64(TypedDict):
    b64_json: Required[str]
    refined_prompt: Optional[str]


ImageGeneration: TypeAlias = Union[ImageUrl, ImageB64]


class ImageObject(BaseModel):
    created: float = Field(default_factory=lambda: datetime.now().timestamp())
    data: list[ImageGeneration]
    revised_prompt: Optional[str] = Field(default=None)

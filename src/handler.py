import base64
import io
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch
from boto3 import client  # type: ignore
from cachetools import TTLCache, cached
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
    StableDiffusion3Img2ImgPipeline,
)
from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
from schema import BUCKET_NAME, ImageB64, ImageUrl, Request, Response
from typing_extensions import ParamSpec

from .utils import parse_image, parse_mask, refine_prompt

T = TypeVar("T")
P = ParamSpec("P")


def ttl_cache(
    maxsize: int = 128, ttl: int = 300
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return cached(TTLCache[str, T](maxsize, ttl))


@ttl_cache()
def load_sd3() -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")  # type: ignore
    pipe.enable_model_cpu_offload()  # type: ignore
    pipe.enable_sequential_cpu_offload()  # type: ignore
    pipe.enable_xformers_memory_efficient_attention()  # type: ignore
    return pipe  # type: ignore


@ttl_cache()
def load_sd3_img2img() -> StableDiffusion3Img2ImgPipeline:
    return StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")  # type: ignore


@dataclass
class SD3Pipeline:
    model_id: str = field(default="stabilityai/stable-diffusion-3-medium", init=False)
    pipeline_class: type[StableDiffusion3Pipeline] = field(
        default=StableDiffusion3Pipeline, init=False, repr=False
    )
    pipeline_class: type[StableDiffusion3Pipeline] = field(
        default=StableDiffusion3Pipeline, init=False, repr=False
    )
    pipeline: torch.nn.Module = field(init=False)

    def __post_init__(self):
        self.pipeline = self.pipeline_class.from_pretrained(  # type: ignore
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )  # type: ignore
        self.pipeline.enable_model_cpu_offload()  # type: ignore
        self.pipeline.enable_sequential_cpu_offload()  # type: ignore
        self.pipeline.enable_attention_slicing()

    def to_url(self, image: bytes, id: str) -> str:
        s3 = client(service_name="s3", region_name="us-east-1")
        s3.put_object(  # type: ignore
            Body=image,
            Bucket=BUCKET_NAME,
            Key=f"images/{id}.jpg",
            ContentType="image/jpeg",
            ContentDisposition="inline",
        )
        return s3.generate_presigned_url(  # type: ignore
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": f"images/{id}.jpg"},
            ExpiresIn=3600,
        )

    @torch.inference_mode()
    async def generate(self, request: Request) -> bytes:
        try:
            refined_prompt = await refine_prompt(request.prompt, request.style)
            output = self.pipeline(
                prompt=refined_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images_per_prompt=request.n,
                negative_prompt=request.negative_prompt,
            ).images
            assert isinstance(output, list)
            img_byte_arr = io.BytesIO()
            output[0].save(img_byte_arr, format="PNG", quality=100)  # type: ignore
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error in image generation: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Image generation failed: {str(e)}"
            )


@dataclass
class SD3Img2Img(SD3Pipeline):
    pipeline_class: type = field(default=StableDiffusion3Img2ImgPipeline, init=False)

    @torch.inference_mode()
    async def variations_handler(self, request: Request) -> bytes:
        try:
            assert request.image is not None, "Image not found"
            async with AsyncClient() as session:
                refined_prompt = await refine_prompt(request.prompt, request.style)
                image = await parse_image(request.image, request.size, session)  # type: ignore
                output = self.pipeline(
                    prompt=refined_prompt,
                    image=image,
                    strength=request.strength,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    num_images_per_prompt=request.n,
                    negative_prompt=request.negative_prompt,
                ).images

                assert (
                    isinstance(output, list) and len(output) > 0  # type: ignore
                ), "No images were generated"

                img_byte_arr = io.BytesIO()
                output[0].save(img_byte_arr, format="PNG", quality=95)  # type: ignore
                return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error in image variation: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Image variation failed: {str(e)}"
            )

    @torch.inference_mode()
    async def edits_handler(self, request: Request) -> bytes:
        try:
            refined_prompt = await refine_prompt(request.prompt)
            width, height = map(int, request.size.split("x"))

            async with AsyncClient() as session:
                image = await parse_image(request.image, (width, height), session)  # type: ignore
                mask = await parse_mask(request.mask, (width, height), session)  # type: ignore

            generated_image = self.pipeline(
                prompt=refined_prompt,
                image=image,
                mask=mask,
                strength=request.strength,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images_per_prompt=request.n,
                negative_prompt=request.negative_prompt,
            ).images

            assert (
                isinstance(generated_image, list) and len(generated_image) > 0  # type: ignore
            ), "No images were generated"

            img_byte_arr = io.BytesIO()
            generated_image[0].save(  # type: ignore
                img_byte_arr, format=request.output_format, quality=95  # type: ignore
            )
            return img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error in image edit: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")


@dataclass
class SD3Img(SD3Pipeline):
    pipeline_class: type = field(default=StableDiffusion3Pipeline, init=False)


sd3_img2img = SD3Img2Img()
sd3_img = SD3Img()


async def process_request(request: Request) -> Response:
    if request.image and request.mask:
        image_data = await sd3_img2img.edits_handler(request)
    elif request.image:
        image_data = await sd3_img2img.variations_handler(request)
    else:
        image_data = await sd3_img.generate(request)

    if request.response_format == "url":
        return Response(
            data=[
                ImageUrl(
                    url=sd3_img2img.to_url(image_data, request.id),  # type: ignore
                    refined_prompt=request.prompt,
                )
            ],
            revised_prompt=request.prompt,
        )
    return Response(
        data=[
            ImageB64(
                b64_json=base64.b64encode(image_data).decode(),
                refined_prompt=request.prompt,
            )
        ],
        revised_prompt=request.prompt,
    )


def create_app():
    app = FastAPI(
        title="Stable Diffusion 3 API",
        description="Stable Diffusion 3 API",
        version="0.1",
    )

    @app.post("/v1/images/generations")
    async def _(request: Request):
        return await process_request(request)

    @app.post("/v1/images/variations")
    async def _(request: Request):
        if request.image is None:
            raise HTTPException(status_code=400, detail="Image not found")
        return await process_request(request)

    @app.post("/v1/images/edits")
    async def _(request: Request):
        if request.image is None and request.mask is None:
            raise HTTPException(status_code=400, detail="Image and mask not found")
        return await process_request(request)

    return app

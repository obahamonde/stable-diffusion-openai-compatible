import base64
import io
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from boto3 import client  # type: ignore
from botocore.client import Config
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import \
	StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import \
	StableDiffusion3Img2ImgPipeline
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from httpx import AsyncClient
from PIL import Image

from .schema import (BUCKET_NAME, ImageB64, ImageGenerationParams, ImageObject,
					 ImageObject, ImageUrl)
from .utils import asyncify, parse_mask, refine_prompt, ttl_cache, parse_image, image_to_tensor

load_dotenv()
DEFAULT_NEGATIVE_PROMPT = "ugly, malformed, distorted"
s3 = client(service_name="s3", region_name="us-east-1", endpoint_url="https://storage.indiecloud.co", config=Config(signature_version="s3v4"))  # type: ignore


@ttl_cache
def load_sd3() -> StableDiffusion3Pipeline:
	pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")  # type: ignore
	pipe.enable_model_cpu_offload()  # type: ignore
	pipe.enable_sequential_cpu_offload()  # type: ignore
	pipe.enable_xformers_memory_efficient_attention()  # type: ignore
	return pipe  # type: ignore


@ttl_cache
def load_sd3_img2img() -> StableDiffusion3Img2ImgPipeline:
	pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")  # type: ignore
	pipe.enable_model_cpu_offload()  # type: ignore
	pipe.enable_sequential_cpu_offload()  # type: ignore
	pipe.enable_xformers_memory_efficient_attention()  # type: ignore
	return pipe  # type: ignore


@dataclass
class Service:
	sd3: StableDiffusion3Pipeline = field(default_factory=load_sd3)
	sd3_img2img: StableDiffusion3Img2ImgPipeline = field(
		default_factory=load_sd3_img2img
	)

	@asyncify
	def to_url(self, image: bytes) -> str:
		key = f"images/{uuid.uuid4()}.jpg"
		s3.put_object(  # type: ignore
			Body=image,
			Bucket=BUCKET_NAME,
			Key=key,
			ContentType="image/jpeg",
			ContentDisposition="inline",
		)
		return s3.generate_presigned_url(  # type: ignore
			"get_object",
			Params={"Bucket": BUCKET_NAME, "Key": key},
			ExpiresIn=3600,
		)

	@asyncify
	def to_base64(self, image: bytes) -> str:
		return base64.b64encode(image).decode()

	@asyncify
	def to_image(self, image: bytes) -> Image.Image:
		return Image.open(io.BytesIO(image))

	@torch.inference_mode()
	async def generate(self, request: ImageGenerationParams) -> tuple[str, str]:
		try:
			refined_prompt = await refine_prompt(request.prompt)
			output = self.sd3(  # type: ignore
				prompt=refined_prompt,
				num_inference_steps=20,
				guidance_scale=0.8,
				num_images_per_prompt=request.n,
				negative_prompt=DEFAULT_NEGATIVE_PROMPT,
			).images  # type: ignore
			assert isinstance(output, list)
			img_byte_arr = io.BytesIO()
			output[0].save(img_byte_arr, format="PNG", quality=100)  # type: ignore
			image_value = img_byte_arr.getvalue()
			if request.response_format == "url":
				return await self.to_url(image_value), refined_prompt
			else:
				return await self.to_base64(image_value), refined_prompt
		except Exception as e:
			print(f"Error in image generation: {str(e)}")
			raise HTTPException(
				status_code=500, detail=f"Image generation failed: {str(e)}"
			)

	@torch.inference_mode()
	async def variations(
		self,
		*,
		image: bytes,
		n: int,
		prompt: str,
		size:Literal["256x256", "512x512", "1024x1024"],
		response_format: Literal["url", "b64_json"],
	)->tuple[str, str]:
		async with AsyncClient() as session:
			parsed_image = await parse_image(image, size, session)
			refined_prompt = await refine_prompt(prompt)
			output = self.sd3_img2img(  # type: ignore
				prompt=refined_prompt,
				image=parsed_image,
				strength=0.8,
				num_inference_steps=20,
				guidance_scale=0.8,
				num_images_per_prompt=n,
				negative_prompt="ugly, malformed, distorted",
			).images  # type: ignore

			assert (
				isinstance(output, list) and len(output) > 0  # type: ignore
			), "No images were generated"

			img_byte_arr = io.BytesIO()
			output[0].save(img_byte_arr, format="PNG", quality=100)  # type: ignore
			image_value = img_byte_arr.getvalue()
			if response_format == "url":
				return await self.to_url(image_value), refined_prompt
			else:
				return await self.to_base64(image_value), refined_prompt

	@torch.inference_mode()
	async def edits(
		self,
		*,
		prompt: str,
		image: bytes,
		size: Literal["256x256", "512x512", "1024x1024"],
		response_format: Literal["url", "b64_json"],
		n: int,
		mask: Optional[bytes] = None,
	) ->tuple[str, str]:
		refined_prompt = await refine_prompt(prompt)
		try:
			if mask:
				async with AsyncClient() as session:
					
					image_obj = await self.to_image(image)
					processed_mask = await parse_mask(mask, size, session)
					generated_image = self.sd3_img2img(
						prompt=refined_prompt,
						image=image_obj,
						latents = (await image_to_tensor(processed_mask)).to(torch.float32), 
						strength=0.8,
						num_inference_steps=20,
						guidance_scale=0.8,
						num_images_per_prompt=n,
						negative_prompt="ugly, malformed, distorted",
					).images # type: ignore
					assert (
						isinstance(generated_image, list) and len(generated_image) > 0  # type: ignore
					), "No images were generated"


				assert (
					isinstance(generated_image, list) and len(generated_image) > 0  # type: ignore
				), "No images were generated"
			binary = generated_image[0].tobytes()  # type: ignore
			assert isinstance(binary, bytes)
			if response_format == "url":
				return await self.to_url(binary), refined_prompt
			else:
				return await self.to_base64(binary), refined_prompt
		except Exception as e:
			print(f"Error in image edit: {str(e)}")
			raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")


def create_app():
	app = FastAPI(
		title="Stable Diffusion 3 API",
		description="Stable Diffusion 3 API",
		version="0.1",
	)

	service = Service()

	@app.post("/generations")
	async def _(request: ImageGenerationParams):
		return await service.generate(request)	

	@app.post("/variations")
	async def _(
		image: UploadFile = File(...),
		prompt: str = Form(...),
		n: int = Form(default=1),
		size: Literal["256x256", "512x512", "1024x1024"] = Form(default="1024x1024"),
		response_format: Literal["url", "b64_json"] = Form(default="url"),
	):
		return await service.variations(
			image=await image.read(),
			n=n,
			prompt=prompt,
			size=size,
			response_format=response_format,
		)

	@app.post("/edits")
	async def _(
		prompt: str = Form(...),
		image: UploadFile = File(...),
		size: Literal["256x256", "512x512", "1024x1024"] = Form(default="1024x1024"),
		response_format: Literal["url", "b64_json"] = Form(default="url"),
		n: int = Form(default=1),
		mask: Optional[UploadFile] = File(default=None),
	):
		return await service.edits(
			prompt=prompt,
			image=await image.read(),
			size=size,
			response_format=response_format,
			n=n,
			mask=await mask.read() if mask else None,
		)






	return app

"""Custom Ollama service for marker with robust timeout/retry handling."""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import Annotated, List

import requests
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService


class SimplifyOllamaService(BaseService):
    """Marker LLM service backed by Ollama with explicit timeout and retries."""

    ollama_base_url: Annotated[
        str,
        "The base url to use for ollama. No trailing slash.",
    ] = "http://localhost:11434"
    ollama_model: Annotated[
        str,
        "The model name to use for ollama.",
    ] = "qwen2.5vl:7b"

    # Defaults tuned for local CPU/GPU systems that may be slower.
    timeout: Annotated[
        int,
        "HTTP timeout in seconds for each Ollama request.",
    ] = 900
    max_retries: Annotated[
        int,
        "Number of retries for transient Ollama errors.",
    ] = 3
    min_image_side: Annotated[
        int,
        "Minimum side length for each image sent to Ollama.",
    ] = 56
    max_image_side: Annotated[
        int,
        "Maximum side length for each image sent to Ollama.",
    ] = 1536
    max_error_body_chars: Annotated[
        int,
        "Maximum number of response-body characters to include in logs.",
    ] = 600
    fail_fast_on_known_errors: Annotated[
        bool,
        "Whether to stop retrying when a known fatal Ollama panic is detected.",
    ] = True
    text_only_fallback_on_invalid_images: Annotated[
        bool,
        "Whether to retry as text-only when all images are invalid.",
    ] = True

    def _truncate(self, value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "..."

    def _is_known_fatal_error(self, value: str) -> bool:
        lowered = value.lower()
        return "must be larger than factor" in lowered or "post predict" in lowered

    def _sanitize_image(self, image: Image.Image) -> Image.Image | None:
        if image is None:
            return None

        if image.mode != "RGB":
            if "A" in image.getbands():
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

        width, height = image.size
        if width <= 0 or height <= 0:
            return None

        min_side = max(1, int(self.min_image_side))
        max_side = max(min_side, int(self.max_image_side))

        # Clamp large images first to avoid oversized payloads.
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            if hasattr(Image, "Resampling"):
                image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
            else:
                image = image.resize((width, height))

        # Pad tiny crops so qwen2.5vl's internal smart resize does not panic.
        width, height = image.size
        if width < min_side or height < min_side:
            padded_width = max(width, min_side)
            padded_height = max(height, min_side)
            canvas = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
            offset = ((padded_width - width) // 2, (padded_height - height) // 2)
            canvas.paste(image, offset)
            image = canvas

        return image

    def _prepare_images(self, images: List[Image.Image]) -> tuple[list[Image.Image], list[tuple[int, int]]]:
        prepared: list[Image.Image] = []
        dimensions: list[tuple[int, int]] = []
        for raw_image in images:
            sanitized = self._sanitize_image(raw_image)
            if sanitized is None:
                continue
            prepared.append(sanitized)
            dimensions.append(sanitized.size)
        return prepared, dimensions

    def image_to_base64(self, image: Image.Image) -> str:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    def __call__(
        self,
        prompt: str,
        image: Image.Image | List[Image.Image],
        block: Block,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        url = f"{self.ollama_base_url}/api/generate"
        headers = {"Content-Type": "application/json"}

        schema = response_schema.model_json_schema()
        format_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

        if not isinstance(image, list):
            image = [image]

        prepared_images, image_dimensions = self._prepare_images(image)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": format_schema,
        }

        if prepared_images:
            payload["images"] = [self.image_to_base64(img) for img in prepared_images]
        elif not self.text_only_fallback_on_invalid_images:
            print("Ollama inference skipped because all images were invalid after sanitization.")
            return {}

        effective_timeout = timeout or self.timeout
        effective_retries = max_retries if max_retries is not None else self.max_retries
        last_error = None
        total_attempts = effective_retries + 1

        for attempt in range(total_attempts):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=effective_timeout,
                )
                if not response.ok:
                    body = self._truncate(response.text, self.max_error_body_chars)
                    details = (
                        f"HTTP {response.status_code} from Ollama /api/generate "
                        f"(attempt {attempt + 1}/{total_attempts}, model={self.ollama_model}, "
                        f"images={len(prepared_images)}, image_sizes={image_dimensions}): {body}"
                    )
                    last_error = RuntimeError(details)

                    if self.fail_fast_on_known_errors and self._is_known_fatal_error(body):
                        print(details)
                        break

                    raise requests.HTTPError(details, response=response)

                response_data = response.json()

                total_tokens = response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
                block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

                data = response_data.get("response", "{}")
                if isinstance(data, dict):
                    return data
                return json.loads(data)
            except Exception as exc:
                last_error = exc
                if attempt < effective_retries:
                    print(
                        f"Ollama attempt {attempt + 1}/{total_attempts} failed "
                        f"(model={self.ollama_model}, images={len(prepared_images)}, "
                        f"image_sizes={image_dimensions}): {exc}"
                    )
                    time.sleep(min(2 ** attempt, 5))

        print(f"Ollama inference failed after {effective_retries + 1} attempts: {last_error}")
        return {}

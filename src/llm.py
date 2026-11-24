"""Cliente de LLM con la API compatible con OpenAI.

Funciona igual con Ollama, Lemonade, LM Studio, vLLM o una API en la nube; solo cambian las variables:
  LLM_BASE_URL  p. ej. http://localhost:11434/v1 (Ollama) · http://localhost:8000/api/v1 (Lemonade)
  LLM_MODEL     nombre del modelo en ese servidor, p. ej. qwen2.5vl:7b
  LLM_API_KEY   solo para APIs en la nube (los servidores locales aceptan cualquier texto)
  LLM_VISION    1 si el modelo acepta imágenes (por defecto), 0 si no
"""
import base64
import io
import json
import os
import re
import time

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5vl:7b"


def parse_json(text):
    """Extrae el primer objeto JSON de un texto (tolera ```json ...``` y texto alrededor). None si no hay."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def image_part(img, max_side=512):
    """PIL.Image → bloque de imagen para el mensaje (JPEG en base64, reducido para gastar menos tokens)."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


class LLM:
    def __init__(self, base_url=None, model=None, api_key=None, vision=None, timeout=300):
        from openai import OpenAI
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.vision = vision if vision is not None else os.environ.get("LLM_VISION", "1") != "0"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key or os.environ.get("LLM_API_KEY", "local"),
                             timeout=timeout)
        self.json_mode = True   # se apaga solo si el servidor no soporta response_format

    def chat(self, messages, temperature=0.2, max_tokens=800, json_mode=False):
        """Devuelve (texto, segundos)."""
        kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        t = time.perf_counter()
        if json_mode and self.json_mode:
            try:
                r = self.client.chat.completions.create(response_format={"type": "json_object"}, **kw)
                return r.choices[0].message.content or "", time.perf_counter() - t
            except Exception as e:  # algunos servidores no aceptan response_format
                if "response_format" not in str(e).lower():
                    raise
                self.json_mode = False
        r = self.client.chat.completions.create(**kw)
        return r.choices[0].message.content or "", time.perf_counter() - t

    def chat_json(self, messages, **kw):
        """Devuelve (dict o None, segundos). Reintenta una vez si la respuesta no es JSON válido."""
        total = 0.0
        for _ in range(2):
            text, s = self.chat(messages, json_mode=True, **kw)
            total += s
            obj = parse_json(text)
            if obj is not None:
                return obj, total
        return None, total

    def ask(self, prompt, image=None, system=None, **kw):
        """Atajo: un mensaje de usuario (con imagen opcional) → JSON."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append(image_part(image))
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat_json(msgs, **kw)

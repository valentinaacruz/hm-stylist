"""Cliente de LLM con la API compatible con OpenAI.

Funciona igual con Ollama, Lemonade, LM Studio, vLLM o una API en la nube; solo cambian las variables:
  LLM_BASE_URL  p. ej. http://localhost:11434/v1 (Ollama) · http://localhost:8000/api/v1 (Lemonade)
  LLM_MODEL     nombre del modelo en ese servidor, p. ej. qwen2.5vl:7b
  LLM_API_KEY   solo para APIs en la nube (los servidores locales aceptan cualquier texto)
  LLM_VISION    1 si el modelo acepta imágenes (por defecto), 0 si no
  LLM_THINKING  0 para pedir que los modelos con razonamiento (Qwen3.x) respondan sin "pensar" (más rápido)
"""
import base64
import io
import json
import os
import re
import time

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5vl:7b"


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text):
    """Quita el razonamiento que algunos modelos (Qwen3.x) escriben antes de la respuesta."""
    text = THINK_RE.sub("", text or "")
    return text.split("</think>")[-1].strip()


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
                             timeout=timeout, max_retries=1)
        self._down_until = 0.0   # "cortacircuitos": tras un fallo, 30 s sin intentar (respuesta inmediata)
        self.json_mode = True   # se apaga solo si el servidor no soporta response_format
        self.last_error = None
        # Sin razonamiento (llama.cpp / Lemonade lo aceptan vía chat_template_kwargs); se apaga si el servidor falla
        self.extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                           if os.environ.get("LLM_THINKING") == "0" else None)

    def _create(self, **kw):
        if self.extra_body:
            try:
                return self.client.chat.completions.create(extra_body=self.extra_body, **kw)
            except Exception as e:
                if "response_format" in str(e).lower():
                    raise
                self.extra_body = None
        return self.client.chat.completions.create(**kw)

    def chat(self, messages, temperature=0.2, max_tokens=800, json_mode=False):
        """Devuelve (texto, segundos)."""
        kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        t = time.perf_counter()
        if json_mode and self.json_mode:
            try:
                r = self._create(response_format={"type": "json_object"}, **kw)
                return strip_thinking(r.choices[0].message.content), time.perf_counter() - t
            except Exception as e:  # algunos servidores no aceptan response_format
                if "response_format" not in str(e).lower():
                    raise
                self.json_mode = False
        r = self._create(**kw)
        return strip_thinking(r.choices[0].message.content), time.perf_counter() - t

    def chat_json(self, messages, **kw):
        """Devuelve (dict o None, segundos). Reintenta una vez si la respuesta no es JSON válido.
        Si el servidor no responde, devuelve None (el estilista sigue en modo de respaldo) y guarda el error."""
        total, t = 0.0, time.perf_counter()
        if time.time() < self._down_until:      # falló hace poco: no esperar otra vez a que falle
            return None, 0.0
        for _ in range(2):
            try:
                text, s = self.chat(messages, json_mode=True, **kw)
            except Exception as e:     # servidor apagado, modelo no cargado, timeout...
                self.last_error = f"{type(e).__name__}: {e}"
                self._down_until = time.time() + 30
                return None, time.perf_counter() - t
            total += s
            obj = parse_json(text)
            if obj is not None:
                self.last_error = None
                return obj, total
        return None, total

    def ping(self):
        """True si el servidor responde y conoce el modelo configurado."""
        try:
            ids = [m.id for m in self.client.models.list().data]
            self.last_error = None if self.model in ids else f"El modelo {self.model} no está en el servidor: {ids}"
            if self.model in ids:
                self._down_until = 0.0
            return self.model in ids
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._down_until = time.time() + 30
            return False

    def ask(self, prompt, image=None, system=None, **kw):
        """Atajo: un mensaje de usuario (con imagen opcional) → JSON."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append(image_part(image))
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.chat_json(msgs, **kw)

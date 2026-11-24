"""Codificador Fashion-CLIP para las consultas del chat (mismo modelo y preprocesado que el notebook 03)."""
import numpy as np

MODEL = "patrickjohncyh/fashion-clip"

# Prompts para decidir, sin entrenar nada, si una imagen es de moda (zero-shot)
FASHION_PROMPTS = ["a photo of a garment", "a photo of clothing", "a photo of shoes",
                   "a photo of a bag, hat, belt or jewelry", "a photo of a person wearing clothes",
                   "a product photo of an item of clothing"]
OTHER_PROMPTS = ["a photo of food", "a photo of an animal", "a photo of a landscape", "a photo of a vehicle",
                 "a photo of furniture", "a screenshot or a document with text", "random colorful noise",
                 "a photo of an electronic device", "a photo of a building", "an astronomical image"]


def normalize(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class FashionCLIP:
    def __init__(self, name=MODEL, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self._prompt_emb = None

    @staticmethod
    def _as_tensor(out):
        # transformers<5 devuelve un tensor; transformers>=5 un objeto cuyo pooler_output es el embedding
        import torch
        return out if torch.is_tensor(out) else out.pooler_output

    def image(self, img):
        """PIL.Image o ruta → vector (512,) normalizado. Rellena a cuadrado con blanco (no recorta la prenda)."""
        from PIL import Image, ImageOps
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        x = self.proc.image_processor(ImageOps.pad(img.convert("RGB"), (224, 224), color="white"),
                                      return_tensors="pt")["pixel_values"].to(self.device)
        with self.torch.no_grad():
            f = self._as_tensor(self.model.get_image_features(pixel_values=x))
        return self.torch.nn.functional.normalize(f.float(), dim=-1)[0].cpu().numpy()

    def text(self, texts):
        """Lista de textos (en inglés) → matriz (n, 512) normalizada."""
        if isinstance(texts, str):
            texts = [texts]
        t = self.proc.tokenizer(list(texts), padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            f = self._as_tensor(self.model.get_text_features(**t))
        return self.torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    def fashion_prob(self, img_vec):
        """Probabilidad (zero-shot) de que la imagen sea de moda, comparándola contra los prompts de arriba."""
        if self._prompt_emb is None:
            self._prompt_emb = self.text(FASHION_PROMPTS + OTHER_PROMPTS)
        logits = 100.0 * (self._prompt_emb @ img_vec)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[:len(FASHION_PROMPTS)].sum())

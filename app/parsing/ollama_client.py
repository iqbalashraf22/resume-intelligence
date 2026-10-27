import json
import urllib.error
import urllib.request
from types import SimpleNamespace


class OllamaClient:
    """Ollama se baat karta hai: local (localhost) ya cloud (https://ollama.com + API key).
    Interface wahi hai jo Hugging Face client ka tha."""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 api_key: str | None = None, timeout: int = 600,
                 num_ctx: int = 8192, num_gpu: int | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.num_ctx = num_ctx      # local model ka context size (kam karne se GPU memory bachti hai)
        self.num_gpu = num_gpu      # 0 = sirf CPU par chalao, GPU use mat karo

    def chat_completion(self, messages, max_tokens: int = 2000, temperature: float = 0, **kwargs):
        options = {"temperature": temperature, "num_predict": max_tokens}
        if not self.api_key:
            options["num_ctx"] = self.num_ctx          # cloud khud sambhalta hai
            if self.num_gpu is not None:
                options["num_gpu"] = self.num_gpu

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",                      # sirf valid JSON likhne par majboor karo
            "options": options,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:        # HTTPError, URLError ka bachcha hai, isliye pehle
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama ne error diya ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama ({self.host}) se connect nahi ho paya. Local ho toh Ollama app chal raha hai? "
                "Cloud ho toh internet aur OLLAMA_HOST check karo."
            ) from e

        content = data["message"]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

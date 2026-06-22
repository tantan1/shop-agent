from abc import ABC, abstractmethod
from typing import Optional
from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("tts_service")


class TTSProvider(ABC):

    @abstractmethod
    async def synthesize(self, text: str, voice_type: str = "female") -> str:
        pass


class EdgeTTSProvider(TTSProvider):

    def __init__(self):
        self._voices = {
            "female": "zh-CN-XiaoxiaoNeural",
            "male": "zh-CN-YunyangNeural",
            "child": "zh-CN-YunxiNeural"
        }

    async def synthesize(self, text: str, voice_type: str = "female") -> str:
        import edge_tts
        voice = self._voices.get(voice_type, self._voices["female"])
        communicate = edge_tts.Communicate(text, voice)
        audio_path = f"/tmp/tts_{hash(text)}.mp3"
        await communicate.save(audio_path)
        return audio_path


class BaiduTTSProvider(TTSProvider):

    def __init__(self):
        self.api_key = chat_config.baidu_tts_api_key
        self.secret_key = chat_config.baidu_tts_secret_key
        self._token = None

    async def _get_token(self) -> str:
        if self._token:
            return self._token
        import requests
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        resp = requests.post(url, params=params)
        self._token = resp.json().get("access_token")
        return self._token

    async def synthesize(self, text: str, voice_type: str = "female") -> str:
        import requests
        token = await self._get_token()
        url = f"https://tsn.baidu.com/text2audio?access_token={token}"
        params = {
            "tex": text,
            "lan": "zh",
            "per": "0" if voice_type == "female" else "1",
            "cuid": "shop-agent",
            "ctp": "1",
            "aue": "6"
        }
        resp = requests.get(url)
        audio_path = f"/tmp/tts_{hash(text)}.mp3"
        with open(audio_path, "wb") as f:
            f.write(resp.content)
        return audio_path


class TTSService:

    def __init__(self):
        self._provider: TTSProvider = self._create_provider()

    def _create_provider(self) -> TTSProvider:
        provider_type = chat_config.tts_provider
        if provider_type == "baidu" and chat_config.baidu_tts_api_key:
            return BaiduTTSProvider()
        else:
            return EdgeTTSProvider()

    async def synthesize(self, text: str, voice_type: str = "female") -> str:
        return await self._provider.synthesize(text, voice_type)
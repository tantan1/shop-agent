from abc import ABC, abstractmethod
from typing import Optional
from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("avatar_service")


class AvatarProvider(ABC):

    @abstractmethod
    async def render(self, text: str, audio_path: str, avatar_id: str = "default") -> str:
        pass


class BaiduAvatarProvider(AvatarProvider):

    def __init__(self):
        self.api_key = chat_config.baidu_avatar_api_key
        self.secret_key = chat_config.baidu_avatar_secret_key
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

    async def render(self, text: str, audio_path: str, avatar_id: str = "default") -> str:
        import requests
        token = await self._get_token()
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/digital_human/video_generation?access_token={token}"
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        files = {"audio": ("audio.mp3", audio_data, "audio/mpeg")}
        data = {
            "text": text,
            "avatar_id": avatar_id,
            "output_format": "mp4"
        }
        resp = requests.post(url, files=files, data=data)
        result = resp.json()
        return result.get("video_url", "")


class StaticAvatarProvider(AvatarProvider):

    async def render(self, text: str, audio_path: str, avatar_id: str = "default") -> str:
        import os
        static_dir = "src/modules/chat/digital_human/static"
        avatar_images = {
            "default": os.path.join(static_dir, "default_avatar.png"),
            "female": os.path.join(static_dir, "female_avatar.png"),
            "male": os.path.join(static_dir, "male_avatar.png")
        }
        return avatar_images.get(avatar_id, avatar_images["default"])


class AvatarService:

    def __init__(self):
        self._provider: AvatarProvider = self._create_provider()

    def _create_provider(self) -> AvatarProvider:
        provider_type = chat_config.avatar_provider
        if provider_type == "baidu" and chat_config.baidu_avatar_api_key:
            return BaiduAvatarProvider()
        else:
            return StaticAvatarProvider()

    async def render(self, text: str, audio_path: str, avatar_id: str = "default") -> str:
        return await self._provider.render(text, audio_path, avatar_id)
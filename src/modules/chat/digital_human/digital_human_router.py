from fastapi import APIRouter, Depends
from typing import Optional, Literal
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database import get_db
from src.modules.auth.dependencies import verify_api_key
from src.modules.chat.services import ChatAgentService
from src.modules.chat.schemas import ChatRequest, ChatResponse
from src.modules.chat.digital_human.tts_service import TTSService
from src.modules.chat.digital_human.avatar_service import AvatarService
from src.shared.responses import success_response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/digital-human", tags=["数字人客服"])


class DigitalHumanChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="用户消息")
    conversation_id: Optional[str] = Field(default=None, description="对话ID")
    avatar_id: str = Field(default="default", description="数字人形象ID")
    voice_type: Literal["female", "male", "child"] = Field(default="female", description="语音类型")


class DigitalHumanChatResponse(BaseModel):
    video_url: str = Field(..., description="数字人视频地址")
    audio_url: str = Field(..., description="语音文件地址")
    text_response: str = Field(..., description="文本回复")
    conversation_id: str = Field(..., description="对话ID")


async def get_chatagent_service(db: AsyncSession = Depends(get_db)) -> ChatAgentService:
    return ChatAgentService(db)


@router.post("/chat", summary="数字人对话")
async def digital_human_chat(
    request: DigitalHumanChatRequest,
    _: None = Depends(verify_api_key),
    chat_service: ChatAgentService = Depends(get_chatagent_service),
):
    chat_request = ChatRequest(
        message=request.message,
        conversation_id=request.conversation_id,
        stream=False,
    )
    chat_response: ChatResponse = await chat_service.chat(chat_request)

    tts_service = TTSService()
    audio_url = await tts_service.synthesize(
        chat_response.message,
        voice_type=request.voice_type,
    )

    avatar_service = AvatarService()
    video_url = await avatar_service.render(
        chat_response.message,
        audio_url,
        avatar_id=request.avatar_id,
    )

    response = DigitalHumanChatResponse(
        video_url=video_url,
        audio_url=audio_url,
        text_response=chat_response.message,
        conversation_id=chat_response.conversation_id,
    )

    return success_response(data=response.model_dump())
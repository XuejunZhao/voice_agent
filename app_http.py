# # 伪代码示例，用 FastAPI 展示（你也可以用 Flask）
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "voice-ai-by-cursor")
if VOICE_DIR not in sys.path:
    sys.path.insert(0, VOICE_DIR)

from voice_ai_realtime import VoiceAIConfig, RealTimeVoiceAI
from deep_research.service import DeepResearchService  # 你已有
from fastapi.middleware.cors import CORSMiddleware

# 全局存储：session_id -> RealTimeVoiceAI 实例（用于保持对话历史）
voice_ai_sessions: dict[str, RealTimeVoiceAI] = {}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from deep_research.service import DeepResearchService
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper

llm = ChatOpenAI(
    model="deepseek-r1",
    api_key=os.environ.get('DASHSCOPE_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
search_engine = SearxSearchWrapper(searx_host="http://127.0.0.1:38000", k=3)

dr = DeepResearchService(llm, search_engine)

# answer = dr.simple_rag("请介绍一下 Apple Vision Pro 的核心卖点和市场反馈")
# print(answer)

class BrandRequest(BaseModel):
    brand: str
    mode: str = "traditional"  # "traditional" / "omni"

# @app.post("/api/brand_research_voice")
# def brand_research_voice(req: BrandRequest):
#     # 1. 初始化 VoiceAIConfig / DeepResearchService（和 CLI 一样）
#     config = VoiceAIConfig(mode=req.mode)
#     # 这里你已经在 RealTimeVoiceAI.__init__ 里初始化了 deep_research_service
#     v = RealTimeVoiceAI(config)

#     # 2. 跑 deep research（不走麦克风，只用文字）
#     summary = v.fetch_deep_research_background(req.brand)

#     # 3. 用你现有的 TTS 合成语音（例如 dashscope tts）
#     audio_bytes = v.synthesize_speech(summary or "暂无可用总结")

#     import base64
#     audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

#     mode = "omni"

#     return {
#     "summary": summary,
#     "audioBase64": audio_b64,
#     "brand": req.brand,
#     # "mode": req.mode,
#     # 以后如果你想从 deep_research 返回更多，比如：
#     # "plan": final_state.get("search_plan", []),
#     # "knowledge": final_state.get("knowledge", []),
#     }
# # 伪代码：新增一个接口，用于多轮语音对话
# # from fastapi import UploadFile, File, Form

# # class VoiceTurnResponse(BaseModel):
# #     session_id: str
# #     user_text: str
# #     assistant_text: str
# #     audioBase64: str

# # @app.post("/api/voice_turn", response_model=VoiceTurnResponse)
# # async def voice_turn(
# #     session_id: str = Form(...),   # 会话 id，第一轮可以前端传 ""，后端新建
# #     brand: str = Form(...),        # 同一个品牌上下文
# #     mode: str = Form("traditional"),
# #     audio: UploadFile = File(...), # 用户这一轮的语音（wav/pcm 等）
# # ):
# #     """
# #     后端逻辑大致是：
# #     1. 根据 session_id 找到对应的 RealTimeVoiceAI 实例（或聊天历史），没有就新建一个
# #     2. 把上传的 audio 转成 bytes
# #     3. 调用 v.transcribe_audio(...) 得到 user_text
# #     4. 调用 v.get_llm_response(user_text) 得到 assistant_text
# #     5. 调用 v.synthesize_speech(assistant_text) 得到 audio_bytes
# #     6. 返回 session_id / user_text / assistant_text / audioBase64
# #     """
# #     ...

# from typing import Optional
# from fastapi import UploadFile, File, Form
# import base64

# class VoiceTurnResponse(BaseModel):
#     session_id: str
#     user_text: str
#     assistant_text: str
#     audioBase64: str

# # @app.post("/api/voice_turn", response_model=VoiceTurnResponse)
# # async def voice_turn(
# #     session_id: Optional[str] = Form(None),   # 允许第一次不传 / 传空
# #     brand: str = Form(...),                  # 同一个品牌上下文
# #     mode: str = Form("traditional"),
# #     audio: UploadFile = File(...),           # 用户这一轮的语音（webm）
# # ):
# #     """
# #     这里先不接入真实 ASR/LLM，只验证前后端传参正确：
# #     - 读出上传的音频 bytes
# #     - 假装 user_text / assistant_text
# #     - 把原始音频直接回传（前端能播放就说明链路 OK）
# #     """
# #     # 简单生成一个 session_id（真实情况你可以用 uuid 或存到全局 dict 里）
# #     if not session_id:
# #         import uuid
# #         session_id = str(uuid.uuid4())

# #     audio_bytes = await audio.read()

# #     # 这里只是 demo，user_text / assistant_text 先写死，确认链路
# #     user_text = f"（示例）收到一段语音，长度 {len(audio_bytes)} 字节"
# #     assistant_text = f"（示例回复）你好，这是基于品牌 {brand} 的第一个语音回合（模式：{mode}）。"

# #     audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

# #     return VoiceTurnResponse(
# #         session_id=session_id,
# #         user_text=user_text,
# #         assistant_text=assistant_text,
# #         audioBase64=audio_b64,
# #     )

# @app.post("/api/voice_turn", response_model=VoiceTurnResponse)
# async def voice_turn(
#     session_id: Optional[str] = Form(None),
#     brand: str = Form(...),
#     # mode: str = Form("omni"),
#     audio: UploadFile = File(...),
# ):
#     """
#     真实实现：
#     1. 根据 session_id 找到或创建 RealTimeVoiceAI 实例
#     2. 把上传的音频保存到临时文件
#     3. 调用 transcribe_audio 得到 user_text
#     4. 调用 get_llm_response 得到 assistant_text（带 RAG context）
#     5. 调用 synthesize_speech 得到 audio_bytes
#     6. 返回结果
#     """
#     import uuid
#     import tempfile
#     import os

#     mode = "omni"

#     # 生成或使用 session_id
#     if not session_id:
#         session_id = str(uuid.uuid4())

#     # 获取或创建 RealTimeVoiceAI 实例
#     if session_id not in voice_ai_sessions:
#         config = VoiceAIConfig(mode=mode)
#         v = RealTimeVoiceAI(config)
#         # 第一次：设置品牌 RAG context
#         v.prime_rag_with_brand(brand)
#         voice_ai_sessions[session_id] = v
#     else:
#         v = voice_ai_sessions[session_id]

#     # 读取上传的音频 bytes
#     audio_bytes = await audio.read()

#     # 保存到临时文件（webm 格式）
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
#         tmp_file.write(audio_bytes)
#         tmp_audio_path = tmp_file.name

#     try:
#         # 1. ASR：转录音频为文本
#         # 注意：transcribe_audio 期望 wav，如果后端 ASR 不支持 webm，需要先转换
#         # 这里先假设可以直接用，如果报错再改成用 ffmpeg 转 wav
#         user_text = v.transcribe_audio(audio_bytes)  # 如果 transcribe_audio 需要文件路径，改成 tmp_audio_path

#         if not user_text or len(user_text.strip()) < 2:
#             user_text = "（未识别到有效语音）"

#         # 2. LLM：基于 RAG context 生成回复
#         assistant_text = v.get_llm_response(user_text)

#         if not assistant_text:
#             assistant_text = "抱歉，我无法生成回复。"

#         # 3. TTS：合成语音
#         audio_response_bytes = v.synthesize_speech(assistant_text)

#         if not audio_response_bytes:
#             # 如果 TTS 失败，返回一个空音频或错误提示
#             audio_response_bytes = b""

#         audio_b64 = base64.b64encode(audio_response_bytes).decode("utf-8")

#         return VoiceTurnResponse(
#             session_id=session_id,
#             user_text=user_text,
#             assistant_text=assistant_text,
#             audioBase64=audio_b64,
#         )
#     finally:
#         # 清理临时文件
#         if os.path.exists(tmp_audio_path):
#             os.unlink(tmp_audio_path)

from typing import Optional
from fastapi import UploadFile, File, Form
import base64
import uuid
import tempfile
import os


class VoiceTurnResponse(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str
    audioBase64: str

@app.post("/api/voice_turn", response_model=VoiceTurnResponse)
async def voice_turn(
    session_id: Optional[str] = Form(None),
    brand: str = Form(...),              # 现在仅作标识/日志用，不再触发 deep research
    audio: UploadFile = File(...),
):
    """
    只做语音交互：
    1. 根据 session_id 找到或创建 RealTimeVoiceAI 实例（mode 固定为 omni）
    2. 把上传的音频转成 bytes
    3. 调用 transcribe_audio 得到 user_text
    4. 调用 get_llm_response 得到 assistant_text
    5. 调用 synthesize_speech 得到 audio_bytes
    6. 返回 session_id / user_text / assistant_text / audioBase64
    """
    mode = "traditional"

    # 生成或使用 session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    # 获取或创建 RealTimeVoiceAI 实例（不再 prime_rag_with_brand，不再 deep research）
    if session_id not in voice_ai_sessions:
        config = VoiceAIConfig(mode=mode)
        v = RealTimeVoiceAI(config)
        voice_ai_sessions[session_id] = v
    else:
        v = voice_ai_sessions[session_id]

    # 读取上传的音频 bytes
    # 读取上传的音频 bytes
    audio_bytes = await audio.read()

    # 保存 webm 到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        tmp_webm.write(audio_bytes)
        tmp_webm_path = tmp_webm.name

    # 转换为 wav（16kHz, 单声道，PCM 16bit）- ASR 需要的格式
    import subprocess
    tmp_wav_path = tmp_webm_path.replace(".webm", ".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", tmp_webm_path,
                "-ar", "16000",      # 采样率 16kHz
                "-ac", "1",          # 单声道
                "-f", "wav",         # 输出 wav
                "-y",                 # 覆盖已存在文件
                tmp_wav_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # ffmpeg 转换失败
        return VoiceTurnResponse(
            session_id=session_id,
            user_text="（音频格式转换失败）",
            assistant_text="抱歉，无法处理音频格式。",
            audioBase64="",
        )

    # 读取转换后的 wav bytes
    with open(tmp_wav_path, "rb") as f:
        wav_bytes = f.read()

    try:
        # 1. ASR：用 wav bytes 转录音频为文本
        user_text = v.transcribe_audio(wav_bytes)  # 现在传的是 wav，不是 webm

        if not user_text or len(user_text.strip()) < 2:
            user_text = "（未识别到有效语音）"

        # 2. LLM：生成回复
        assistant_text = v.get_llm_response(user_text)

        if not assistant_text:
            assistant_text = "抱歉，我无法生成回复。"

        # 3. TTS：合成语音
        audio_response_bytes = v.synthesize_speech(assistant_text)

        if not audio_response_bytes:
            audio_response_bytes = b""

        audio_b64 = base64.b64encode(audio_response_bytes).decode("utf-8")

        return VoiceTurnResponse(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            audioBase64=audio_b64,
        )
    finally:
        # 清理临时文件
        for path in [tmp_webm_path, tmp_wav_path]:
            if os.path.exists(path):
                os.unlink(path)
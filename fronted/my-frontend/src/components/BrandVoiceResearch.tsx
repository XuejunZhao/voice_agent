import React, { useState, useRef } from "react";

const BrandVoiceResearch: React.FC = () => {
  // 固定品牌和模式（只用于告诉后端 / 显示用）
  const DEFAULT_BRAND = "Apple";
  const DEFAULT_MODE: "traditional" | "omni" = "omni";

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turns, setTurns] = useState<
    { userText: string; assistantText: string }[]
  >([]);
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  const startRecording = async () => {
    try {
      setError("");

      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType =
        MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "audio/webm;codecs=opus";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // 录音结束，合成 Blob 并发送到后端
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

        setLoading(true);
        try {
          const formData = new FormData();
          formData.append("audio", audioBlob, "recording.webm");
          formData.append("brand", DEFAULT_BRAND);
          // 后端已经强制 omni，这里传不传都可以；如果你后端完全忽略 mode，可以去掉这一行
        //   formData.append("mode", DEFAULT_MODE);
          formData.append("session_id", sessionId || "");

          const resp = await fetch("http://localhost:8001/api/voice_turn", {
            method: "POST",
            body: formData,
          });

          if (!resp.ok) {
            const txt = await resp.text();
            throw new Error(txt || `HTTP ${resp.status}`);
          }

          const data: {
            session_id: string;
            user_text: string;
            assistant_text: string;
            audioBase64: string;
          } = await resp.json();

          if (!sessionId && data.session_id) {
            setSessionId(data.session_id);
          }

          setTurns((prev) => [
            ...prev,
            { userText: data.user_text, assistantText: data.assistant_text },
          ]);

          if (data.audioBase64) {
            const byteString = atob(data.audioBase64);
            const len = byteString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {
              bytes[i] = byteString.charCodeAt(i);
            }
            const blob = new Blob([bytes.buffer], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            setAudioUrl(url);
          }
        } catch (e: any) {
          setError(e.message || "语音对话请求失败");
        } finally {
          setLoading(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (e: any) {
      setError(e.message || "无法访问麦克风，请检查浏览器权限");
    }
  };

  const stopRecording = () => {
    const mediaRecorder = mediaRecorderRef.current;
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }
    setIsRecording(false);
  };

  return (
    <div
      style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}
    >
      <h2>语音对话测试</h2>
      <p style={{ marginBottom: 8, color: "#555" }}>
        品牌：{DEFAULT_BRAND}，模式：{DEFAULT_MODE}
      </p>

      {error && (
        <div style={{ marginTop: 16, color: "red" }}>
          错误：{error}
        </div>
      )}

      {/* 多轮语音对话区域 */}
      <div
        style={{
          marginTop: 24,
          paddingTop: 16,
          borderTop: "1px solid #ddd",
        }}
      >
        <h3>多轮语音对话</h3>

        <div style={{ marginBottom: 12 }}>
          <p style={{ marginBottom: 8 }}>
            点击「开始录音」说话，再点击「停止并发送」把这一轮语音发送到后端。
          </p>
          <button
            type="button"
            disabled={loading}
            onClick={isRecording ? stopRecording : startRecording}
          >
            {isRecording ? "停止并发送这一轮语音" : "开始录音"}
          </button>
          {isRecording && (
            <span style={{ marginLeft: 12, color: "red" }}>● 录音中...</span>
          )}
        </div>

        {turns.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h4>对话记录</h4>
            <ol>
              {turns.map((t, idx) => (
                <li key={idx} style={{ marginBottom: 8 }}>
                  <div>
                    <strong>你：</strong>
                    {t.userText}
                  </div>
                  <div>
                    <strong>助手：</strong>
                    {t.assistantText}
                  </div>
                </li>
              ))}
            </ol>
          </div>
        )}

        {audioUrl && (
          <div style={{ marginTop: 16 }}>
            <h4>最新回复语音</h4>
            <audio controls src={audioUrl} />
          </div>
        )}
      </div>
    </div>
  );
};

export default BrandVoiceResearch;
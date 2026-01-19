import os
import sys
import asyncio
import wave
import json
import base64
import threading
import queue
from typing import Optional, Callable
from datetime import datetime
from http import HTTPStatus
import numpy as np

try:
	import pyaudio
except ImportError:
	print(f'❌pyaudio not installed. Installing...')
	os.system('pip install pyaudio')
	import pyaudio

from openai import OpenAI
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

class VoiceAIConfig:
	def __init__(self, mode:str="traditional"):
		self.api_key=os.environ.get('DASHSCOPE_API_KEY')
		if not self.api_key:
			raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

		self.mode = mode

		self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

		if mode == 'traditional':
			self.asr_model = 'fun-asr-realtime-2025-11-07'
			self.llm_model = 'deepseek-v3.2'
			self.tts_model = 'sambert-zhichu-v1'
			self.tts_voice = 'zhichu'
		else:
			# self.omni_model = 'qwen-omni-turbo'
			self.omni_model = 'qwen3-omni-flash-2025-12-01'

		# audio configuration 
		self.sample_rate=16000
		self.channels = 1
		self.chunk_size = 3200
		self.format = pyaudio.paInt16

		# VAD Voice Activity Detection
		self.vad_threshold = 500
		self.silence_duration = 2.0

		# LLM configuration 
		self.enable_thinking = False
		self.temperature = 0.7
		self.max_tokens = 2000

		# RAG configuration
		self.enable_rag = False
		self.system_prompt = 'Please respond shortly without expressive words'
		self.rag_context = ''

class MicrophoneInput:
	def __init__(self, config: VoiceAIConfig):
		self.config = config
		self.audio = pyaudio.PyAudio()
		self.stream = None # please find the stream usage 
		self.is_recording = False
		self.is_listening = False
		self.audio_queue = queue.Queue()

	def start(self):
		# start microphone input stream (but not capturing)
		try:
			self.stream = self.audio.open(
				format=self.config.format,
				channels=self.config.channels,
				rate=self.config.sample_rate,
				input=True,
				frames_per_buffer=self.config.chunk_size,
				stream_callback=self._audio_callback)
			self.stream.start_stream()
			print(f'🎙️ Microphone ready')
		except Exception as e:
			print(f'❌fail to start microphone:{e}')
			raise

	def enable_listening(self):
		# enabling audio capture - only capture audio when this is called
		while not self.audio_queue.empty():
			try:
				self.audio_queue.get_nowait()
			except queue.Empty:
				break
		self.is_listening = True
		self.is_recording = True
		print (f'🎙️ Microphone Enabled - Listening your voice...')

	def disable_listening(self):
		#disabling audio capture - prevent capturing AI output 
		self.is_listening = False
		self.is_recording = False
		while not self.audio_queue.empty():
			try:
				self.audio_queue.get_nowait()
			except queue.Empty:
				break
		
		print (f'🎙️ Microphone Disabled - not capture AI voice')

	def _audio_callback(self, in_data, frame_count, time_info, status):
		# Callback for audio stream - only capture when listening is enabled
		if self.is_recording and self.is_listening:
			self.audio_queue.put(in_data)
		return (in_data, pyaudio.paContinue)

	def read_chunk(self) -> Optional[bytes]:
		# read one chunk for audio data
		# !!! there is one problem that one chunk is too big 
		try:
			return self.audio_queue.get(timeout=0.1)
		except queue.Empty:
			return None

	def stop(self):
		# stop microphone input completely 
		self.is_listening = False
		self.is_recording = False
		if self.stream:
			self.stream.stop_stream()
			self.stream.close()
		self.audio.terminate()
		print (f'🎙️ Microphone terminated')

class SpeakerOutput: 
	def __init__(self, config:VoiceAIConfig):
		self.config = config
		self.audio = pyaudio.PyAudio()
		self.stream = None # please find the stream usage 
		self.is_playing = False

	def start(self):
		'''start speaker output stream'''
		try:
			self.stream = self.audio.open(
				format=self.config.format,
				channels=self.config.channels,
				rate=self.config.sample_rate,
				output=True,
				frames_per_buffer=self.config.chunk_size)
			print(f'📢 speaker is ready')
		except Exception as e: 
			print(f'❌fail to start speaker:{e}')
			raise

	def play(self, audio_data: bytes):
		'''Play audio data through speaker'''
		if self.stream and not self.is_playing:
			self.is_playing = True
			self.stream.write(audio_data)
			self.is_playing = False

	def stop(self):
		'''stop speaker output'''
		if self.stream:
			self.stream.stop_stream()
			self.stream.close()
		self.audio.terminate()
		print (f'📢 speaker terminated')

class RealTimeVoiceAI:
	'''
	voice in - ASR - LLM - tts - voice out
	'''
	def __init__(self, config: VoiceAIConfig):
		self.config = config
		self.microphone = MicrophoneInput(config)
		self.speaker = SpeakerOutput(config)

		dashscope.api_key = config.api_key

		self.llm_client = OpenAI(
			api_key = config.api_key,
			base_url = config.base_url
			)

		self.conversation_history = []

		self.is_running = False
		self.is_listening = False

	def set_rag_context(self, context: str):
		# set knowledge base for injecting 
		self.config.rag_context = context
		self.config.enable_rag = True if context else False
		print(f'rag context is updated: {len(context)} characters')

	def set_system_prompt(self, prompt: str):
		self.config.system_prompt = prompt
		print(f'rag context is updated: {prompt[:50]} ...')

	def clear_chat_history(self):
		self.conversation_history = []
		print(f'chat history is clear')

	def initialize(self):
		# start audio IO
		self.microphone.start()
		self.speaker.start()

	def transcribe_audio(self, audio_data: bytes) -> str:
		# audio -> text
		try: 
			# save audio to temporary file
			# audio saving for how long 
			print(f'Transcribing {len(audio_data)} bytes of audio')
			temp_file = '/tmp/voice_ai_input.wav'
			with wave.open(temp_file, 'wb') as wf:
				wf.setnchannels(self.config.channels)
				wf.setsampwidth(self.microphone.audio.get_sample_size(self.config.format))
				wf.setframerate(self.config.sample_rate)
				wf.writeframes(audio_data)

			recognition = Recognition(
				model=self.config.asr_model,
				format='wav',
				sample_rate=self.config.sample_rate,
				callback=None
				)

			result = recognition.call(temp_file)
			print(f'Recognition result raw: {result}')

			if result and 'output' in result:
				text = result['output']['sentence'][0]['text']
				print(f'transcription: {text}')
				return text
			else: 
				print(f'NO transcription')
				return ''

		except Exception as e:
			print(f'❌fail to transcript:{e}')

	def get_llm_response(self, user_text: str) -> str:
		"""
		Args:
			user_text: user;s transcribed text

		Output:
			LLM response text
		"""

		if len(self.conversation_history) == 0:
			# add system prompt 
			system_context = self.config.system_prompt

			# add rag context
			if self.config.enable_rag and self.config.rag_context:
				system_context += f'\n\ncontext/knowledge base:\n{self.config.rag_context}'
				print(f'rag context is added to system prompt with {len(context)} characters')

			# add user message to history
			self.conversation_history.append({
					'role':'system',
					'content':system_context
				})

		self.conversation_history.append({
			'role':'user',
			'content':user_text
		})
		print(self.conversation_history)

		try:
			response = self.llm_client.chat.completions.create(
				model=self.config.llm_model,
				messages=self.conversation_history,
				temperature=self.config.temperature,
				max_tokens=self.config.max_tokens,
				stream=False #!!!find meaning
				)

			assistant_text = response.choices[0].message.content


			self.conversation_history.append({
				'role':'assistant',
				'content':assistant_text
			})
			return assistant_text
		except Exception as e:
			print(f'❌LLM error:{e}')
			return "sorry, I cannot answer"

	def synthesize_speech(self, text:str) -> bytes:
		"""
		Args:
			user_text: LLM synthesis text

		Output:
			audio bytes
		"""
		try: 
			result = dashscope.audio.tts.SpeechSynthesizer.call(
				model=self.config.tts_model,
				text=text,
				sample_rate=self.config.sample_rate,
				format='pcm')

			if result:
				try:
					audio_data = result.get_audio_data()
					if audio_data:
						print(f'synthesis {len(audio_data)} bytes audio')
						return audio_data
				except Exception as e:
					print(f'❌audio synthesis error:{e}')

			return b''
		except Exception as e:
			print(f'❌TTS error:{e}')
			return b''

	def record_voice_input(self) -> bytes:
		# record until slience 
		self.microphone.enable_listening()
		frames = []
		silent_chunks = 0 
		max_silent_chunks = int(self.config.silence_duration * self.config.sample_rate / self.config.chunk_size) #!!! why equation is this
		print(f'max_silent_chunks_length{max_silent_chunks}')
		started_speaking = False

		try:
			while True:
				chunk = self.microphone.read_chunk()
				if chunk is None:
					continue 

				frames.append(chunk)

				import array
				audio_array = array.array('h', chunk) #!!! meaning
				max_amplitude = max(abs(x) for x in audio_array)

				if max_amplitude > self.config.vad_threshold:
					silent_chunks = 0
					if not started_speaking:
						started_speaking = True
						print(f'🎙️ Speaking detected')
				else:
					if started_speaking:
						silent_chunks += 1

				if started_speaking and silent_chunks >= max_silent_chunks:
					print(f'🎙️ silence detected, stop recording')
					break
		finally:
			self.microphone.disable_listening()
		
		return b''.join(frames)

	def process_voice_with_omni(self, audio_data:bytes) -> bytes:
		try:
			temp_file = '/tmp/voice_ai_omni_input.wav'
			with wave.open(temp_file, 'wb') as wf:
				wf.setnchannels(self.config.channels)
				wf.setsampwidth(self.microphone.audio.get_sample_size(self.config.format))
				wf.setframerate(self.config.sample_rate)
				wf.writeframes(audio_data)

			with open(temp_file, "rb") as audio_file:
				base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
			
			messages = []
			system_context = self.config.system_prompt
			user_context_list = []

			# The audio part
			audio_part = {
				"type": "input_audio",
				"input_audio": {
					"data": f"data:;base64,{base64_audio}",
					"format": "wav",
				},
			}
			user_context_list.append(audio_part)

			# The text part (for RAG)
			if self.config.enable_rag and self.config.rag_context:
				system_context += f'\n\ncontext/knowledge base:\n{self.config.rag_context}'
				text_part = {
					"type": "text",
					"text": "Please answer based on provided context"
				}
				user_context_list.append(text_part)

			if system_context:
				messages.append({
					'role':'system',
					'content':system_context
				})
			
			messages.append({
				'role':'user',
				'content':user_context_list
			})
			
			print(f'Sending request to Omni model: {self.config.omni_model}')
			client = OpenAI(
				api_key=self.config.api_key,  # 确认已配置环境变量
				base_url=self.config.base_url,
				)
			# responses = dashscope.MultiModalConversation.call(
			# 	model=self.config.omni_model,
			# 	messages=messages,
			# 	stream=True
			# 	)
			responses = client.chat.completions.create(
				model=self.config.omni_model,
				messages=messages,
				modalities=["text", "audio"],  # 指定输出文本和音频
				audio={"voice": "Cherry", "format": "wav"},
				stream=True,  # 必须设置为 True
				stream_options={"include_usage": True},
				)
			
			audio_chunks = []
			audio_base64_string = b""
			p = pyaudio.PyAudio()
			stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True)
			for response in responses:
				if response.choices:
					# if response.choices[0].delta.content:
					# 	print(response.choices[0].delta.content, end="")
					if hasattr(response.choices[0].delta, "audio"):
						try:
							audio_string = response.choices[0].delta.audio["data"]							
							wav_bytes = base64.b64decode(audio_string)
							audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
							# print(wav_bytes)
							# audio_base64_string += audio_np.tobytes()
							stream.write(audio_np.tobytes())
						except Exception as e:
							print(response.choices[0].delta.content)
			
			return audio_base64_string

		except Exception as e:
			print(f'omni model fail with error {e}')
		return b''

	def run_conversation_turn_traditional(self):
		try:
			audio_data = self.record_voice_input()
			if len(audio_data) < 3200:
				return

			user_text = self.transcribe_audio(audio_data)
			print(f'User Text: "{user_text}"')

			if not user_text or len(user_text.strip()) < 2: 
				print('User text too short, skipping')
				return 

			assistant_text = self.get_llm_response(user_text)
			print(f'Assistant Text: "{assistant_text}"')

			if not assistant_text:
				print('Assistant text empty, skipping')
				return 

			response_audio = self.synthesize_speech(assistant_text)
			print(f'Response Audio: {len(response_audio) if response_audio else 0} bytes')

			if not response_audio:
				print('Response audio empty, skipping')
				return 
			
			print('Playing audio...')
			self.speaker.play(response_audio) 

		except KeyboardInterrupt:
			raise
		except Exception as e:
			print(f'traditional loop fail with error {e}')

	def run_conversation_turn_omni(self):
		try:
			audio_data = self.record_voice_input()
			if len(audio_data) < 3200:
				return

			response_audio = self.process_voice_with_omni(audio_data)
			print(f'Response Audio: {len(response_audio) if response_audio else 0} bytes')

			if not response_audio:
				return 
			
			self.speaker.play(response_audio) 

		except KeyboardInterrupt:
			raise
		except Exception as e:
			print(f'omni loop fail with error {e}')

	def run_conversation_turn(self):
		if self.config.mode == 'traditional':
			self.run_conversation_turn_traditional()
		else:
			self.run_conversation_turn_omni()

	def run(self):
		self.initialize()
		self.is_running = True
		try:
			while self.is_running: 
				user_input = input('press enter to speak or quit for exit')

				if user_input.lower() in ['quit', 'exit', 'q']:
					break
				self.run_conversation_turn()
		except KeyboardInterrupt: 
			print('Interrupt by user')
		finally:
			self.shutdown()

	def shutdown(self):
		self.is_running=False
		self.microphone.stop()
		self.speaker.stop()

def main():
	if not os.environ.get("DASHSCOPE_API_KEY"):
		sys.exit(1)
	while True: 
		mode_choice = input('choose mode 1Traditional or 2omni with number only').strip()
		if mode_choice == '1':
			mode = 'traditional'
			break
		elif mode_choice == '2': 
			mode = 'omni'
			break
		else:
			print(f'invalid, please choose 1 or 2')
	# print()
	try:
		config = VoiceAIConfig(mode=mode)
		voice_ai = RealTimeVoiceAI(config)
		voice_ai.run()
	except Exception as e: 
		import traceback
		traceback.print_exc()
		sys.exit(1)

if __name__ == "__main__":
	main()
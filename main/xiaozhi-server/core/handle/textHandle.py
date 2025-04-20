from config.logger import setup_logging
import json
import base64
import os
import aiohttp
from datetime import datetime
from core.handle.abortHandle import handleAbortMessage
from core.handle.helloHandle import handleHelloMessage
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import startToChat, handleAudioMessage
from core.handle.sendAudioHandle import send_stt_message, send_tts_message
from core.handle.iotHandle import handleIotDescriptors, handleIotStatus
from plugins_func.register import Action, ActionResponse
from core.utils.dialogue import Message
from core.utils.vl import create_instance
import asyncio

TAG = __name__
logger = setup_logging()


async def handleTextMessage(conn, message):
    """处理文本消息"""
    logger.bind(tag=TAG).info(f"收到文本消息：{message}")
    try:
        msg_json = json.loads(message)
        if isinstance(msg_json, int):
            await conn.websocket.send(message)
            return
        if msg_json["type"] == "hello":
            await handleHelloMessage(conn)
        elif msg_json["type"] == "abort":
            await handleAbortMessage(conn)
        elif msg_json["type"] == "listen":
            if "mode" in msg_json:
                conn.client_listen_mode = msg_json["mode"]
                logger.bind(tag=TAG).debug(f"客户端拾音模式：{conn.client_listen_mode}")
            if msg_json["state"] == "start":
                conn.client_have_voice = True
                conn.client_voice_stop = False
            elif msg_json["state"] == "stop":
                conn.client_have_voice = True
                conn.client_voice_stop = True
                if len(conn.asr_audio) > 0:
                    await handleAudioMessage(conn, b"")
            elif msg_json["state"] == "detect":
                conn.asr_server_receive = False
                conn.client_have_voice = False
                conn.asr_audio.clear()
                if "text" in msg_json:
                    text = msg_json["text"]
                    _, text = remove_punctuation_and_length(text)

                    # 识别是否是唤醒词
                    is_wakeup_words = text in conn.config.get("wakeup_words")
                    # 是否开启唤醒词回复
                    enable_greeting = conn.config.get("enable_greeting", True)

                    if is_wakeup_words and not enable_greeting:
                        # 如果是唤醒词，且关闭了唤醒词回复，就不用回答
                        await send_stt_message(conn, text)
                        await send_tts_message(conn, "stop", None)
                    else:
                        # 否则需要LLM对文字内容进行答复
                        await startToChat(conn, text)
        elif msg_json["type"] == "iot":
            if "descriptors" in msg_json:
                asyncio.create_task(handleIotDescriptors(conn, msg_json["descriptors"]))
            if "states" in msg_json:
                asyncio.create_task(handleIotStatus(conn, msg_json["states"]))
            # 处理摄像头图片
            if "camera_photo" in msg_json:
                # 创建一个任务来处理照片，但不等待它完成
                asyncio.create_task(process_camera_photo(conn, msg_json))
                
                # 立即向客户端发送确认消息
                response = {
                    "type": "iot", 
                    "camera_photo_response": {
                        "status": "processing",
                        "message": "照片已接收，正在处理..."
                    }
                }
                if "session_id" in msg_json:
                    response["session_id"] = msg_json["session_id"]
                
                await conn.websocket.send(json.dumps(response))
    except json.JSONDecodeError:
        await conn.websocket.send(message)

async def process_camera_photo(conn, msg_json):
    """异步处理照片并返回结果"""
    try:
        result = await handleIotCameraPhoto(conn, msg_json)
        if result and isinstance(result, ActionResponse) and result.action == Action.RESPONSE:
            # 向客户端发送识别文本和准备播放语音的消息
            photo_description = f"照片显示: {result.response}"
            await send_stt_message(conn, "看一下这是什么")
            
            # 在对话历史中记录助手响应
            conn.dialogue.put(Message(role="assistant", content=photo_description))
            
            # 模拟function call处理结果
            text_index = conn.tts_last_text_index + 1 if hasattr(conn, "tts_last_text_index") else 0
            conn.recode_first_last_text(photo_description, text_index)
            
            # 创建TTS任务并放入队列
            future = conn.executor.submit(conn.speak_and_play, photo_description, text_index)
            conn.llm_finish_task = True
            conn.tts_queue.put(future)
    except Exception as e:
        logger.bind(tag=TAG).error(f"处理照片任务出错: {str(e)}")

async def handleIotCameraPhoto(conn, msg_json):
    """处理摄像头照片数据"""
    try:
        # 开始计时
        start_time = asyncio.get_event_loop().time()
        
        # 确保照片保存目录存在
        photos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "photos")
        os.makedirs(photos_dir, exist_ok=True)
        
        # 获取照片数据
        photo_data = msg_json["camera_photo"]
        width = photo_data.get("width", 0)
        height = photo_data.get("height", 0)
        format = photo_data.get("format", "jpg")
        base64_data = photo_data.get("data", "")
        
        # 生成文件名
        filename = f"camera_photo.{format}"
        filepath = os.path.join(photos_dir, filename)
        
        # 解码Base64数据并保存为文件
        binary_data = base64.b64decode(base64_data)
        with open(filepath, "wb") as f:
            f.write(binary_data)
            
        logger.bind(tag=TAG).info(f"已保存照片：{filepath}，尺寸：{width}x{height}")

        # 从配置中获取VL提供者
        selected_vl = conn.config.get("selected_module", {}).get("VL", "DoubaoVL")
        vl_config = conn.config.get("VL", {}).get(selected_vl, {})
        vl_type = vl_config.get("type", "openai")
        
        # 创建VL提供者实例
        vl_provider = create_instance(vl_type, vl_config)
        
        # 处理图像
        image_description = await vl_provider.process_image(base64_data)
        
        # 发送确认消息给客户端
        response = {
            "type": "iot", 
            "camera_photo_response": {
                "status": "success",
                "message": f"照片已保存：{filename}"
            }
        }
        if image_description:
            response["camera_photo_response"]["description"] = image_description
            
        if "session_id" in msg_json:
            response["session_id"] = msg_json["session_id"]
            
        await conn.websocket.send(json.dumps(response))
        
        # 计算处理耗时
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        logger.bind(tag=TAG).info(f"照片处理总耗时: {processing_time:.2f}秒")
        
        # 如果成功获取到图片描述，返回ActionResponse
        if image_description:
            return ActionResponse(Action.RESPONSE, None, image_description)
        return None
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"处理照片错误：{str(e)}")
        # 发送错误消息给客户端
        error_response = {
            "type": "iot", 
            "camera_photo_response": {
                "status": "error",
                "message": f"处理照片失败：{str(e)}"
            }
        }
        if "session_id" in msg_json:
            error_response["session_id"] = msg_json["session_id"]
            
        await conn.websocket.send(json.dumps(error_response))
        return None

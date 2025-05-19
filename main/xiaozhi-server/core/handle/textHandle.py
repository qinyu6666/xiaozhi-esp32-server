from config.logger import setup_logging

import json
from core.handle.abortHandle import handleAbortMessage
from core.handle.helloHandle import handleHelloMessage
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import startToChat, handleAudioMessage
from core.handle.sendAudioHandle import send_stt_message, send_tts_message
from core.handle.iotHandle import handleIotDescriptors, handleIotStatus
from core.handle.reportHandle import enqueue_asr_report
import asyncio
import base64
import os
from datetime import datetime
import importlib
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 导入人体检测函数
try:
    from plugins_func.functions.detect_human import detect_human
    DETECT_HUMAN_AVAILABLE = True
except ImportError:
    DETECT_HUMAN_AVAILABLE = False

TAG = __name__
logger = setup_logging()

# 存储上一次检测到的region_index
last_region_index = None

async def handleTextMessage(conn, message):
    """处理文本消息"""
    # conn.logger.bind(tag=TAG).info(f"收到文本消息：{message}")
    try:
        msg_json = json.loads(message)
        if isinstance(msg_json, int):
            await conn.websocket.send(message)
            return
        if msg_json["type"] == "hello":
            await handleHelloMessage(conn, msg_json)
        elif msg_json["type"] == "abort":
            await handleAbortMessage(conn)
        elif msg_json["type"] == "listen":
            if "mode" in msg_json:
                conn.client_listen_mode = msg_json["mode"]
                conn.logger.bind(tag=TAG).debug(
                    f"客户端拾音模式：{conn.client_listen_mode}"
                )
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
                    elif is_wakeup_words:
                        # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                        enqueue_asr_report(conn, "嘿，你好呀", [])
                        await startToChat(conn, "嘿，你好呀")
                    else:
                        # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                        enqueue_asr_report(conn, text, [])
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
                asyncio.create_task(handleIotCameraPhoto(conn, msg_json))
                
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
        elif msg_json["type"] == "server":
            # 如果配置是从API读取的，则需要验证secret
            if not conn.read_config_from_api:
                return
            # 获取post请求的secret
            post_secret = msg_json.get("content", {}).get("secret", "")
            secret = conn.config["manager-api"].get("secret", "")
            # 如果secret不匹配，则返回
            if post_secret != secret:
                await conn.websocket.send(
                    json.dumps(
                        {
                            "type": "server",
                            "status": "error",
                            "message": "服务器密钥验证失败",
                        }
                    )
                )
                return
            # 动态更新配置
            if msg_json["action"] == "update_config":
                try:
                    # 更新WebSocketServer的配置
                    if not conn.server:
                        await conn.websocket.send(
                            json.dumps(
                                {
                                    "type": "config_update_response",
                                    "status": "error",
                                    "message": "无法获取服务器实例",
                                }
                            )
                        )
                        return

                    if not await conn.server.update_config():
                        await conn.websocket.send(
                            json.dumps(
                                {
                                    "type": "config_update_response",
                                    "status": "error",
                                    "message": "更新服务器配置失败",
                                }
                            )
                        )
                        return

                    # 发送成功响应
                    await conn.websocket.send(
                        json.dumps(
                            {
                                "type": "config_update_response",
                                "status": "success",
                                "message": "配置更新成功",
                            }
                        )
                    )
                except Exception as e:
                    conn.logger.bind(tag=TAG).error(f"更新配置失败: {str(e)}")
                    await conn.websocket.send(
                        json.dumps(
                            {
                                "type": "config_update_response",
                                "status": "error",
                                "message": f"更新配置失败: {str(e)}",
                            }
                        )
                    )
            # 重启服务器
            elif msg_json["action"] == "restart":
                await conn.handle_restart(msg_json)
    except json.JSONDecodeError:
        await conn.websocket.send(message)


async def handleIotCameraPhoto(conn, msg_json):
    """处理摄像头照片数据"""
    global last_region_index
    
    try:
        # 开始计时（使用毫秒精度）
        start_time = asyncio.get_event_loop().time() * 1000  # 转换为毫秒
        
        logger.bind(tag=TAG).info(f"开始处理摄像头照片，消息类型: {msg_json.get('type')}")
        
        # 确保照片保存目录存在
        photos_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "photos")
        os.makedirs(photos_dir, exist_ok=True)
        
        # 获取照片数据
        photo_data = msg_json["camera_photo"]
        width = photo_data.get("width", 0)
        height = photo_data.get("height", 0)
        format = photo_data.get("format", "jpg")
        base64_data = photo_data.get("data", "")
        
        logger.bind(tag=TAG).debug(f"照片数据: 格式={format}, 宽度={width}, 高度={height}, Base64长度={len(base64_data) if base64_data else 0}")
        
        # 检查base64数据
        if not base64_data:
            logger.bind(tag=TAG).error("照片数据为空")
            raise ValueError("照片数据为空")
        
        # 生成带时间戳的文件名 (年月日_时分秒_毫秒)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度（去掉微秒的后3位）
        filename = f"camera_photo.{format}"
        filepath = os.path.join(photos_dir, filename)
        
        logger.bind(tag=TAG).info(f"将保存照片到: {filepath}")
        
        # 解码Base64数据并保存为文件
        try:
            binary_data = base64.b64decode(base64_data)
            logger.bind(tag=TAG).debug(f"Base64解码后数据大小: {len(binary_data)} 字节")
            
            with open(filepath, "wb") as f:
                f.write(binary_data)
                
            logger.bind(tag=TAG).info(f"已保存照片：{filepath}，尺寸：{width}x{height}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存照片失败: {str(e)}")
            raise
        
        # 执行人体检测
        detection_result = None
        detected_humans = False
        region_indexes = []
        current_region_index = None
        
        if DETECT_HUMAN_AVAILABLE:
            logger.bind(tag=TAG).info("开始进行人体检测")
            try:
                # 调用人体检测功能
                detection_result = detect_human({
                    "image_base64": base64_data,
                    "image_filename": filename
                })
                
                logger.bind(tag=TAG).info(f"人体检测结果: {detection_result.get('message', '无结果')}")
                logger.bind(tag=TAG).debug(f"人体检测详细结果: {json.dumps(detection_result, ensure_ascii=False)}")
                
                # 检查是否检测到人体
                if detection_result and detection_result.get("count", 0) > 0:
                    detected_humans = True
                    # 提取所有检测到的人体的region_index
                    detections = detection_result.get("detections", [])
                    region_indexes = [d.get("region_index") for d in detections if "region_index" in d]
                    
                    if region_indexes:
                        current_region_index = region_indexes[0]
                        
                        # 检查当前区域索引是否与上一次相同
                        if current_region_index == last_region_index:
                            logger.bind(tag=TAG).info(f"当前区域索引 {current_region_index} 与上一次相同，不发送消息给客户端")
                            detected_humans = False  # 不发送消息
                        else:
                            logger.bind(tag=TAG).info(f"区域索引变化: {last_region_index} -> {current_region_index}")
                            # 更新上一次的区域索引
                            last_region_index = current_region_index
                    
                    logger.bind(tag=TAG).info(f"检测到人体，区域索引: {region_indexes}")
                else:
                    logger.bind(tag=TAG).info("未检测到人体，不发送消息给客户端")
                    
            except Exception as e:
                logger.bind(tag=TAG).error(f"人体检测失败: {str(e)}", exc_info=True)
                detection_result = {"error": f"人体检测失败: {str(e)}"}
        else:
            logger.bind(tag=TAG).warning("人体检测功能不可用，跳过检测")

        # 只有在检测到人体且区域索引发生变化时才发送消息给客户端
        if detected_humans:
            # 发送确认消息给客户端
            response = {
                "type": "iot", 
                "camera_photo_response": {
                    "status": "success",
                    "message": f"检测到人体，位于区域: {region_indexes}",
                    "filename": filename,
                    "timestamp": timestamp,
                    "filepath": filepath,
                    "width": width,
                    "height": height,
                    "detection_result": detection_result,
                    "region_index": current_region_index  # 将区域索引放到最外层
                }
            }
                
            if "session_id" in msg_json:
                response["session_id"] = msg_json["session_id"]
            
            # 计算处理耗时（毫秒）
            end_time = asyncio.get_event_loop().time() * 1000  # 转换为毫秒
            processing_time = end_time - start_time
            processing_time_ms = int(processing_time)  # 整数毫秒
            logger.bind(tag=TAG).info(f"照片处理总耗时: {processing_time_ms}毫秒")

            # 将处理时间添加到响应中
            response["processing_time_ms"] = processing_time_ms

            logger.bind(tag=TAG).debug(f"发送响应: {json.dumps(response, ensure_ascii=False)}")
            await conn.websocket.send(json.dumps(response))
        else:
            logger.bind(tag=TAG).info("未检测到人体或区域索引未变化，不发送消息给客户端")
        
        return None
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"处理照片错误：{str(e)}", exc_info=True)
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


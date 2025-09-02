import os
import time
import base64
from typing import Optional, Dict

import requests
import json

TAG = __name__


class DeviceNotFoundException(Exception):
    pass


class DeviceBindException(Exception):
    def __init__(self, bind_code):
        self.bind_code = bind_code
        super().__init__(f"设备绑定异常，绑定码: {bind_code}")


class ManageApiClient:
    _instance = None
    _base_url = None
    _headers = None

    def __new__(cls, config):
        """单例模式确保全局唯一实例，并支持传入配置参数"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._init_client(config)
        return cls._instance

    @classmethod
    def _init_client(cls, config):
        """初始化HTTP客户端配置"""
        cls.config = config.get("manager-api")

        if not cls.config:
            raise Exception("manager-api配置错误")

        if not cls.config.get("url") or not cls.config.get("secret"):
            raise Exception("manager-api的url或secret配置错误")

        if "你" in cls.config.get("secret"):
            raise Exception("请先配置manager-api的secret")

        cls._secret = cls.config.get("secret")
        
        cls.max_retries = cls.config.get("max_retries", 6)  # 最大重试次数
        cls.retry_delay = cls.config.get("retry_delay", 10)  # 初始重试延迟(秒)
        cls.timeout = cls.config.get("timeout", 30)  # 默认超时时间30秒
        
        # 设置base URL和headers
        cls._base_url = cls.config.get("url").rstrip("/")
        cls._headers = {
            "User-Agent": f"PythonClient/2.0 (PID:{os.getpid()})",
            "Accept": "application/json",
            "Authorization": f"Bearer {cls._secret}",
            "Content-Type": "application/json"
        }

    @classmethod
    def _request(cls, method: str, endpoint: str, **kwargs) -> Dict:
        """发送单次HTTP请求并处理响应"""
        # 构建完整URL
        endpoint = endpoint.lstrip("/")
        full_url = f"{cls._base_url}/{endpoint}"
        
        # print(f"[DEBUG] 发送HTTP请求: {method} {full_url}")
        # print(f"[DEBUG] Headers: {cls._headers}")
        
        # 使用requests发送请求
        if method.upper() == "POST":
            if 'json' in kwargs:
                response = requests.post(
                    full_url, 
                    headers=cls._headers, 
                    json=kwargs['json'], 
                    timeout=cls.timeout
                )
            else:
                response = requests.post(
                    full_url, 
                    headers=cls._headers, 
                    timeout=cls.timeout
                )
        elif method.upper() == "PUT":
            if 'json' in kwargs:
                response = requests.put(
                    full_url, 
                    headers=cls._headers, 
                    json=kwargs['json'], 
                    timeout=cls.timeout
                )
            else:
                response = requests.put(
                    full_url, 
                    headers=cls._headers, 
                    timeout=cls.timeout
                )
        else:
            response = requests.request(
                method, 
                full_url, 
                headers=cls._headers, 
                timeout=cls.timeout, 
                **kwargs
            )
        
        # print(f"[DEBUG] 响应状态码: {response.status_code}")
        # print(f"[DEBUG] 实际请求URL: {response.url}")
        
        # 检查HTTP状态码
        response.raise_for_status()

        result = response.json()

        # 处理API返回的业务错误
        if result.get("code") == 10041:
            raise DeviceNotFoundException(result.get("msg"))
        elif result.get("code") == 10042:
            raise DeviceBindException(result.get("msg"))
        elif result.get("code") != 0:
            raise Exception(f"API返回错误: {result.get('msg', '未知错误')}")

        # 返回成功数据
        return result.get("data") if result.get("code") == 0 else None

    @classmethod
    def _should_retry(cls, exception: Exception) -> bool:
        """判断异常是否应该重试"""
        # 网络连接相关错误
        if isinstance(
            exception, (requests.ConnectionError, requests.Timeout, requests.RequestException)
        ):
            return True

        # HTTP状态码错误
        if isinstance(exception, requests.HTTPError):
            status_code = exception.response.status_code
            return status_code in [408, 429, 500, 502, 503, 504]

        return False

    @classmethod
    def _execute_request(cls, method: str, endpoint: str, **kwargs) -> Dict:
        """带重试机制的请求执行器"""
        retry_count = 0

        while retry_count <= cls.max_retries:
            try:
                # 执行请求
                return cls._request(method, endpoint, **kwargs)
            except Exception as e:
                # 判断是否应该重试
                if retry_count < cls.max_retries and cls._should_retry(e):
                    retry_count += 1
                    print(
                        f"{method} {endpoint} 请求失败，将在 {cls.retry_delay:.1f} 秒后进行第 {retry_count} 次重试"
                    )
                    time.sleep(cls.retry_delay)
                    continue
                else:
                    # 不重试，直接抛出异常
                    raise

    @classmethod
    def safe_close(cls):
        """清理实例"""
        # requests库没有显式的关闭方法，这里只是清理实例
        cls._instance = None


def get_server_config() -> Optional[Dict]:
    """获取服务器基础配置"""
    return ManageApiClient._instance._execute_request("POST", "/config/server-base")


def get_agent_models(
    mac_address: str, client_id: str, selected_module: Dict
) -> Optional[Dict]:
    """获取代理模型配置"""
    return ManageApiClient._instance._execute_request(
        "POST",
        "/config/agent-models",
        json={
            "macAddress": mac_address,
            "clientId": client_id,
            "selectedModule": selected_module,
        },
    )


def save_mem_local_short(mac_address: str, short_momery: str) -> Optional[Dict]:
    try:
        return ManageApiClient._instance._execute_request(
            "PUT",
            f"/agent/saveMemory/" + mac_address,
            json={
                "summaryMemory": short_momery,
            },
        )
    except Exception as e:
        print(f"存储短期记忆到服务器失败: {e}")
        return None


def report(
    mac_address: str, session_id: str, chat_type: int, content: str, audio, report_time
) -> Optional[Dict]:
    """带熔断的业务方法示例"""
    if not content or not ManageApiClient._instance:
        return None
    try:
        return ManageApiClient._instance._execute_request(
            "POST",
            f"/agent/chat-history/report",
            json={
                "macAddress": mac_address,
                "sessionId": session_id,
                "chatType": chat_type,
                "content": content,
                "reportTime": report_time,
                "audioBase64": (
                    base64.b64encode(audio).decode("utf-8") if audio else None
                ),
            },
        )
    except Exception as e:
        print(f"TTS上报失败: {e}")
        return None


def init_service(config):
    ManageApiClient(config)


def manage_api_http_safe_close():
    ManageApiClient.safe_close()

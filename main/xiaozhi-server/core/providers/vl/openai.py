import aiohttp
import asyncio
from config.logger import setup_logging
from core.providers.vl.base import VLProviderBase

TAG = __name__
logger = setup_logging()

class VLProvider(VLProviderBase):
    """方舟模型的视觉语言处理实现"""
    
    def __init__(self, config):
        """初始化方舟模型视觉语言处理
        
        Args:
            config (dict): 配置信息，包含API密钥等
        """
        self.api_url = config.get("api_url", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model_name", "doubao-1.5-vision-pro-32k-250115")
        self.temperature = config.get("temperature", 0.9)
        self.max_tokens = config.get("max_tokens", 80)
        self.top_p = config.get("top_p", 0.95)
        
        # 配置日志
        logger.bind(tag=TAG).info(f"初始化方舟VL配置: model={self.model_name}, url={self.api_url}")
    
    async def process_image(self, base64_data, prompt=None):
        """处理图像并返回描述
        
        Args:
            base64_data (str): 图像的Base64编码数据
            prompt (str, optional): 提示词，默认为简短描述图片内容
            
        Returns:
            str: 图像描述文本
        """
        if not prompt:
            prompt = "简短的描述一下这个图片里面主要的内容"
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_data}"
                            }
                        }
                    ]
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
        
        # 设置超时时间为5秒
        timeout = aiohttp.ClientTimeout(total=5.0)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        image_description = response_data['choices'][0]['message']['content']
                        logger.bind(tag=TAG).info(f"图片描述：{image_description}")
                        return image_description
                    else:
                        error_text = await response.text()
                        logger.bind(tag=TAG).error(f"API调用失败：{response.status}, {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.bind(tag=TAG).warning("API请求超时，使用默认描述")
            return "这是一张照片。由于处理速度原因，无法提供详细描述。"
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理图像时出错: {str(e)}")
            return None

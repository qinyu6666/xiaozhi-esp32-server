from abc import ABC, abstractmethod

class VLProviderBase(ABC):
    """视觉语言处理提供商基类"""
    
    @abstractmethod
    async def process_image(self, base64_data, prompt=None):
        """处理图像并返回描述
        
        Args:
            base64_data (str): 图像的Base64编码数据
            prompt (str, optional): 提示词，用于指导模型生成描述
            
        Returns:
            str: 图像描述文本
        """
        pass 
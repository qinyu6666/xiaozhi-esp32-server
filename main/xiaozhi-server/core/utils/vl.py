import importlib
import os
import sys
from core.providers.vl.base import VLProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


def create_instance(class_name: str, *args, **kwargs) -> VLProviderBase:
    """工厂方法创建VL实例"""
    if os.path.exists(os.path.join("core", "providers", "vl", f"{class_name}.py")):
        lib_name = f"core.providers.vl.{class_name}"
        if lib_name not in sys.modules:
            sys.modules[lib_name] = importlib.import_module(f"{lib_name}")
        return sys.modules[lib_name].VLProvider(*args, **kwargs)
    
    raise ValueError(f"不支持的VL类型: {class_name}，请检查该配置的type是否设置正确")

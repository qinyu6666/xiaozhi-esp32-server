import os
import logging
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch

# 配置日志
logger = logging.getLogger(__name__)

# 函数信息
function_info = {
    "name": "detect_human",
    "description": "检测图片中的人体位置",
    "parameters": {
        "type": "object",
        "properties": {
            "image_base64": {
                "type": "string",
                "description": "base64编码的图片"
            }
        },
        "required": ["image_base64"]
    }
}

class DetectHuman:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                      'models', 'yolov8n.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """加载YOLOv8模型"""
        if self.model is None:
            try:
                # 如果模型文件不存在，会自动下载
                self.model = YOLO(self.model_path)
                logger.info(f"YOLOv8模型已加载，使用设备: {self.device}")
            except Exception as e:
                logger.error(f"加载YOLOv8模型失败: {str(e)}")
                raise e
        return self.model
    
    def base64_to_image(self, image_base64):
        """将base64编码的图片转换为PIL Image对象"""
        try:
            # 如果base64字符串包含前缀，需要移除
            if ',' in image_base64:
                logger.debug("检测到base64数据包含前缀，将移除")
                image_base64 = image_base64.split(',')[1]
            
            logger.debug(f"开始解码base64图片数据，数据长度: {len(image_base64)}")
            image_data = base64.b64decode(image_base64)
            logger.debug(f"base64解码完成，数据大小: {len(image_data)}字节")
            
            image = Image.open(BytesIO(image_data))
            logger.debug(f"图片加载成功，格式: {image.format}, 尺寸: {image.width}x{image.height}")
            return image
        except Exception as e:
            logger.error(f"Base64转换为图片失败: {str(e)}", exc_info=True)
            raise e
    
    def detect(self, image):
        """使用YOLOv8检测图片中的人体"""
        try:
            logger.info(f"开始检测图片，尺寸: {image.width}x{image.height}")
            model = self.load_model()
            # 只检测人类（class 0在COCO数据集中是人类）
            logger.debug("使用YOLOv8执行人体检测，仅检测class 0（人类）")
            results = model(image, classes=[0])
            
            detections = []
            if results and len(results) > 0:
                logger.info(f"YOLOv8模型返回结果，处理结果")
                for result in results:
                    boxes = result.boxes
                    logger.info(f"检测到{len(boxes)}个边界框")
                    
                    img_height = result.orig_shape[0]
                    img_width = result.orig_shape[1]
                    logger.debug(f"原始图像尺寸: {img_width}x{img_height}")
                    
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        
                        logger.debug(f"边界框 #{i}: 类别={class_id}, 置信度={confidence:.3f}, 坐标=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        
                        # 计算相对位置（上/中/下）
                        center_y = (y1 + y2) / 2
                        relative_pos = "上方" if center_y < img_height/3 else "中间" if center_y < 2*img_height/3 else "下方"
                        
                        # 计算水平中心点
                        center_x = (x1 + x2) / 2
                        
                        # 计算区域索引（假设图像宽度为320像素，分成18个区域）
                        # 对于非320宽度的图像，先进行比例缩放
                        scaled_center_x = (center_x / img_width) * 320
                        region_width = 320 / 18
                        region_index = int(scaled_center_x / region_width)
                        # 确保索引在0-17的范围内
                        region_index = max(0, min(17, region_index))
                        
                        logger.debug(f"人体 #{i}: 位置={relative_pos}, 中心点=({center_x:.1f},{center_y:.1f}), 区域索引={region_index}")
                        
                        detections.append({
                            "id": i,
                            "class": "person",
                            "confidence": round(confidence, 3),
                            "box": {
                                "x1": round(x1, 2),
                                "y1": round(y1, 2),
                                "x2": round(x2, 2),
                                "y2": round(y2, 2)
                            },
                            "position": relative_pos,
                            "region_index": region_index,
                            "center_x": round(center_x, 2),
                            "img_width": img_width
                        })
            else:
                logger.info("YOLOv8模型未返回检测结果")
            
            return detections
        except Exception as e:
            logger.error(f"人体检测失败: {str(e)}", exc_info=True)
            raise e

# 创建检测器实例
detector = DetectHuman()

def detect_human(params, prompt_result=""):
    """检测图片中的人体位置"""
    # 开始计时（毫秒精度）
    start_time = time.time() * 1000  # 转换为毫秒
    
    try:
        logger.info(f"开始人体检测，参数: {params.keys()}")
        
        image_base64 = params.get("image_base64", "")
        if not image_base64:
            logger.error("未提供图片数据")
            return {"error": "未提供图片数据"}
        
        # 图片文件名（如果有提供）
        image_filename = params.get("image_filename", "未命名图片")
        logger.info(f"处理图片: {image_filename}, 图片数据长度: {len(image_base64)}")
        
        # 将base64转换为图片
        try:
            image = detector.base64_to_image(image_base64)
            logger.info(f"图片解码成功，尺寸: {image.width}x{image.height}")
        except Exception as e:
            logger.error(f"图片解码失败: {str(e)}", exc_info=True)
            return {"error": f"图片解码失败: {str(e)}"}
        
        # 执行检测
        logger.info("开始执行YOLOv8检测")
        detections = detector.detect(image)
        logger.info(f"检测完成，发现{len(detections)}个人体")
        
        # 构建结果
        if not detections:
            logger.info(f"未检测到人体 (图片: {image_filename})")
            result = {
                "count": 0, 
                "message": f"未检测到人体 (图片: {image_filename})",
                "image_filename": image_filename
            }
        else:
            # 获取人体位置描述
            positions = [d["position"] for d in detections]
            position_count = {}
            for pos in positions:
                position_count[pos] = position_count.get(pos, 0) + 1
            
            # 获取区域分布
            regions = [d["region_index"] for d in detections]
            region_desc = ", ".join([f"区域{r}" for r in regions])
            
            position_desc = ", ".join([f"{pos}有{count}人" for pos, count in position_count.items()])
            
            logger.info(f"检测到{len(detections)}个人体，{position_desc}，位于{region_desc} (图片: {image_filename})")
            logger.debug(f"检测到的人体详细信息: {detections}")
            
            result = {
                "count": len(detections),
                "message": f"检测到{len(detections)}个人体，{position_desc}，位于{region_desc} (图片: {image_filename})",
                "detections": detections,
                "image_filename": image_filename
            }
        
        # 添加处理时间（毫秒）
        elapsed_time = time.time() * 1000 - start_time
        result["process_time"] = int(elapsed_time)  # 整数毫秒
        logger.info(f"人体检测处理时间: {result['process_time']}毫秒")
        
        return result
    
    except Exception as e:
        logger.error(f"执行人体检测时出错: {str(e)}", exc_info=True)
        return {"error": f"检测失败: {str(e)}"}

if __name__ == "__main__":
    # 示例：从文件加载图片并进行测试
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        with open(img_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        result = detect_human({"image_base64": img_base64})
        print(result) 
import math
import re
from typing import List

class LocalSemanticEmbedding:
    """完全离线的教学型工业语义向量。检索阶段仅比较向量余弦相似度。"""
    def __init__(self):
        self.semantic_dimensions = {
            "temperature_rise": ["油温持续升高","油温升高","油温继续升高","油温逐步升高","温度持续升高","温度升高","逐步升高","越来越热","机器越来越热","持续升温","发热"],
            "stable_load": ["负载变化不大","负荷变化不大","负荷没怎么变","负载没怎么变","负载稳定","负荷稳定","负载没变化","负荷没变化"],
            "cooling": ["散热能力","散热能力下降","散热不好","散热","冷却","冷却系统"],
            "fan": ["冷却风扇","风扇","风扇报警","风扇停转","风扇异常"],
            "oil_pump": ["油泵","油泵运行","油泵运行状态"],
            "cooling_channel": ["散热通道","散热通道受阻","通道受阻","通道堵塞","堵塞"],
            "temperature_jump": ["温度瞬时跳变","瞬时跳变","温度突然跳","突然跳变","温度突变","突然变化"],
            "sensor": ["传感器","传感器接线","采样异常","温度采样","采样","接线异常"],
            "overload": ["持续过载","长期过载","设备长期过载","过载","负荷过大","负载过大"],
            "vibration": ["振动异常","异常振动","设备振动","振动"],
            "mechanical": ["机械连接","机械","连接"],
            "bearing": ["轴承","轴承状态"]
        }
        self.dimension_names = list(self.semantic_dimensions)

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def encode(self, text: str) -> List[float]:
        normalized = self.normalize_text(text)
        vector = []
        for dimension in self.dimension_names:
            activation = 0.0
            for expression in self.semantic_dimensions[dimension]:
                if self.normalize_text(expression) in normalized:
                    activation = 1.0
                    break
            vector.append(activation)
        norm = math.sqrt(sum(v*v for v in vector))
        return [v/norm for v in vector] if norm else vector

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("两个向量维度必须一致")
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

# from abc import ABCMeta, abstractmethod
from enum import Enum, auto

class BaseDataType(Enum):
    """
    数据类型的抽象基类，用于定义不同领域的数据类型应实现的公共接口。
    """
    @classmethod
    def get_all_types(cls):
        return list(cls)

class WritingDataType(BaseDataType):
    """
    文本数据类型。
    """
    Character = auto()
    Location = auto()
    Plot = auto()

# class ImageDataType(BaseDataType):
#     """
#     图像数据类型。
#     """
#     Scene = auto()
#     Object = auto()
#     Event = auto()
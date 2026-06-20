# -*- coding: utf-8 -*-
"""
AutoShot 核心算法包
基于 TransNetV2Supernet 的镜头边界检测模型

改造自 https://github.com/PhucNguyenLamp/Shot_Detection
"""

from .model import AutoShot

__all__ = ["AutoShot"]

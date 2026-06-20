# -*- coding: utf-8 -*-
"""
ComfyUI-AutoShot 插件入口
基于深度学习的视频镜头自动分割与关键帧提取插件

核心算法改造自: https://github.com/PhucNguyenLamp/Shot_Detection
模型: TransNetV2Supernet
"""

import os
from .nodes import AutoShotDetectionNode

# 前端静态资源目录（ComfyUI 会自动加载此目录下的 JS）
WEB_DIRECTORY = "./web"

# 节点类映射 - ComfyUI 核心注册机制
NODE_CLASS_MAPPINGS = {
    "AutoShotDetection": AutoShotDetectionNode
}

# 节点显示名称映射（界面上展示的友好名称）
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoShotDetection": "Auto Shot Detection 镜头分割取关键帧"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]

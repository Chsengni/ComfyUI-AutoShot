# -*- coding: utf-8 -*-
"""
AutoShot 工具函数
改造自 Shot_Detection 仓库，将 ffmpeg 依赖替换为 OpenCV 以适配 ComfyUI 环境
"""

import numpy as np
import os
import cv2


def get_frames(video_file_path: str, width: int = 48, height: int = 27) -> np.ndarray:
    """
    从视频中提取帧并缩放到指定尺寸（模型输入要求 48x27 RGB）

    Args:
        video_file_path (str): 视频文件路径
        width (int, optional): 输出帧宽度，默认 48（模型输入尺寸）
        height (int, optional): 输出帧高度，默认 27（模型输入尺寸）

    Returns:
        np.ndarray: 视频帧数组，shape [N, H, W, C]，dtype uint8，RGB 顺序
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_file_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 缩放到目标尺寸
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"未能从视频中提取到任何帧: {video_file_path}")

    return np.stack(frames, axis=0)


def get_batches(frames: np.ndarray):
    """
    准备模型推理的帧批次
    对输入帧进行前后 padding，然后按 100 帧为一批进行切分

    Args:
        frames (np.ndarray): 视频帧数组，shape [N, H, W, C]

    Yields:
        np.ndarray: 批次帧，shape [100, H, W, C]
    """
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0

    # 前后各 padding 25 帧，确保模型能正确处理边界
    frames = np.concatenate(
        [frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25),
        axis=0
    )

    # 按步长 50 帧切分，每批 100 帧
    for i in range(0, len(frames) - 50, 50):
        yield frames[i:i + 100]


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    """
    将二值预测结果转换为镜头边界

    Args:
        predictions (np.ndarray): 每帧的二值预测结果

    Returns:
        np.ndarray: 镜头边界数组，shape [N, 2]，每行为 [start_frame, end_frame]
    """
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # 兜底：如果所有预测都是 1，则视为一个镜头
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)

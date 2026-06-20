# -*- coding: utf-8 -*-
"""
ComfyUI-AutoShot 节点定义
基于 Shot_Detection 仓库的 AutoShot 模型，实现视频镜头自动分割与关键帧提取

改造自 https://github.com/PhucNguyenLamp/Shot_Detection
适配 ComfyUI 节点标准接口，支持前端可视化多选关键帧
"""

import os
import torch
import numpy as np
import cv2
import hashlib

from comfy.utils import ProgressBar
#from folder_paths import get_temp_directory
from folder_paths import get_output_directory
from .autoshot_core import AutoShot


class AutoShotDetectionNode:
    """
    AutoShot 镜头分割检测节点
    基于深度学习的 TransNetV2Supernet 模型，自动检测视频镜头边界
    提取每个镜头的关键帧，支持前端可视化多选

    输入：
        - video_path: 视频文件路径
        - threshold: 镜头边界检测阈值（0.1~0.9），值越大越保守
        - min_shot_frames: 最小镜头帧数，过滤极短镜头
        - keyframe_position: 关键帧位置（首帧/中间帧/末帧）


    输出：
        - IMAGE: 关键帧图片列表（标准 ComfyUI IMAGE 格式）
        - INT_LIST: 关键帧帧号列表
        - STRING_LIST: 预览图路径列表（用于前端展示）
    """

    # 模型单例缓存（避免重复加载）
    _model_instance = None
    _model_device = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "输入视频，从 Load Video 节点接入，支持 mp4/avi/mov 等常见格式"
                    }
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "镜头边界检测阈值，数值越大越不容易判定为镜头切换"
                    }
                ),
                "min_shot_frames": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 500,
                        "tooltip": "单个镜头最少帧数，过滤极短的闪镜"
                    }
                ),
                "keyframe_position": (
                    ["middle", "first", "last"],
                    {
                        "default": "middle",
                        "tooltip": "每个镜头取哪一帧作为关键帧"
                    }
                ),
            },
            "optional": {
                "selected_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "手动选择的关键帧帧号（逗号分隔），留空则输出全部关键帧。前端预览面板点击选择后自动填充。"
                    }
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("IMAGE", "INT_LIST", "STRING_LIST")
    RETURN_NAMES = ("关键帧图片列表", "关键帧帧号列表", "预览图路径列表")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_NODE = True

    FUNCTION = "run_detection"
    CATEGORY = "Video/Analysis"
    DESCRIPTION = "基于AutoShot深度学习模型自动检测视频镜头边界，提取关键帧，支持前端可视化多选"

    @classmethod
    def _get_model(cls):
        """
        获取或创建模型单例
        避免每次执行节点都重新加载模型权重
        """
        if cls._model_instance is None:
            # 模型权重路径
            model_dir = os.path.join(os.path.dirname(__file__), "autoshot_core", "model_weight")
            weight_path = os.path.join(model_dir, "ckpt_0_200_0.pth")

            if not os.path.exists(weight_path) or os.path.getsize(weight_path) < 1024:
                raise FileNotFoundError(
                    f"模型权重文件不存在或未正确下载: {weight_path}\n"
                    f"请确保已通过 Git LFS 下载完整的模型权重文件（约 57MB）。\n"
                    f"参考命令: git lfs pull"
                )

            print(f"[AutoShot] 正在加载模型权重: {weight_path}")
            cls._model_instance = AutoShot(pretrained_path=weight_path)
            cls._model_device = cls._model_instance.device
            print(f"[AutoShot] 模型加载完成，运行设备: {cls._model_device}")

        return cls._model_instance

    def _get_cache_filename(self, video_path, unique_id, frame_idx):
        """
        生成关键帧预览图的缓存文件名
        """
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        filename = f"autoshot_{video_hash}_{unique_id}_{frame_idx}.jpg"
        return filename

    def _get_cache_path(self, video_path, unique_id, frame_idx):
        """
        生成关键帧预览图的完整缓存路径
        """
        output_dir = get_output_directory()
        filename = self._get_cache_filename(video_path, unique_id, frame_idx)
        return os.path.join(output_dir, filename)

    def _frame_to_tensor(self, frame):
        """
        将 OpenCV BGR 帧转换为 ComfyUI 标准 IMAGE 张量
        格式: [1, H, W, C]，值域 0.0~1.0，RGB 顺序
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        return tensor

    def _extract_keyframes_from_video(self, video_path, keyframe_indices):
        """
        从原视频中提取指定帧的原始分辨率图像

        Args:
            video_path: 视频文件路径
            keyframe_indices: 关键帧索引列表

        Returns:
            tuple: (图像张量列表, 预览图信息列表)
                预览图信息格式: {filename, subfolder, type}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        image_tensors = []
        preview_infos = []

        for fidx in keyframe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # 保存预览图（JPEG 压缩，用于前端快速展示）
            preview_path = self._get_cache_path(video_path, self.unique_id, fidx)
            preview_filename = self._get_cache_filename(video_path, self.unique_id, fidx)
            cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # ComfyUI 标准预览图信息格式
            preview_info = {
                "filename": preview_filename,
                "subfolder": "",
                "type": "output"
            }
            preview_infos.append(preview_info)

            # 转换为 ComfyUI 标准张量
            tensor = self._frame_to_tensor(frame)
            image_tensors.append(tensor)

        cap.release()
        return image_tensors, preview_infos

    def _filter_by_selected_indices(self, keyframe_indices, selected_indices_str):
        """
        根据用户选择的帧号过滤关键帧

        Args:
            keyframe_indices: 所有检测到的关键帧索引
            selected_indices_str: 用户选择的帧号字符串（逗号分隔）

        Returns:
            list: 过滤后的关键帧索引
        """
        if not selected_indices_str or not selected_indices_str.strip():
            return keyframe_indices

        try:
            selected = set()
            for part in selected_indices_str.split(","):
                part = part.strip()
                if part:
                    selected.add(int(part))

            if len(selected) == 0:
                return keyframe_indices

            # 只保留用户选中的帧
            filtered = [idx for idx in keyframe_indices if idx in selected]
            return filtered if len(filtered) > 0 else keyframe_indices

        except (ValueError, TypeError):
            print("[AutoShot] 警告: selected_indices 格式错误，将输出全部关键帧")
            return keyframe_indices

    def _extract_video_path(self, video_input):
        """
        从 VIDEO 类型输入中提取视频文件路径
        兼容多种 VIDEO 格式：
        - 字符串路径
        - 字典包含 path/video_path 键
        - 字典包含 filename + subfolder（ComfyUI 上传格式）
        - VideoFromFile 等视频对象
        - 其他自定义格式

        Args:
            video_input: VIDEO 类型输入

        Returns:
            str: 视频文件绝对路径
        """
        # 情况1：直接是字符串路径
        if isinstance(video_input, str):
            if os.path.exists(video_input):
                return video_input
            # 可能是相对路径，尝试在 input 目录查找
            from folder_paths import get_input_directory
            input_dir = get_input_directory()
            candidate = os.path.join(input_dir, video_input)
            if os.path.exists(candidate):
                return candidate

        # 情况2：字典格式
        if isinstance(video_input, dict):
            # 尝试常见的路径键
            path_keys = ["path", "video_path", "file_path", "filepath", "source"]
            for key in path_keys:
                if key in video_input and isinstance(video_input[key], str):
                    p = video_input[key]
                    if os.path.exists(p):
                        return p

            # ComfyUI 上传文件格式：filename + subfolder
            if "filename" in video_input:
                from folder_paths import get_input_directory, get_full_path
                filename = video_input["filename"]
                subfolder = video_input.get("subfolder", "")
                try:
                    full_path = get_full_path("input", filename, subfolder)
                    if full_path and os.path.exists(full_path):
                        return full_path
                except Exception:
                    pass

                # 兜底：手动拼接
                input_dir = get_input_directory()
                candidate = os.path.join(input_dir, subfolder, filename)
                if os.path.exists(candidate):
                    return candidate

        # 情况3：元组或列表，第一个元素是路径
        if isinstance(video_input, (list, tuple)) and len(video_input) > 0:
            if isinstance(video_input[0], str) and os.path.exists(video_input[0]):
                return video_input[0]
            # 递归处理第一个元素
            try:
                return self._extract_video_path(video_input[0])
            except Exception:
                pass

        # 情况4：视频对象（如 VideoFromFile）
        if hasattr(video_input, "__class__") and not isinstance(video_input, (str, dict, list, tuple)):
            # 尝试常见的路径属性
            path_attrs = [
                "path", "video_path", "file_path", "filepath",
                "source", "filename", "file", "video_file",
                "media_path", "input_path"
            ]
            for attr in path_attrs:
                if hasattr(video_input, attr):
                    try:
                        p = getattr(video_input, attr)
                        if isinstance(p, str) and os.path.exists(p):
                            return p
                    except Exception:
                        pass

            # 尝试常见的获取路径方法
            path_methods = [
                "get_path", "get_filepath", "get_source",
                "get_video_path", "get_filename", "path"
            ]
            for method_name in path_methods:
                if hasattr(video_input, method_name):
                    try:
                        method = getattr(video_input, method_name)
                        if callable(method):
                            p = method()
                            if isinstance(p, str) and os.path.exists(p):
                                return p
                    except Exception:
                        pass

            # 尝试访问内部的 _path 或私有属性
            private_attrs = [
                "_path", "_file_path", "_filename", "_source",
                "_filepath", "_full_path", "_video_path", "_file"
            ]
            for attr in private_attrs:
                if hasattr(video_input, attr):
                    try:
                        p = getattr(video_input, attr)
                        if isinstance(p, str) and os.path.exists(p):
                            return p
                    except Exception:
                        pass

            # 尝试 filename + subfolder 组合（ComfyUI 风格）
            if hasattr(video_input, "filename"):
                try:
                    from folder_paths import get_full_path, get_input_directory
                    filename = getattr(video_input, "filename")
                    subfolder = getattr(video_input, "subfolder", "") if hasattr(video_input, "subfolder") else ""
                    if isinstance(filename, str):
                        try:
                            full_path = get_full_path("input", filename, subfolder)
                            if full_path and os.path.exists(full_path):
                                return full_path
                        except Exception:
                            pass
                        # 兜底手动拼接
                        input_dir = get_input_directory()
                        candidate = os.path.join(input_dir, subfolder, filename)
                        if os.path.exists(candidate):
                            return candidate
                except Exception:
                    pass

            # 尝试从嵌套对象中获取（如 video.path 或 video.info.path）
            nested_attrs = ["info", "metadata", "data", "video", "source"]
            for nested_attr in nested_attrs:
                if hasattr(video_input, nested_attr):
                    try:
                        nested_obj = getattr(video_input, nested_attr)
                        # 递归处理嵌套对象
                        return self._extract_video_path(nested_obj)
                    except Exception:
                        continue

            # 尝试 str() 转换
            try:
                p = str(video_input)
                if os.path.exists(p):
                    return p
            except Exception:
                pass

            # 尝试 repr() 转换
            try:
                p = repr(video_input)
                # 从 repr 中提取路径（如 <VideoFromFile path='xxx'>）
                import re
                match = re.search(r"path=['\"](.+?)['\"]", p)
                if match:
                    candidate = match.group(1)
                    if os.path.exists(candidate):
                        return candidate
            except Exception:
                pass

            # 尝试从对象的 __dict__ 中查找所有字符串值
            if hasattr(video_input, "__dict__"):
                for key, value in video_input.__dict__.items():
                    if isinstance(value, str) and os.path.exists(value):
                        return value
                    # 递归查找嵌套字典
                    if isinstance(value, dict):
                        for v in value.values():
                            if isinstance(v, str) and os.path.exists(v):
                                return v

        # 都没找到，报错
        raise ValueError(
            f"无法从 VIDEO 输入中提取视频文件路径。\n"
            f"输入类型: {type(video_input)}\n"
            f"输入内容: {str(video_input)[:200]}\n"
            f"对象属性: {dir(video_input)[:30] if hasattr(video_input, '__dict__') else 'N/A'}\n"
            f"请确保使用 Load Video 节点加载视频后再接入本节点。"
        )

    def run_detection(self, video, threshold, min_shot_frames,
                      keyframe_position, selected_indices="",
                      unique_id=None, prompt=None, **kwargs):
        """
        主执行函数：镜头检测 + 关键帧提取 + 选择过滤

        Args:
            video: VIDEO 类型输入，视频对象或路径
            threshold: 镜头边界检测阈值
            min_shot_frames: 最小镜头帧数
            keyframe_position: 关键帧位置（first/middle/last）
            selected_indices: 用户选择的帧号（逗号分隔字符串）
            unique_id: 节点唯一 ID（ComfyUI 自动传入）
            prompt: ComfyUI prompt 对象（自动传入）
            **kwargs: 兜底接收其他可能的参数，避免版本兼容问题

        Returns:
            dict: 包含 result 和 ui 字段
                - result: (图片张量列表, 帧号列表, 预览路径列表)
                - ui: 前端展示用的数据
        """
        self.unique_id = unique_id or "default"

        # 从 VIDEO 输入中提取视频路径
        video_path = self._extract_video_path(video)
        #print(f"[AutoShot] 视频路径: {video_path}")

        # 参数校验
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 计算缓存 key（视频路径 + 检测参数）
        # 只要这些参数没变，就可以复用之前的检测结果，跳过模型推理
        cache_key = f"{video_path}_{threshold}_{min_shot_frames}_{keyframe_position}"

        # 检查是否有缓存的检测结果
        has_cache = (
            hasattr(self, '_cached_detection_key') and
            self._cached_detection_key == cache_key and
            hasattr(self, '_cached_keyframe_indices') and
            len(self._cached_keyframe_indices) > 0
        )

        if has_cache:
            # 命中缓存，直接使用之前的检测结果
            #print(f"[AutoShot] 命中检测缓存，跳过模型推理")
            keyframe_indices_all = self._cached_keyframe_indices.copy()
            total_frames = getattr(self, '_cached_total_frames', 0)
        else:
            # 未命中缓存，重新运行模型检测
            #print(f"[AutoShot] 未命中缓存，开始模型推理")
            #print(f"[AutoShot] 检测阈值: {threshold}, 最小镜头帧数: {min_shot_frames}")

            # 获取模型
            model = self._get_model()

            # 提取低分辨率帧用于模型推理（48x27）
            from .autoshot_core.utils import get_frames
            frames_lowres = get_frames(video_path)
            total_frames = len(frames_lowres)
            #print(f"[AutoShot] 视频总帧数: {total_frames}")

            # 模型推理，获取每帧的镜头边界概率
            predictions = model.detect_shots(frames_lowres)

            # 转换为镜头边界（应用阈值）
            scenes = model.predictions_to_scenes(predictions, threshold=threshold)
            #print(f"[AutoShot] 检测到 {len(scenes)} 个镜头")

            # 过滤掉过短的镜头
            filtered_scenes = []
            for scene in scenes:
                start, end = scene
                if end - start + 1 >= min_shot_frames:
                    filtered_scenes.append(scene)

            if len(filtered_scenes) == 0:
                # 兜底：如果所有镜头都太短，至少保留第一个镜头
                filtered_scenes = [scenes[0]] if len(scenes) > 0 else [[0, total_frames - 1]]

            #print(f"[AutoShot] 过滤后剩余 {len(filtered_scenes)} 个镜头")

            # 根据选择的位置计算关键帧索引（所有关键帧，未经过滤）
            keyframe_indices_all = []
            for scene in filtered_scenes:
                start, end = scene
                if keyframe_position == "first":
                    key_idx = int(start)
                elif keyframe_position == "last":
                    key_idx = int(min(end, total_frames - 1))
                else:  # middle
                    key_idx = int((start + end) // 2)
                keyframe_indices_all.append(key_idx)

            # 保存到缓存
            self._cached_detection_key = cache_key
            self._cached_keyframe_indices = keyframe_indices_all.copy()
            self._cached_total_frames = total_frames
            #print(f"[AutoShot] 检测结果已缓存")

        # 第四步：根据用户选择过滤关键帧
        keyframe_indices = self._filter_by_selected_indices(keyframe_indices_all, selected_indices)
        #print(f"[AutoShot] 输出关键帧数量: {len(keyframe_indices)} (共 {len(keyframe_indices_all)} 个)")

        # 第五步：从原视频中提取原始分辨率的关键帧
        image_tensors, preview_infos = self._extract_keyframes_from_video(
            video_path, keyframe_indices
        )

        # 兜底：确保至少有一帧输出
        if len(image_tensors) == 0:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                tensor = self._frame_to_tensor(frame)
                image_tensors.append(tensor)
                keyframe_indices = [0]
                preview_path = self._get_cache_path(video_path, self.unique_id, 0)
                preview_filename = self._get_cache_filename(video_path, self.unique_id, 0)
                cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                preview_infos.append({
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "output"
                })
            cap.release()

        # 提取预览图路径列表（用于 result 输出，保持向后兼容）
        preview_paths = [info["filename"] for info in preview_infos]

        # 返回结果
        # OUTPUT_IS_LIST=True 表示每个输出都是列表，ComfyUI 会自动展开
        return {
            "result": (image_tensors, keyframe_indices, preview_paths),
            "ui": {
                "keyframes": preview_infos,
                "frame_indices": keyframe_indices,
            }
        }
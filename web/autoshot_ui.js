/**
 * ComfyUI-AutoShot 前端扩展
 * 功能：在节点下方渲染关键帧预览缩略图，支持点击多选
 * 选中的关键帧索引会自动同步到节点的 selected_indices 参数
 *
 * 用法：
 * 1. 运行节点后自动显示所有镜头关键帧缩略图
 * 2. 点击缩略图切换选中状态（蓝色边框+勾选标记=选中）
 * 3. 可使用全选/取消全选按钮批量操作
 * 4. 点击"应用选择"按钮重新运行节点，仅输出选中的关键帧
 */

import { app } from "../../scripts/app.js";

// 扩展注册
app.registerExtension({
    name: "ComfyUI-AutoShot",

    // 节点类型定义注册完成后，给 AutoShotDetection 节点注入自定义行为
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AutoShotDetection") {
            return;
        }

        // ============================================================
        // 1. 节点创建时初始化选择状态和预览面板
        // ============================================================
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            // 初始化选中帧集合（存储帧号）
            this.selectedKeyframes = new Set();
            // 存储所有检测到的帧号列表
            this.allFrameIndices = [];
            // 存储预览图URL列表
            this.previewUrls = [];

            // 确保 selected_indices widget 存在
            // 注意：后端定义为 optional，前端可能需要手动添加
            let selectedWidget = this.widgets.find(w => w.name === "selected_indices");
            if (!selectedWidget) {
                selectedWidget = this.addWidget(
                    "text",
                    "selected_indices",
                    "",
                    (v) => { this.selected_indices = v; },
                    { multiline: false }
                );
            }

            // 创建预览面板 DOM 容器
            this.createPreviewPanel();
        };

        // ============================================================
        // 2. 创建预览面板 DOM
        // ============================================================
        nodeType.prototype.createPreviewPanel = function () {
            const self = this;

            // 主容器
            const container = document.createElement("div");
            container.className = "autoshot-preview-container";
            container.style.width = "100%";
            container.style.minHeight = "60px";
            container.style.padding = "10px";
            container.style.background = "#1a1a1a";
            container.style.borderRadius = "8px";
            container.style.marginTop = "8px";
            container.style.border = "1px solid #333";
            container.style.boxSizing = "border-box";

            // 工具栏
            const toolbar = document.createElement("div");
            toolbar.style.display = "flex";
            toolbar.style.justifyContent = "space-between";
            toolbar.style.alignItems = "center";
            toolbar.style.marginBottom = "8px";
            toolbar.style.gap = "8px";
            toolbar.style.flexWrap = "wrap";

            // 标题 + 计数
            const titleWrap = document.createElement("div");
            titleWrap.style.display = "flex";
            titleWrap.style.alignItems = "center";
            titleWrap.style.gap = "8px";

            const title = document.createElement("span");
            title.textContent = "关键帧预览";
            title.style.color = "#ddd";
            title.style.fontSize = "13px";
            title.style.fontWeight = "bold";

            const countLabel = document.createElement("span");
            countLabel.className = "autoshot-count-label";
            countLabel.textContent = "共 0 帧 / 已选 0 帧";
            countLabel.style.color = "#888";
            countLabel.style.fontSize = "11px";

            titleWrap.appendChild(title);
            titleWrap.appendChild(countLabel);

            // 按钮组
            const btnGroup = document.createElement("div");
            btnGroup.style.display = "flex";
            btnGroup.style.gap = "6px";

            // 全选按钮
            const selectAllBtn = document.createElement("button");
            selectAllBtn.textContent = "全选";
            selectAllBtn.style.padding = "4px 10px";
            selectAllBtn.style.fontSize = "11px";
            selectAllBtn.style.background = "#2a2a2a";
            selectAllBtn.style.color = "#ccc";
            selectAllBtn.style.border = "1px solid #444";
            selectAllBtn.style.borderRadius = "4px";
            selectAllBtn.style.cursor = "pointer";
            selectAllBtn.onmouseover = () => { selectAllBtn.style.background = "#3a3a3a"; };
            selectAllBtn.onmouseout = () => { selectAllBtn.style.background = "#2a2a2a"; };
            selectAllBtn.onclick = () => self.selectAllKeyframes();

            // 取消全选按钮
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "取消全选";
            clearBtn.style.padding = "4px 10px";
            clearBtn.style.fontSize = "11px";
            clearBtn.style.background = "#2a2a2a";
            clearBtn.style.color = "#ccc";
            clearBtn.style.border = "1px solid #444";
            clearBtn.style.borderRadius = "4px";
            clearBtn.style.cursor = "pointer";
            clearBtn.onmouseover = () => { clearBtn.style.background = "#3a3a3a"; };
            clearBtn.onmouseout = () => { clearBtn.style.background = "#2a2a2a"; };
            clearBtn.onclick = () => self.clearAllKeyframes();

            // 应用选择按钮（重新执行）
            const applyBtn = document.createElement("button");
            applyBtn.textContent = "应用选择";
            applyBtn.style.padding = "4px 10px";
            applyBtn.style.fontSize = "11px";
            applyBtn.style.background = "#007acc";
            applyBtn.style.color = "#fff";
            applyBtn.style.border = "1px solid #007acc";
            applyBtn.style.borderRadius = "4px";
            applyBtn.style.cursor = "pointer";
            applyBtn.onmouseover = () => { applyBtn.style.background = "#0090e0"; };
            applyBtn.onmouseout = () => { applyBtn.style.background = "#007acc"; };
            applyBtn.onclick = () => self.applySelectionAndRun();

            btnGroup.appendChild(selectAllBtn);
            btnGroup.appendChild(clearBtn);
            btnGroup.appendChild(applyBtn);

            toolbar.appendChild(titleWrap);
            toolbar.appendChild(btnGroup);

            // 缩略图滚动容器
            const thumbsWrap = document.createElement("div");
            thumbsWrap.className = "autoshot-thumbs-wrap";
            thumbsWrap.style.display = "flex";
            thumbsWrap.style.flexWrap = "wrap";
            thumbsWrap.style.gap = "6px";
            thumbsWrap.style.maxHeight = "240px";
            thumbsWrap.style.overflowY = "auto";
            thumbsWrap.style.padding = "4px";

            // 空状态提示
            const emptyHint = document.createElement("div");
            emptyHint.className = "autoshot-empty-hint";
            emptyHint.textContent = "运行节点后，此处将显示检测到的关键帧缩略图";
            emptyHint.style.color = "#666";
            emptyHint.style.fontSize = "12px";
            emptyHint.style.textAlign = "center";
            emptyHint.style.padding = "20px";
            emptyHint.style.width = "100%";
            thumbsWrap.appendChild(emptyHint);

            container.appendChild(toolbar);
            container.appendChild(thumbsWrap);

            // 保存引用
            this.previewContainer = container;
            this.thumbsWrap = thumbsWrap;
            this.countLabel = countLabel;

            // 添加为 DOM widget
            this.addDOMWidget("autoshot_preview", "div", container, {
                serialize: false,
                hideOnZoom: false,
            });

            // 调整节点大小以适应预览面板
            requestAnimationFrame(() => {
                this.setSize([Math.max(this.size[0], 320), this.size[1] + 220]);
            });
        };

        // ============================================================
        // 3. 节点执行完成后，更新预览图
        // ============================================================
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            // 从 ui 输出中获取预览图和帧号
            let keyframes = [];
            let frameIndices = [];

            if (output && output.keyframes) {
                keyframes = output.keyframes;
            }
            if (output && output.frame_indices) {
                frameIndices = output.frame_indices;
            }

            // 如果没有从 ui 拿到数据，尝试从 result 中取
            if (keyframes.length === 0 && output && output.result) {
                if (Array.isArray(output.result) && output.result[2]) {
                    keyframes = output.result[2];
                }
                if (Array.isArray(output.result) && output.result[1]) {
                    frameIndices = output.result[1];
                }
            }

            // 保存数据
            this.previewUrls = keyframes;
            this.allFrameIndices = frameIndices;

            // 每次运行都重置选择状态，默认全选所有关键帧
            // 避免上次的选择残留到新的检测结果中
            const selectedWidget = this.widgets.find(w => w.name === "selected_indices");
            this.selectedKeyframes = new Set(frameIndices);
            if (selectedWidget) {
                selectedWidget.value = frameIndices.join(",");
            }

            // 渲染缩略图
            this.renderThumbnails();
        };

        // ============================================================
        // 4. 构建预览图 URL
        // ============================================================
        nodeType.prototype.getPreviewUrl = function (previewInfo) {
            // 根据 ComfyUI 标准格式构建预览图 URL
            if (typeof previewInfo === "string") {
                // 如果是字符串路径，直接返回（兼容旧格式）
                return previewInfo;
            }
            // 对象格式：{filename, subfolder, type}
            if (previewInfo && previewInfo.filename) {
                const params = new URLSearchParams();
                params.append("filename", previewInfo.filename);
                params.append("subfolder", previewInfo.subfolder || "");
                params.append("type", previewInfo.type || "output");
                return `/view?${params.toString()}`;
            }
            return "";
        };

        // ============================================================
        // 5. 渲染缩略图
        // ============================================================
        nodeType.prototype.renderThumbnails = function () {
            const self = this;

            // 清空现有内容
            this.thumbsWrap.innerHTML = "";

            if (this.previewUrls.length === 0) {
                // 显示空状态
                const emptyHint = document.createElement("div");
                emptyHint.textContent = "运行节点后，此处将显示检测到的关键帧缩略图";
                emptyHint.style.color = "#666";
                emptyHint.style.fontSize = "12px";
                emptyHint.style.textAlign = "center";
                emptyHint.style.padding = "20px";
                emptyHint.style.width = "100%";
                this.thumbsWrap.appendChild(emptyHint);
                this.updateCountLabel();
                return;
            }

            // 渲染每张缩略图
            this.previewUrls.forEach((previewInfo, index) => {
                const frameIdx = self.allFrameIndices[index] ?? index;
                const url = self.getPreviewUrl(previewInfo);

                const wrapper = document.createElement("div");
                wrapper.className = "autoshot-thumb-wrapper";
                wrapper.style.position = "relative";
                wrapper.style.cursor = "pointer";
                wrapper.style.flexShrink = "0";
                wrapper.dataset.frameIndex = frameIdx;

                // 缩略图
                const thumb = document.createElement("img");
                thumb.src = url;
                thumb.style.width = "100px";
                thumb.style.height = "auto";
                thumb.style.border = "2px solid transparent";
                thumb.style.borderRadius = "4px";
                thumb.style.objectFit = "cover";
                thumb.style.transition = "border-color 0.15s";
                thumb.loading = "lazy";

                // 帧号标签
                const label = document.createElement("span");
                label.style.position = "absolute";
                label.style.bottom = "4px";
                label.style.left = "4px";
                label.style.background = "rgba(0, 0, 0, 0.75)";
                label.style.color = "#fff";
                label.style.fontSize = "10px";
                label.style.padding = "2px 5px";
                label.style.borderRadius = "3px";
                label.style.pointerEvents = "none";
                label.textContent = `#${frameIdx}`;

                // 选中勾选标记
                const checkmark = document.createElement("div");
                checkmark.style.position = "absolute";
                checkmark.style.top = "4px";
                checkmark.style.right = "4px";
                checkmark.style.width = "18px";
                checkmark.style.height = "18px";
                checkmark.style.borderRadius = "50%";
                checkmark.style.background = "#007acc";
                checkmark.style.color = "#fff";
                checkmark.style.fontSize = "12px";
                checkmark.style.display = "none";
                checkmark.style.alignItems = "center";
                checkmark.style.justifyContent = "center";
                checkmark.style.fontWeight = "bold";
                checkmark.style.pointerEvents = "none";
                checkmark.textContent = "✓";

                wrapper.appendChild(thumb);
                wrapper.appendChild(label);
                wrapper.appendChild(checkmark);

                // 设置初始选中状态
                if (self.selectedKeyframes.has(frameIdx)) {
                    thumb.style.border = "2px solid #007acc";
                    checkmark.style.display = "flex";
                }

                // 点击切换选中
                wrapper.onclick = () => {
                    self.toggleKeyframeSelection(frameIdx, thumb, checkmark);
                };

                self.thumbsWrap.appendChild(wrapper);
            });

            this.updateCountLabel();
        };

        // ============================================================
        // 5. 切换单帧选中状态
        // ============================================================
        nodeType.prototype.toggleKeyframeSelection = function (frameIdx, thumbEl, checkEl) {
            if (this.selectedKeyframes.has(frameIdx)) {
                // 取消选中
                this.selectedKeyframes.delete(frameIdx);
                thumbEl.style.border = "2px solid transparent";
                checkEl.style.display = "none";
            } else {
                // 选中
                this.selectedKeyframes.add(frameIdx);
                thumbEl.style.border = "2px solid #007acc";
                checkEl.style.display = "flex";
            }

            // 同步到 widget
            this.syncSelectionToWidget();
            this.updateCountLabel();
        };

        // ============================================================
        // 6. 全选
        // ============================================================
        nodeType.prototype.selectAllKeyframes = function () {
            this.allFrameIndices.forEach(idx => this.selectedKeyframes.add(idx));
            this.renderThumbnails();
            this.syncSelectionToWidget();
        };

        // ============================================================
        // 7. 取消全选
        // ============================================================
        nodeType.prototype.clearAllKeyframes = function () {
            this.selectedKeyframes.clear();
            this.renderThumbnails();
            this.syncSelectionToWidget();
        };

        // ============================================================
        // 8. 将选中状态同步到 selected_indices widget
        // ============================================================
        nodeType.prototype.syncSelectionToWidget = function () {
            const selectedWidget = this.widgets.find(w => w.name === "selected_indices");
            if (selectedWidget) {
                const sorted = Array.from(this.selectedKeyframes).sort((a, b) => a - b);
                selectedWidget.value = sorted.join(",");
                // 触发 widget 回调
                if (selectedWidget.callback) {
                    selectedWidget.callback(selectedWidget.value);
                }
                // 标记节点为已修改
                if (this.graph) {
                    this.graph.setDirtyCanvas(true, false);
                }
            }
        };

        // ============================================================
        // 9. 更新计数标签
        // ============================================================
        nodeType.prototype.updateCountLabel = function () {
            if (this.countLabel) {
                this.countLabel.textContent = `共 ${this.allFrameIndices.length} 帧 / 已选 ${this.selectedKeyframes.size} 帧`;
            }
        };

        // ============================================================
        // 10. 应用选择并重新执行
        // ============================================================
        nodeType.prototype.applySelectionAndRun = function () {
            // 先同步选择到 widget
            this.syncSelectionToWidget();

            // 触发队列执行
            try {
                app.queuePrompt(0, 1);
            } catch (e) {
                console.warn("[AutoShot] 自动执行失败，请手动点击 Queue Prompt:", e);
                alert("应用选择成功，请手动点击 'Queue Prompt' 重新运行节点");
            }
        };
    },
});

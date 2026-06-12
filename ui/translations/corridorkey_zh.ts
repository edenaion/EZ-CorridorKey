<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN" sourcelanguage="en_US">
<context>
    <name>BatchPipelineDialog</name>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="69"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="96"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="500"/>
        <source>Batch Pipeline</source>
        <translation>批处理流程</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="101"/>
        <source>Select a folder containing video clips. Files with &quot;alphahint&quot; or &quot;maskhint&quot; in the name are automatically paired as hints.</source>
        <translation>选择包含视频片段的文件夹。文件名中含 &quot;alphahint&quot; 或 &quot;maskhint&quot; 的文件会自动配对为提示。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="111"/>
        <source>Select Folder...</source>
        <translation>选择文件夹...</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="115"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="462"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="501"/>
        <source>No folder selected</source>
        <translation>未选择文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="122"/>
        <source>Global Settings</source>
        <translation>全局设置</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="127"/>
        <source>No-hint clips:</source>
        <translation>无提示片段：</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="130"/>
        <source>Alpha generation method for clips with no companion hint file.
GVM: fast automatic alpha.
BiRefNet: higher quality, select a model variant.</source>
        <translation>用于无配套提示文件片段的 Alpha 生成方式。
GVM：快速自动 Alpha。
BiRefNet：质量更高，需选择模型变体。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="157"/>
        <source>MaskHint clips:</source>
        <translation>MaskHint 片段：</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="160"/>
        <source>Mask refinement method for clips with a companion MaskHint file.
VideoMaMa: temporal consistency, best for video.
MatAnyone2: single-frame matting with mask guidance.</source>
        <translation>用于有配套 MaskHint 文件片段的蒙版精修方式。
VideoMaMa：时序一致性，最适合视频素材。
MatAnyone2：基于蒙版引导的单帧抠像。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="176"/>
        <source>Per-clip overrides</source>
        <translation>逐片段覆盖设置</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Clip</source>
        <translation>片段</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Detected</source>
        <translation>已检测</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Pipeline</source>
        <translation>处理流程</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Status</source>
        <translation>状态</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="204"/>
        <source>Clear Pipeline</source>
        <translation>清空流程</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="205"/>
        <source>Cancel all pending batch jobs and reset.</source>
        <translation>取消所有等待中的批处理任务并重置。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="210"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="509"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="213"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="508"/>
        <source>Run Batch</source>
        <translation>开始批处理</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="216"/>
        <source>Inference settings (despill, refiner, edge, color space, etc.) are inherited from the right panel. Adjust them there before running.</source>
        <translation>推理设置（去溢色、精修、边缘、色彩空间等）继承自右侧面板。请在运行前在右侧面板中调整。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="234"/>
        <source>Select Batch Folder</source>
        <translation>选择批处理文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="267"/>
        <source>No hint</source>
        <translation>无提示</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="268"/>
        <source>AlphaHint</source>
        <translation>AlphaHint</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="269"/>
        <source>MaskHint</source>
        <translation>MaskHint</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="281"/>
        <source>CK Inference</source>
        <translation>CK 推理</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="299"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="330"/>
        <source>→ CK</source>
        <translation>→ CK</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="355"/>
        <source>Found %d clip(s): %s</source>
        <translation>找到 %d 个片段：%s</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="356"/>
        <source>No video clips found in this folder.</source>
        <translation>此文件夹中未找到视频片段。</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="476"/>
        <source>Batch Pipeline - Processing</source>
        <translation>批处理流程 - 处理中</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="478"/>
        <source>Running...</source>
        <translation>运行中...</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="569"/>
        <source>Processing failed</source>
        <translation>处理失败</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="576"/>
        <source>Batch Pipeline - Complete</source>
        <translation>批处理流程 - 已完成</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="577"/>
        <source>Done</source>
        <translation>完成</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="579"/>
        <source>Close</source>
        <translation>关闭</translation>
    </message>
</context>
<context>
    <name>DebugConsoleWidget</name>
    <message>
        <location filename="../widgets/debug_console.py" line="86"/>
        <source>Console</source>
        <translation>控制台</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="129"/>
        <source>CONSOLE</source>
        <translation>控制台</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="172"/>
        <source>Level:</source>
        <translation>级别：</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="178"/>
        <location filename="../widgets/debug_console.py" line="334"/>
        <source>Pause</source>
        <translation>暂停</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="185"/>
        <source>Clear</source>
        <translation>清空</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="334"/>
        <source>Resume</source>
        <translation>继续</translation>
    </message>
</context>
<context>
    <name>DiagnosticDialog</name>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="47"/>
        <source>Diagnostic: %s</source>
        <translation>诊断：%s</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="108"/>
        <source>Error: %s</source>
        <translation>错误：%s</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="122"/>
        <source>Report Issue on GitHub</source>
        <translation>在 GitHub 上报告问题</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="129"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
</context>
<context>
    <name>FrameScrubber</name>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="52"/>
        <source>Go to first frame</source>
        <translation>跳转到第一帧</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="60"/>
        <source>Previous frame</source>
        <translation>上一帧</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="68"/>
        <source>Play / Pause (Space)</source>
        <translation>播放 / 暂停 (Space)</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="82"/>
        <source>Coverage bar — shows which frames have been processed.
Green lane: painted frames (brush strokes).
White lane: alpha hint coverage.
Yellow lane: inference output coverage.</source>
        <translation>覆盖指示条 — 显示哪些帧已处理。
绿色色带：已绘制帧（画笔笔触）。
白色色带：Alpha 提示覆盖范围。
黄色色带：推理输出覆盖范围。</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="95"/>
        <source>Scrub through frames. Scroll wheel or Left/Right to step.</source>
        <translation>拖动浏览帧。滚轮或左/右方向键逐帧步进。</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="118"/>
        <source>Next frame</source>
        <translation>下一帧</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="126"/>
        <source>Go to last frame</source>
        <translation>跳转到最后一帧</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="310"/>
        <source>Pause (Space)</source>
        <translation>暂停 (Space)</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="317"/>
        <source>Play (Space)</source>
        <translation>播放 (Space)</translation>
    </message>
</context>
<context>
    <name>HotkeysDialog</name>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="133"/>
        <source>Hotkeys</source>
        <translation>快捷键</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="145"/>
        <source>Filter shortcuts...</source>
        <translation>筛选快捷键...</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="212"/>
        <source>Reset</source>
        <translation>重置</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="220"/>
        <source>Reset to default: %s</source>
        <translation>重置为默认值：%s</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="240"/>
        <source>Reset All to Defaults</source>
        <translation>全部恢复默认</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="252"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="263"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="309"/>
        <source>Reset All Shortcuts</source>
        <translation>重置所有快捷键</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="310"/>
        <source>Reset all shortcuts to their default values?</source>
        <translation>将所有快捷键恢复为默认值？</translation>
    </message>
</context>
<context>
    <name>IOTrayActionsMixin</name>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="63"/>
        <source>Run Extraction (%d clips)</source>
        <translation>提取帧（%d 个片段）</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="64"/>
        <source>Run Extraction</source>
        <translation>提取帧</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="72"/>
        <source>Rename...</source>
        <translation>重命名...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="78"/>
        <source>Finder</source>
        <translation>访达</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="78"/>
        <source>Explorer</source>
        <translation>文件资源管理器</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="79"/>
        <source>Open in %s</source>
        <translation>在 %s 中打开</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="89"/>
        <source>Clear Mask (%d clips)</source>
        <translation>清除蒙版（%d 个片段）</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="89"/>
        <location filename="../widgets/io_tray_actions.py" line="232"/>
        <source>Clear Mask</source>
        <translation>清除蒙版</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="97"/>
        <source>Clear Alpha (%d clips)</source>
        <translation>清除 Alpha（%d 个片段）</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="97"/>
        <location filename="../widgets/io_tray_actions.py" line="341"/>
        <source>Clear Alpha</source>
        <translation>清除 Alpha</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="105"/>
        <source>Clear Outputs (%d clips)</source>
        <translation>清除输出（%d 个片段）</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="105"/>
        <location filename="../widgets/io_tray_actions.py" line="373"/>
        <source>Clear Outputs</source>
        <translation>清除输出</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="113"/>
        <source>Clear All (%d clips)</source>
        <translation>全部清除（%d 个片段）</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="113"/>
        <location filename="../widgets/io_tray_actions.py" line="296"/>
        <source>Clear All</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="120"/>
        <source>Set Output Directory...</source>
        <translation>设置输出目录...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="126"/>
        <source>Clear Output Directory Override</source>
        <translation>清除输出目录覆盖</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="133"/>
        <source>Remove (%d clips)...</source>
        <translation>移除（%d 个片段）...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="133"/>
        <source>Remove...</source>
        <translation>移除...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="157"/>
        <source>Export %s as Video...</source>
        <translation>将 %s 导出为视频...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="169"/>
        <source>Open Containing Folder</source>
        <translation>打开所在文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="183"/>
        <source>Output Directory for &apos;%s&apos;</source>
        <translation>&apos;%s&apos; 的输出目录</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="214"/>
        <source>Rename Clip</source>
        <translation>重命名片段</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="214"/>
        <source>New name:</source>
        <translation>新名称：</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="233"/>
        <source>Delete tracked masks for %d clip(s)?
%s

This will remove all SAM2 mask frames from disk.</source>
        <translation>删除 %d 个片段的跟踪蒙版？
%s

此操作将从磁盘移除所有 SAM2 蒙版帧。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="297"/>
        <source>Remove ALL generated data for %d clip(s)?
%s

This will delete masks, alpha hints, and all output frames.</source>
        <translation>移除 %d 个片段的所有生成数据？
%s

此操作将删除蒙版、Alpha 提示及所有输出帧。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="342"/>
        <source>Delete AlphaHint for %d clip(s)?
%s

This will remove all generated alpha hint frames from disk.</source>
        <translation>删除 %d 个片段的 AlphaHint？
%s

此操作将从磁盘移除所有生成的 Alpha 提示帧。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="374"/>
        <source>Remove all output files for %d clip(s)?
%s

This will delete FG, Matte, Comp, and Processed frames.</source>
        <translation>移除 %d 个片段的所有输出文件？
%s

此操作将删除 FG、Matte、Comp 和 Processed 帧。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="400"/>
        <source>Remove %d clip(s)?</source>
        <translation>移除 %d 个片段？</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="404"/>
        <source>
... and %d more</source>
        <translation>
... 以及另外 %d 个</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="409"/>
        <source>How would you like to remove %d clip(s)?</source>
        <translation>如何移除 %d 个片段？</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="412"/>
        <source>Remove from List</source>
        <translation>从列表中移除</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="413"/>
        <source>Delete from Disk</source>
        <translation>从磁盘删除</translation>
    </message>
</context>
<context>
    <name>IOTrayPanel</name>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="81"/>
        <source>INPUT (0)</source>
        <translation>输入 (0)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="86"/>
        <source>RESET I/O</source>
        <translation>重置出入点</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="88"/>
        <source>Clear in/out markers on all clips</source>
        <translation>清除所有片段的出入点标记</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="92"/>
        <source>+ ADD</source>
        <translation>+ 添加</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="94"/>
        <source>Import clips — choose a folder or video file(s)</source>
        <translation>导入片段 — 选择文件夹或视频文件</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="122"/>
        <source>EXPORTS (0)</source>
        <translation>导出 (0)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="169"/>
        <source>Import Folder...</source>
        <translation>导入文件夹...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="170"/>
        <source>Import Video(s)...</source>
        <translation>导入视频...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="171"/>
        <source>Import Image Sequence...</source>
        <translation>导入图像序列...</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="180"/>
        <source>No Markers</source>
        <translation>无标记</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="181"/>
        <source>No clips have in/out markers set.</source>
        <translation>所有片段均未设置出入点标记。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="188"/>
        <source>Reset In/Out Markers</source>
        <translation>重置出入点标记</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="189"/>
        <source>This will clear in/out markers on %d clip(s).

All clips will revert to full-clip processing.
Continue?</source>
        <translation>此操作将清除 %d 个片段的出入点标记。

所有片段将恢复为整段处理。
是否继续？</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="199"/>
        <source>Confirm Reset</source>
        <translation>确认重置</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="200"/>
        <source>Are you sure? This cannot be undone.

Clearing in/out markers on %d clip(s).</source>
        <translation>确认执行？此操作无法撤销。

将清除 %d 个片段的出入点标记。</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="212"/>
        <source>Select Clips Directory</source>
        <translation>选择片段目录</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="220"/>
        <source>Select Video Files</source>
        <translation>选择视频文件</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="228"/>
        <source>Select Image Sequence Folder</source>
        <translation>选择图像序列文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="359"/>
        <source>INPUT (%d)</source>
        <translation>输入 (%d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="360"/>
        <source>EXPORTS (%d)</source>
        <translation>导出 (%d)</translation>
    </message>
</context>
<context>
    <name>KeyBindButton</name>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="56"/>
        <source>(none)</source>
        <translation>（无）</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="62"/>
        <source>Press a key...</source>
        <translation>按下按键...</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="94"/>
        <source>Shortcut Conflict</source>
        <translation>快捷键冲突</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="95"/>
        <source>&quot;%s&quot; is already assigned to:
%s

Reassign anyway? The conflicting binding will be cleared.</source>
        <translation>&quot;%s&quot; 已分配给：
%s

仍然重新分配？冲突的绑定将被清除。</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../main_window.py" line="263"/>
        <source>%s — Mac Performance Warning</source>
        <translation>%s — Mac 性能警告</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="265"/>
        <source>GPU-intensive features (SAM2, GVM, VideoMaMa, MatAnyone2) are very slow on Mac (Apple Silicon MPS).

This may take hours for longer clips and could freeze your system.

Recommendation: Import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke instead.

Continue anyway? (This warning won&apos;t appear again this session.)</source>
        <translation>GPU 密集型功能（SAM2、GVM、VideoMaMa、MatAnyone2）在 Mac（Apple Silicon MPS）上速度非常慢。

较长片段可能需要数小时，且可能导致系统卡顿。

建议：改为从 After Effects、DaVinci Resolve 或 Nuke 导入预制的 Alpha 遮罩。

仍然继续？（本会话期间不再显示此警告。）</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="283"/>
        <source>EZ-CorridorKey</source>
        <translation>EZ-CorridorKey</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="439"/>
        <source>Detected GPU used for inference</source>
        <translation>检测到的推理 GPU</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="442"/>
        <source>VRAM</source>
        <translation>VRAM</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="453"/>
        <source>GPU video memory usage — updates during inference</source>
        <translation>GPU 显存占用 — 推理期间实时更新</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="459"/>
        <source>Current VRAM used / total available</source>
        <translation>当前已用显存 / 总可用显存</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="651"/>
        <source>No GPU</source>
        <translation>无 GPU</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="658"/>
        <source>Memory</source>
        <translation>内存</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="659"/>
        <source>Unified memory usage — CPU and GPU share the same pool</source>
        <translation>统一内存占用 — CPU 与 GPU 共享同一内存池</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="660"/>
        <source>Current unified memory used / total available</source>
        <translation>当前已用统一内存 / 总可用统一内存</translation>
    </message>
</context>
<context>
    <name>ParameterPanel</name>
    <message>
        <location filename="../widgets/parameter_panel.py" line="132"/>
        <source>ALPHA GENERATION</source>
        <translation>Alpha 生成</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="137"/>
        <source>Manual</source>
        <translation>手动</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="142"/>
        <source>CHROMA KEY</source>
        <translation>色度键</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="147"/>
        <source>Generate alpha hints using a traditional chroma keyer.
Best for clean green/blue screen shots.
No GPU or AI model required — instant processing.

Click to expand parameters, then click GENERATE.
Hotkey: `</source>
        <translation>使用传统色度抠像器生成 Alpha 提示。
适合干净的绿幕/蓝幕镜头。
无需 GPU 或 AI 模型 — 即时处理。

点击展开参数，然后点击生成。
快捷键：`</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="166"/>
        <source> Pick Screen Color</source>
        <translation> 拾取幕布颜色</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="169"/>
        <source>Click on the viewer to sample the screen color.
Works on either the input or output viewport.
Hotkey: E</source>
        <translation>在查看器上点击以取样幕布颜色。
适用于输入或输出查看器。
快捷键：E</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="181"/>
        <source>Sampled screen color</source>
        <translation>已取样的幕布颜色</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="186"/>
        <source>Key Strength: 1.0</source>
        <translation>抠像强度：1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="191"/>
        <source>How aggressively to key the screen color. Higher = more separation.</source>
        <translation>抠除幕布颜色的力度。值越高，前景与背景分离越明显。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="193"/>
        <source>Key Strength: %s</source>
        <translation>抠像强度：%s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="199"/>
        <source>Clip Black: 0.0</source>
        <translation>剪切黑色：0.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="204"/>
        <source>Push near-transparent values to fully transparent.
Cleans up noise in background areas.</source>
        <translation>将接近透明的值推至完全透明。
清除背景区域中的噪点。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="206"/>
        <source>Clip Black: %s</source>
        <translation>剪切黑色：%s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="212"/>
        <source>Clip White: 1.0</source>
        <translation>剪切白色：1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="217"/>
        <source>Push near-opaque values to fully opaque.
Solidifies the foreground core.</source>
        <translation>将接近不透明的值推至完全不透明。
使前景主体更为实心。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="219"/>
        <source>Clip White: %s</source>
        <translation>剪切白色：%s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="231"/>
        <source>Shrink/Grow</source>
        <translation>收缩/扩展</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="236"/>
        <source>Erode (negative) or dilate (positive) the matte edge.
0 = no change.</source>
        <translation>腐蚀（负值）或膨胀（正值）遮罩边缘。
0 = 不变。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="240"/>
        <source>Edge Blur</source>
        <translation>边缘模糊</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="245"/>
        <source>Gaussian blur radius for softening matte edges.
0 = no blur.</source>
        <translation>柔化遮罩边缘的高斯模糊半径。
0 = 不模糊。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="252"/>
        <source>GENERATE</source>
        <translation>生成</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="253"/>
        <source>Generate alpha hint frames for the entire clip using these chroma key settings.</source>
        <translation>使用当前色度键设置为整个片段生成 Alpha 提示帧。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="271"/>
        <source>Automatic</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="278"/>
        <source>APPLE VISION</source>
        <translation>APPLE VISION</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="282"/>
        <source>Auto-generate alpha hint using Apple Vision (Neural Engine).
Detects foreground subjects automatically.
macOS 14+ only. Runs on Apple Neural Engine (fast, no GPU needed).</source>
        <translation>使用 Apple Vision（神经网络引擎）自动生成 Alpha 提示。
自动检测前景主体。
仅支持 macOS 14+。在 Apple 神经网络引擎上运行（速度快，无需 GPU）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="300"/>
        <source>GVM AUTO</source>
        <translation>GVM 自动</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="304"/>
        <source>Auto-generate alpha hint for the entire clip.
Uses GVM to predict foreground/background separation.
Available when clip is in RAW state (frames extracted).</source>
        <translation>为整个片段自动生成 Alpha 提示。
使用 GVM 预测前景/背景分离。
片段处于 RAW（已提取帧）状态时可用。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="315"/>
        <source>BIREFNET</source>
        <translation>BIREFNET</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="319"/>
        <source>Auto-generate alpha hint using BiRefNet.
Fully automatic — no painting or annotation needed.
Downloads the selected model variant on first use.

Matting: Best for hair/transparency detail (recommended).
Portrait: Optimized for human close-ups.
General: Balanced foreground/background separation.
HR variants: For 2K/4K footage (uses more VRAM).</source>
        <translation>使用 BiRefNet 自动生成 Alpha 提示。
全自动 — 无需绘制或标注。
首次使用时下载所选模型变体。

Matting：最适合发丝/透明细节（推荐）。
Portrait：针对人物特写优化。
General：均衡的前景/背景分离。
HR 变体：适用于 2K/4K 素材（占用更多显存）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="334"/>
        <source>BiRefNet model variant — changes take effect on next run.</source>
        <translation>BiRefNet 模型变体 — 下次运行时生效。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="355"/>
        <source>Requires brushstrokes</source>
        <translation>需要先绘制笔触</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="360"/>
        <source>Paint subject with 1, background with 2</source>
        <translation>按 1 绘制主体，按 2 绘制背景</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="366"/>
        <source>TRACK MASK</source>
        <translation>跟踪蒙版</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="370"/>
        <source>Use SAM2 to turn painted prompts into a dense mask track.
Required before running MatAnyone2 or VideoMaMa.

HOW TO USE:
1. Press 1 to select the GREEN brush (foreground — subject to keep)
2. Press 2 to select the RED brush (background — area to remove)
3. Paint strokes on the left viewer over your footage
4. Click TRACK MASK to preview SAM2 on the painted frame
5. If the preview looks right, confirm to propagate across all frames

TIPS:
Shift + Left-drag up/down: change brush size
Alt + Left-drag: draw a straight line between two points
Ctrl+Z: undo last stroke</source>
        <translation>使用 SAM2 将绘制的提示笔触转换为逐帧蒙版序列。
运行 MatAnyone2 或 VideoMaMa 前必须执行此步骤。

使用方法：
1. 按 1 选择绿色画笔（前景 — 需要保留的主体）
2. 按 2 选择红色画笔（背景 — 需要去除的区域）
3. 在左侧查看器的素材上绘制笔触
4. 点击跟踪蒙版，预览 SAM2 对已绘制帧的结果
5. 预览效果满意后，确认传播到所有帧

技巧：
Shift + 左键上下拖动：调整画笔大小
Alt + 左键拖动：在两点之间绘制直线
Ctrl+Z：撤销上一笔</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="600"/>
        <source>Edge refinement strength (0.0-3.0).
Scales the CNN refiner&apos;s edge corrections.
1.0 = default, 0.0 = backbone only (no refinement),
higher = sharper edges but may introduce artifacts.</source>
        <translation>边缘精修强度（0.0-3.0）。
调整 CNN 精修模块的边缘修正幅度。
1.0 = 默认，0.0 = 仅主干网络（不精修），
值越高边缘越锐利，但可能引入瑕疵。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="391"/>
        <source>MATANYONE2</source>
        <translation>MATANYONE2</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="395"/>
        <source>Generate alpha hints using MatAnyone2 video matting.
Requires paint strokes on the FIRST FRAME (frame 1).

1. Navigate to frame 1 (the very first frame)
2. Paint foreground (hotkey 1) and background (hotkey 2)
3. Click Track Mask to generate dense masks with SAM2
4. Click MATANYONE2 to generate temporally coherent AlphaHint</source>
        <translation>使用 MatAnyone2 视频抠像生成 Alpha 提示。
需要在第一帧（帧 1）绘制笔触。

1. 导航到帧 1（第一帧）
2. 绘制前景（快捷键 1）和背景（快捷键 2）
3. 点击跟踪蒙版，用 SAM2 生成逐帧蒙版序列
4. 点击 MATANYONE2 生成时序一致的 AlphaHint</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="406"/>
        <source>VIDEOMAMA</source>
        <translation>VIDEOMAMA</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="410"/>
        <source>Generate alpha hints from a dense VideoMaMa mask track.

1. Paint sparse foreground/background prompts
2. Click Track Mask to generate dense masks with SAM2
3. Click VIDEOMAMA to generate AlphaHint</source>
        <translation>从 VideoMaMa 逐帧蒙版序列生成 Alpha 提示。

1. 绘制稀疏的前景/背景提示笔触
2. 点击跟踪蒙版，用 SAM2 生成逐帧蒙版序列
3. 点击 VIDEOMAMA 生成 AlphaHint</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="433"/>
        <source>Import your own mask for VideoMaMa.

Bypasses the Track Mask step. Select a folder or
video of grayscale masks and they will be used as
VideoMaMa&apos;s guidance input directly.</source>
        <translation>为 VideoMaMa 导入自定义蒙版。

跳过跟踪蒙版步骤。选择包含灰度蒙版的文件夹或视频，
将直接作为 VideoMaMa 的引导输入。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="462"/>
        <source>IMPORT ALPHA</source>
        <translation>导入 Alpha</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="466"/>
        <source>Import alpha hints from an image folder or video file.
Supports: PNG/JPG/TIF/EXR sequences, or MOV/MP4/ProRes video.
White = foreground, black = background.
Files are copied into the clip&apos;s AlphaHint/ folder
and the clip advances to READY state for inference.</source>
        <translation>从图像文件夹或视频文件导入 Alpha 提示。
支持：PNG/JPG/TIF/EXR 序列，或 MOV/MP4/ProRes 视频。
白色 = 前景，黑色 = 背景。
文件将复制到片段的 AlphaHint/ 文件夹，
片段随即进入 READY（待推理）状态。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="479"/>
        <source>INFERENCE</source>
        <translation>推理</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="492"/>
        <source>BG Color</source>
        <translation>幕布颜色</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="495"/>
        <source>Background screen color for this clip.

Auto: detected from the middle frame of the clip.
Green: force green screen processing.
Blue: force blue screen processing.

Controls which checkpoint, despill math, and spill
detection are used. Also changes the UI accent color.</source>
        <translation>此片段的背景幕布颜色。

自动：从片段中间帧自动检测。
绿色：强制绿幕处理。
蓝色：强制蓝幕处理。

决定所使用的模型权重、去溢色算法及溢色检测方式，并同步更新界面强调色。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Auto</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Green</source>
        <translation>绿色</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Blue</source>
        <translation>蓝色</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="511"/>
        <source>Color Space</source>
        <translation>色彩空间</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="515"/>
        <source>sRGB</source>
        <translation>sRGB</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="515"/>
        <source>Linear</source>
        <translation>线性</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="525"/>
        <source>Removes small floating noise and speckles from the
alpha by discarding isolated regions smaller than the
size threshold.</source>
        <translation>通过丢弃小于尺寸阈值的孤立区域，去除 Alpha 中的细小浮动杂点。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="547"/>
        <source>Garbage Matte</source>
        <translation>Garbage Matte</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="551"/>
        <source>Expands the alpha hint by N pixels, then zeros out
anything in the predicted matte that falls outside
that expanded region. Removes edge-of-frame artifacts
and background gunk that inference leaves behind.</source>
        <translation>将 Alpha 提示向外扩展 N 像素，然后将预测遮罩中超出扩展区域的部分置零。
可去除帧边缘瑕疵及推理残留的背景杂质。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="565"/>
        <source>Pixel expansion around the alpha hint.
Higher = more breathing room around subject edges.
Lower = tighter crop to the hint boundary.</source>
        <translation>Alpha 提示周围的像素扩展量。
值越高，主体边缘周围的余量越大。
值越低，裁切越贴近提示边界。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="577"/>
        <source>Despill: 0.5</source>
        <translation>Despill: 0.5</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="584"/>
        <source>Screen spill removal strength (0.0-1.0).
Removes background color bleed from hair, skin, and edges.
1.0 = full despill, 0.0 = no despill (keep original colors).</source>
        <translation>幕布溢色去除强度（0.0-1.0）。
去除发丝、皮肤和边缘的背景颜色溢出。
1.0 = 完全去溢色，0.0 = 不去溢色（保留原始颜色）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="521"/>
        <source>Despeckle</source>
        <translation>去除杂点</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="538"/>
        <source>Minimum area (in pixels) for a region to survive.
Isolated alpha blobs smaller than this are removed.
Lower = keep more detail, higher = cleaner matte.</source>
        <translation>区域存活所需的最小面积（像素）。
小于此值的孤立 Alpha 区域将被移除。
值越低保留细节越多，值越高遮罩越干净。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="593"/>
        <source>Refiner: 1.0</source>
        <translation>精修：1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="610"/>
        <source>Live Preview</source>
        <translation>实时预览</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="618"/>
        <source>OUTPUT</source>
        <translation>输出</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="624"/>
        <source>FG</source>
        <translation>FG</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="628"/>
        <source>Foreground — despilled subject on black background.
Screen spill removed from hair and edges.
Straight alpha (not premultiplied).</source>
        <translation>前景（FG）— 黑色背景上已去溢色的主体。
发丝和边缘的幕布溢色已去除。
直接 Alpha（非预乘）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="637"/>
        <location filename="../widgets/parameter_panel.py" line="656"/>
        <source>EXR = 32-bit float (post-production).
PNG = 8-bit (general use).</source>
        <translation>EXR = 32 位浮点（后期制作）。
PNG = 8 位（通用）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="643"/>
        <source>Matte</source>
        <translation>Matte</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="647"/>
        <source>Alpha matte — grayscale transparency map.
White = fully opaque, black = fully transparent.
Use in compositing software for manual keying control.</source>
        <translation>Alpha 遮罩（Matte）— 灰度透明度图。
白色 = 完全不透明，黑色 = 完全透明。
在合成软件中用于手动抠像控制。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="662"/>
        <source>Comp</source>
        <translation>Comp</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="666"/>
        <source>Composite — final keyed result over checkerboard.
Best representation of the key quality.
Colors match the original input faithfully.</source>
        <translation>合成（Comp）— 最终抠像结果叠加在棋盘格背景上。
最佳抠像质量预览。
颜色与原始输入高度吻合。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="675"/>
        <source>PNG = 8-bit with transparency.
EXR = 32-bit float (post-production).</source>
        <translation>PNG = 8 位带透明通道。
EXR = 32 位浮点（后期制作）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="681"/>
        <source>Processed</source>
        <translation>Processed</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="685"/>
        <source>Processed — production-ready RGBA (straight, linear).
Designed for import into Resolve, Premiere, and compositing tools.
Includes despill + garbage matte cleanup applied.</source>
        <translation>成品 RGBA（Processed）— 可直接用于制作的 RGBA（直接 Alpha，线性色彩空间）。
专为导入 Resolve、Premiere 及合成软件而设计。
已应用去溢色和垃圾遮罩清理。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="694"/>
        <source>EXR = 32-bit float (recommended for Processed).
PNG = 8-bit (lossy for straight linear RGBA).</source>
        <translation>EXR = 32 位浮点（推荐用于 Processed）。
PNG = 8 位（直接线性 RGBA 有损）。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="701"/>
        <source>PERFORMANCE</source>
        <translation>性能</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="706"/>
        <source>Parallel frames</source>
        <translation>并行帧数</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="712"/>
        <source>Process multiple frames simultaneously using parallel engines.

Each extra engine loads a full copy of the model.
CUDA: ~6-8 GB VRAM per engine.

Default: 1 (safest). Try 2 first, then increase if stable.

EXPERIMENTAL: Values above 8 are for high-memory CUDA systems
(e.g. RTX 6000).
If you run out of memory, the app will automatically scale
back to however many engines fit.

CUDA only right now. Not currently supported on Apple Silicon.</source>
        <translation>使用并行引擎同时处理多帧。

每个额外引擎会加载一份完整模型副本。
CUDA：每个引擎约需 6-8 GB 显存。

默认：1（最安全）。建议先尝试 2，稳定后再继续增加。

实验性：8 以上适用于大显存 CUDA 系统（如 RTX 6000）。
若显存不足，应用将自动缩减至可容纳的引擎数量。

目前仅支持 CUDA，不支持 Apple Silicon。</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="797"/>
        <source>Despill: %s</source>
        <translation>Despill: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="802"/>
        <source>Refiner: %s</source>
        <translation>精修：%s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="939"/>
        <source>Painted: %d / %d frames</source>
        <translation>已绘制：%d / %d 帧</translation>
    </message>
</context>
<context>
    <name>PreferencesDialog</name>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="152"/>
        <source>Preferences</source>
        <translation>首选项</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="185"/>
        <source>User Interface</source>
        <translation>用户界面</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="188"/>
        <source>Show tooltips on controls</source>
        <translation>在控件上显示工具提示</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="194"/>
        <source>UI sounds</source>
        <translation>界面音效</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="200"/>
        <source>Language</source>
        <translation>语言</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="203"/>
        <source>English</source>
        <translation>English</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="209"/>
        <source>Select display language. Restart required to apply.</source>
        <translation>选择显示语言。需重启后生效。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="216"/>
        <source>Project</source>
        <translation>项目</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="219"/>
        <source>Copy source videos into project folder</source>
        <translation>将源视频复制到项目文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="222"/>
        <source>When enabled, imported videos are copied into the project folder.
When disabled, the project references the original file in place.

Note: Deleting a project never touches the original source file.</source>
        <translation>启用后，导入的视频将复制到项目文件夹。
禁用时，项目直接引用原始文件。

注意：删除项目不会影响原始源文件。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="232"/>
        <source>Copy imported image sequences into project folder</source>
        <translation>将导入的图像序列复制到项目文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="235"/>
        <source>When enabled, imported image sequence files are copied into the project.
When disabled (default), the project references the original files in place.

Referencing saves disk space for large EXR/TIF sequences.
Original files are never modified regardless of this setting.</source>
        <translation>启用后，导入的图像序列文件将复制到项目。
禁用时（默认），项目直接引用原始文件。

引用方式可为大型 EXR/TIF 序列节省磁盘空间。
无论此设置如何，原始文件均不会被修改。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="249"/>
        <source>Output</source>
        <translation>输出</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="252"/>
        <source>EXR compression</source>
        <translation>EXR 压缩</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="263"/>
        <source>Compression used when writing EXR output files.

DWAB: Lossy wavelet, smallest files. Default.
PIZ: Lossless wavelet, preferred by compositors.
ZIP: Lossless deflate, good for clean renders.
None: No compression, fastest write, largest files.</source>
        <translation>写入 EXR 输出文件时使用的压缩方式。

DWAB：有损小波，文件最小。默认。
PIZ：无损小波，合成师首选。
ZIP：无损 deflate，适合干净渲染。
无：不压缩，写入最快，文件最大。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="273"/>
        <source>Default output directory</source>
        <translation>默认输出目录</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="279"/>
        <source>Default (inside project)</source>
        <translation>默认（项目内部）</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="285"/>
        <source>Global default directory for inference output.

When set, outputs go to:
  &lt;this folder&gt;/&lt;ProjectName&gt;/&lt;ClipName&gt;/FG, Matte, etc.

Leave empty to use the default (Output/ inside each clip).
Per-clip overrides (right-click → Set Output Directory) take priority.</source>
        <translation>推理输出的全局默认目录。

设置后，输出路径为：
  &lt;此文件夹&gt;/&lt;项目名称&gt;/&lt;片段名称&gt;/FG、Matte 等。

留空则使用默认路径（各片段内的 Output/ 文件夹）。
逐片段覆盖（右键 → 设置输出目录）优先级更高。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="294"/>
        <location filename="../widgets/preferences_dialog.py" line="478"/>
        <source>Browse...</source>
        <translation>浏览...</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="299"/>
        <source>Clear</source>
        <translation>清除</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="309"/>
        <source>Inference</source>
        <translation>推理</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="312"/>
        <source>Model resolution</source>
        <translation>模型分辨率</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="316"/>
        <source>2048 — Full Quality</source>
        <translation>2048 — 完整质量</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="317"/>
        <source>1024 — Faster, Less Detail</source>
        <translation>1024 — 更快，细节较少</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="323"/>
        <source>Resolution the model processes internally before upscaling to your frame size.
Applies to all backends (CUDA, MPS, MLX, CPU).

2048: Full quality — captures fine hair strands and edge detail.
Matches the original CorridorKey quality. Recommended for CUDA with 8GB+ VRAM.
WARNING: Very slow on Apple Silicon (needs 20GB+ memory).

1024: Faster inference with lower memory usage.
Fine hair detail may be lost. Recommended for Apple Silicon / low-VRAM GPUs.

Changing this requires an engine reload (happens automatically).</source>
        <translation>模型在放大到实际帧尺寸之前内部处理所用的分辨率。
适用于所有后端（CUDA、MPS、MLX、CPU）。

2048：完整质量 — 捕捉细腻发丝和边缘细节。
与原版 CorridorKey 质量一致。推荐用于 8GB+ 显存的 CUDA。
警告：在 Apple Silicon 上速度极慢（需要 20GB+ 内存）。

1024：推理更快，内存占用更低。
细腻发丝细节可能丢失。推荐用于 Apple Silicon 或低显存 GPU。

更改此项需要重新加载引擎（自动执行）。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="338"/>
        <source>Processing backend</source>
        <translation>处理后端</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="342"/>
        <source>Auto — MLX if available, otherwise MPS</source>
        <translation>自动 — 优先 MLX，否则使用 MPS</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="343"/>
        <source>MLX — Apple Metal acceleration (recommended)</source>
        <translation>MLX — Apple Metal 加速（推荐）</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="344"/>
        <source>MPS — PyTorch Metal Performance Shaders</source>
        <translation>MPS — PyTorch Metal Performance Shaders</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="350"/>
        <source>Choose the inference backend for Apple Silicon.

MLX: Native Apple Metal — fastest on M1/M2/M3/M4.
MPS: PyTorch Metal Performance Shaders — compatible fallback.
Auto: Uses MLX if installed, otherwise falls back to MPS.

Changing this requires an engine reload (happens automatically).</source>
        <translation>选择 Apple Silicon 的推理后端。

MLX：原生 Apple Metal — 在 M1/M2/M3/M4 上速度最快。
MPS：PyTorch Metal Performance Shaders — 兼容回退方案。
自动：已安装 MLX 时使用 MLX，否则回退到 MPS。

更改此项需要重新加载引擎（自动执行）。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="362"/>
        <source>Playback</source>
        <translation>播放</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="365"/>
        <source>Loop playback within in/out range</source>
        <translation>在出入点范围内循环播放</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="368"/>
        <source>When enabled, playback loops back to the in-point
after reaching the out-point (or start/end if no range).</source>
        <translation>启用后，播放到出点后将循环回到入点
（未设置出入点范围时，从片段末尾循环回到开头）。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="380"/>
        <source>Tracking</source>
        <translation>跟踪</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="383"/>
        <source>SAM2 model</source>
        <translation>SAM2 模型</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="389"/>
        <source>%s  (%s)</source>
        <translation>%s  (%s)</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="394"/>
        <source>Fast: lower VRAM, lower quality.
Base+: best default tradeoff for this app.
Highest Quality: slowest, heaviest tracker.</source>
        <translation>Fast：显存占用低，质量较低。
Base+：本应用最佳默认平衡选项。
Highest Quality：最慢，跟踪器资源占用最高。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="403"/>
        <source>Models download automatically on first use. Download progress appears in the status bar.</source>
        <translation>模型在首次使用时自动下载。下载进度显示在状态栏中。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="411"/>
        <source>Manage models</source>
        <translation>管理模型</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="423"/>
        <source>Open Cache Folder</source>
        <translation>打开缓存文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="432"/>
        <source>Video Tools</source>
        <translation>视频工具</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="435"/>
        <source>FFmpeg status</source>
        <translation>FFmpeg 状态</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="446"/>
        <source>Windows: Repair downloads a bundled full FFmpeg build into tools/ffmpeg without changing your system install.
macOS: Repair installs FFmpeg via Homebrew.
Linux: Repair copies the install command to your clipboard.</source>
        <translation>Windows：修复将完整 FFmpeg 构建下载到 tools/ffmpeg，不更改系统安装。
macOS：修复通过 Homebrew 安装 FFmpeg。
Linux：修复将安装命令复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="464"/>
        <location filename="../widgets/preferences_dialog.py" line="714"/>
        <location filename="../widgets/preferences_dialog.py" line="737"/>
        <source>Repair FFmpeg</source>
        <translation>修复 FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="467"/>
        <source>Windows: download and install a full bundled FFmpeg build into tools/ffmpeg, validate ffmpeg + ffprobe 7+, and switch CorridorKey to that local copy immediately.

macOS: install FFmpeg via Homebrew and validate ffmpeg + ffprobe 7+.

Linux: do not change system packages. CorridorKey shows the exact install commands and copies them to your clipboard instead.</source>
        <translation>Windows：下载并安装完整 FFmpeg 到 tools/ffmpeg，验证 ffmpeg + ffprobe 7+，并立即切换到本地副本。

macOS：通过 Homebrew 安装 FFmpeg，并验证 ffmpeg + ffprobe 7+。

Linux：不修改系统包。CorridorKey 显示精确的安装命令并复制到剪贴板。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="481"/>
        <source>Point CorridorKey at your own FFmpeg installation.
Select the folder containing ffmpeg.exe and ffprobe.exe.</source>
        <translation>将 CorridorKey 指向自定义 FFmpeg 安装路径。
选择包含 ffmpeg.exe 和 ffprobe.exe 的文件夹。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="488"/>
        <source>Open FFmpeg Folder</source>
        <translation>打开 FFmpeg 文件夹</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="491"/>
        <source>Open CorridorKey&apos;s bundled FFmpeg folder.
If Repair FFmpeg has been run on Windows, this is where the local full build is stored.</source>
        <translation>打开 CorridorKey 内置的 FFmpeg 文件夹。
在 Windows 上执行修复 FFmpeg 后，本地完整构建存储于此处。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="520"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="524"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="595"/>
        <source>Select Default Output Directory</source>
        <translation>选择默认输出目录</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="610"/>
        <source>Select FFmpeg Folder (containing ffmpeg.exe and ffprobe.exe)</source>
        <translation>选择 FFmpeg 文件夹（包含 ffmpeg.exe 和 ffprobe.exe）</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="633"/>
        <source>FFmpeg Not Found</source>
        <translation>未找到 FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="634"/>
        <source>Could not find ffmpeg%s in:

%s

Select the folder that contains ffmpeg.exe and ffprobe.exe (usually the &apos;bin&apos; folder inside the FFmpeg download).</source>
        <translation>在以下路径中找不到 ffmpeg%s：

%s

请选择包含 ffmpeg.exe 和 ffprobe.exe 的文件夹（通常为 FFmpeg 下载包内的 &apos;bin&apos; 文件夹）。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="643"/>
        <source>FFprobe Missing</source>
        <translation>缺少 FFprobe</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="644"/>
        <source>Found ffmpeg%s but ffprobe%s is missing from:

%s

CorridorKey requires both. Download a full FFmpeg build.</source>
        <translation>找到 ffmpeg%s，但以下路径缺少 ffprobe%s：

%s

CorridorKey 需要两者兼备。请下载完整 FFmpeg 构建。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="657"/>
        <source>FFmpeg Found</source>
        <translation>已找到 FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="661"/>
        <source>FFmpeg Issue</source>
        <translation>FFmpeg 异常</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="703"/>
        <source>FFmpeg OK</source>
        <translation>FFmpeg 正常</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="704"/>
        <source>%s

No repair is needed.</source>
        <translation>%s

无需修复。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="716"/>
        <source>

The install command has been copied to your clipboard.
Paste it into a terminal to install.</source>
        <translation>
安装命令已复制到剪贴板。
在终端中粘贴并执行以完成安装。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="724"/>
        <source>CorridorKey will download and install a full bundled FFmpeg build into:

%s

This does not modify your system-wide FFmpeg.

Continue?</source>
        <translation>CorridorKey 将下载并安装完整 FFmpeg 构建到：

%s

此操作不会修改系统级 FFmpeg。

是否继续？</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="730"/>
        <source>CorridorKey will install FFmpeg via Homebrew:

    brew install ffmpeg

Continue?</source>
        <translation>CorridorKey 将通过 Homebrew 安装 FFmpeg：

    brew install ffmpeg

是否继续？</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="744"/>
        <source>Preparing repair...</source>
        <translation>正在准备修复...</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="747"/>
        <source>Repairing FFmpeg...</source>
        <translation>正在修复 FFmpeg...</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="781"/>
        <source>FFmpeg Repaired</source>
        <translation>FFmpeg 已修复</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="782"/>
        <source>%s

CorridorKey will use FFmpeg immediately.</source>
        <translation>%s

CorridorKey 将立即使用 FFmpeg。</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="789"/>
        <source>FFmpeg Repair Failed</source>
        <translation>FFmpeg 修复失败</translation>
    </message>
</context>
<context>
    <name>PreviewViewport</name>
    <message>
        <location filename="../widgets/preview_viewport.py" line="235"/>
        <source>Extracting frames...
%s</source>
        <translation>正在提取帧...
%s</translation>
    </message>
    <message>
        <location filename="../widgets/preview_viewport.py" line="261"/>
        <source>Selected: %s
State: %s</source>
        <translation>已选择：%s
状态：%s</translation>
    </message>
    <message>
        <location filename="../widgets/preview_viewport.py" line="402"/>
        <source>Toggle A/B wipe comparison (hotkey: A)

Overlays input (A) and current output (B) in one viewer
with a diagonal divider line.

Drag the center handle to slide the line.
Drag above or below the handle to rotate the angle.
Scroll wheel to slide the line (Shift+scroll for fine-grain).
Middle-click the line to reset to default.</source>
        <translation>切换 A/B 划像对比（快捷键：A）

在单个查看器中叠加显示输入（A）和当前输出（B），
以斜向分割线分隔。

拖动中心控制点可移动分割线。
在控制点上方或下方拖动可旋转角度。
滚轮移动分割线（Shift+ 滚轮可微调）。
中键点击分割线可恢复默认。</translation>
    </message>
    <message>
        <location filename="../widgets/preview_viewport.py" line="560"/>
        <source>No frame available for stem %d</source>
        <translation>通道 %d 无可用帧</translation>
    </message>
</context>
<context>
    <name>QueuePanel</name>
    <message>
        <location filename="../widgets/queue_panel.py" line="101"/>
        <source>Toggle queue panel (Q)</source>
        <translation>切换队列面板 (Q)</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="138"/>
        <source>QUEUE</source>
        <translation>队列</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="153"/>
        <source>Clear</source>
        <translation>清空</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="162"/>
        <source>Clear completed and cancelled jobs</source>
        <translation>清除已完成和已取消的任务</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="336"/>
        <source>Dismiss</source>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="408"/>
        <source>Processing...</source>
        <translation>处理中...</translation>
    </message>
</context>
<context>
    <name>RecentProjectCard</name>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="57"/>
        <source>Open in Finder</source>
        <translation>在访达中打开</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="57"/>
        <source>Open in Explorer</source>
        <translation>在文件资源管理器中打开</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="66"/>
        <source>Remove project</source>
        <translation>移除项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="110"/>
        <source>Rename Project</source>
        <translation>重命名项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="116"/>
        <source>Delete Project</source>
        <translation>删除项目</translation>
    </message>
</context>
<context>
    <name>RecentProjectsPanel</name>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="139"/>
        <source>RECENT PROJECTS</source>
        <translation>最近项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="161"/>
        <source>No recent projects</source>
        <translation>暂无最近项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="211"/>
        <source>Rename Project</source>
        <translation>重命名项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="211"/>
        <source>Project name:</source>
        <translation>项目名称：</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="238"/>
        <source>Remove Project</source>
        <translation>移除项目</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="239"/>
        <source>Remove &quot;%s&quot; from recent projects?</source>
        <translation>从最近项目中移除 &quot;%s&quot;？</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="241"/>
        <source>Remove from List: hides it from recents (files stay on disk).
Delete from Disk: permanently deletes the project folder.</source>
        <translation>从列表中移除：从最近项目中隐藏（文件保留在磁盘上）。
从磁盘删除：永久删除项目文件夹。</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="245"/>
        <source>Remove from List</source>
        <translation>从列表中移除</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="246"/>
        <source>Delete from Disk</source>
        <translation>从磁盘删除</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="260"/>
        <source>Confirm Delete</source>
        <translation>确认删除</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="261"/>
        <source>Permanently delete this project folder?

%s</source>
        <translation>永久删除此项目文件夹？

%s</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="283"/>
        <source>Delete Failed</source>
        <translation>删除失败</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="284"/>
        <source>Could not delete project:
%s</source>
        <translation>无法删除项目：
%s</translation>
    </message>
</context>
<context>
    <name>ReportIssueDialog</name>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="118"/>
        <source>Report Issue</source>
        <translation>报告问题</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="130"/>
        <source>Issue title:</source>
        <translation>问题标题：</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="132"/>
        <source>Brief summary of the problem</source>
        <translation>问题的简要描述</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="140"/>
        <source>What happened?</source>
        <translation>发生了什么？</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="144"/>
        <source>Describe what you were doing and what went wrong.
Steps to reproduce are very helpful.</source>
        <translation>描述您当时的操作及发生的问题。
提供复现步骤会非常有帮助。</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="156"/>
        <source>System info (auto-collected, included in report)</source>
        <translation>系统信息（自动收集，包含在报告中）</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="171"/>
        <source>This will open GitHub in your browser. A free GitHub account is required to submit issues. Your report is also copied to the clipboard in case you need to paste it after logging in.</source>
        <translation>此操作将在浏览器中打开 GitHub。提交问题需要免费 GitHub 账号。您的报告已复制到剪贴板，以便登录后粘贴。</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="184"/>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="188"/>
        <source>Open GitHub</source>
        <translation>打开 GitHub</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="293"/>
        <source>Bug Report</source>
        <translation>错误报告</translation>
    </message>
</context>
<context>
    <name>SetupWizard</name>
    <message>
        <location filename="../widgets/setup_wizard.py" line="652"/>
        <source>EZ-CorridorKey Setup</source>
        <translation>EZ-CorridorKey 安装向导</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="674"/>
        <source>Select which models to download. The core CorridorKey model is required.
Optional models can be downloaded later from Edit → Download Manager.</source>
        <translation>选择要下载的模型。核心 CorridorKey 模型为必选项。
可选模型可稍后通过编辑 → 下载管理器下载。</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="704"/>
        <source>Browse...</source>
        <translation>浏览...</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="713"/>
        <source>Default Location</source>
        <translation>默认位置</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="686"/>
        <source>Data directory (models, projects, frame cache):</source>
        <translation>数据目录（模型、项目、帧缓存）：</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="716"/>
        <source>Reset the data directory to the platform default (in case you changed it and want to return).</source>
        <translation>将数据目录重置为平台默认值（适用于已更改后想恢复的情况）。</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="757"/>
        <source>Create Desktop shortcut</source>
        <translation>创建桌面快捷方式</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="775"/>
        <source>Cancel &amp;&amp; Exit</source>
        <translation>取消并退出</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="786"/>
        <source>Download &amp;&amp; Install</source>
        <translation>下载并安装</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="832"/>
        <source>Choose Install Location</source>
        <translation>选择安装位置</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="846"/>
        <source>Cancelling...</source>
        <translation>正在取消...</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="872"/>
        <source>Preparing downloads (0/%d)...</source>
        <translation>正在准备下载（0/%d）...</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="887"/>
        <source>Downloading %d/%d: %s...</source>
        <translation>正在下载 %d/%d：%s...</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="913"/>
        <source>All %d downloads complete!</source>
        <translation>全部 %d 个下载完成。</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="917"/>
        <source>Some downloads failed. You can retry from Edit → Download Manager.</source>
        <translation>部分下载失败。可通过编辑 → 下载管理器重试。</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="925"/>
        <source>Continue</source>
        <translation>继续</translation>
    </message>
</context>
<context>
    <name>SplitViewWidget</name>
    <message>
        <location filename="../widgets/split_view.py" line="512"/>
        <source>Extracting frames...</source>
        <translation>正在提取帧...</translation>
    </message>
    <message>
        <location filename="../widgets/split_view.py" line="539"/>
        <source>%d%%  (%d/%d frames)</source>
        <translation>%d%%  (%d/%d 帧)</translation>
    </message>
</context>
<context>
    <name>StartupDiagnosticDialog</name>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="159"/>
        <source>Startup Diagnostics</source>
        <translation>启动诊断</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="169"/>
        <source>EZ-CorridorKey detected issues with your environment that may prevent some features from working correctly.</source>
        <translation>EZ-CorridorKey 检测到您的运行环境存在问题，部分功能可能无法正常使用。</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="197"/>
        <source>Continue Anyway</source>
        <translation>仍然继续</translation>
    </message>
</context>
<context>
    <name>StatusBar</name>
    <message>
        <location filename="../widgets/status_bar.py" line="88"/>
        <source>Inference progress for the current job</source>
        <translation>当前任务的推理进度</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="114"/>
        <location filename="../widgets/status_bar.py" line="251"/>
        <source>RUN INFERENCE</source>
        <translation>运行推理</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="121"/>
        <source>Run AI keying on the selected clip (Ctrl+R).
Requires a READY or COMPLETE clip with alpha hints.
Respects in/out range if set (I/O hotkeys).</source>
        <translation>对所选片段运行 AI 抠像（Ctrl+R）。
需要处于 READY（待推理）或 COMPLETE（已完成）状态且包含 Alpha 提示的片段。
已设置出入点范围时按范围处理（出入点快捷键 I/O）。</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="139"/>
        <source>RESUME</source>
        <translation>继续推理</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="145"/>
        <source>Resume inference — skip already-processed frames,
fill in remaining gaps across the full clip.</source>
        <translation>继续推理 — 跳过已处理帧，
填补整个片段的剩余空缺。</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="154"/>
        <location filename="../widgets/status_bar.py" line="203"/>
        <source>STOP</source>
        <translation>停止</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="158"/>
        <location filename="../widgets/status_bar.py" line="207"/>
        <source>Stop the current job (Escape).
Already-processed frames are kept on disk.</source>
        <translation>停止当前任务（Escape）。
已处理的帧将保留在磁盘上。</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="194"/>
        <source>FORCE STOP</source>
        <translation>强制停止</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="198"/>
        <source>The current GPU step is blocked.
Force Stop will relaunch the app to break the stuck job.</source>
        <translation>当前 GPU 步骤已阻塞。
强制停止将重新启动应用以中断卡住的任务。</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="235"/>
        <source>RUN EXTRACTION</source>
        <translation>提取帧</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="239"/>
        <source>RUN PIPELINE</source>
        <translation>运行流程</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="243"/>
        <source>RUN %d CLIPS</source>
        <translation>运行 %d 个片段</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="247"/>
        <source>RUN SELECTED</source>
        <translation>运行所选</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="345"/>
        <source>1 warning</source>
        <translation>1 个警告</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="347"/>
        <source>%d warnings</source>
        <translation>%d 个警告</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="353"/>
        <source>Latest:
%s

Click for all warnings</source>
        <translation>最新：
%s

点击查看所有警告</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="406"/>
        <source>Warnings (%d)</source>
        <translation>警告（%d）</translation>
    </message>
</context>
<context>
    <name>ViewModeBar</name>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="85"/>
        <source>Original input footage (unprocessed)

Hotkey: F1</source>
        <translation>原始输入素材（未处理）

快捷键：F1</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="87"/>
        <source>Tracked mask — SAM2 segmentation output.
White = foreground, black = background.
This is the binary mask before MatAnyone2/VideoMaMa refinement.

Hotkey: F2</source>
        <translation>跟踪蒙版 — SAM2 分割结果。
白色 = 前景，黑色 = 背景。
这是 MatAnyone2/VideoMaMa 精修前的二值蒙版。

快捷键：F2</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="93"/>
        <source>Alpha hint — generated by GVM, VideoMaMa, or MatAnyone2.
White = foreground, black = background.
This is the pre-inference guide used by CorridorKey.

Hotkey: F3</source>
        <translation>Alpha 提示 — 由 GVM、VideoMaMa 或 MatAnyone2 生成。
白色 = 前景，黑色 = 背景。
这是 CorridorKey 推理前使用的引导图。

快捷键：F3</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="99"/>
        <source>Foreground — subject with screen spill removed.
Colors may look shifted; this is the despilled intermediate.

Hotkey: F4</source>
        <translation>前景（FG）— 已去除幕布溢色的主体。
颜色可能略有偏移，这是去溢色后的中间结果。

快捷键：F4</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="104"/>
        <source>Alpha matte — white = opaque, black = transparent.
Shows the AI&apos;s confidence in foreground vs background.

Hotkey: F5</source>
        <translation>Alpha 遮罩（Matte）— 白色 = 不透明，黑色 = 透明。
显示 AI 对前景与背景判断的置信度。

快捷键：F5</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="109"/>
        <source>Composite — final keyed result over checkerboard.
Best preview of key quality with faithful colors.

Hotkey: F6</source>
        <translation>合成（Comp）— 最终抠像结果叠加在棋盘格背景上。
颜色忠实还原，为抠像质量的最佳预览。

快捷键：F6</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="114"/>
        <source>Processed — production RGBA (straight, linear).
For Resolve, Premiere, and compositing tools.
Preview composites the stored image over black.
Final compositing should happen in your compositor of choice.

Hotkey: F7</source>
        <translation>成品 RGBA（Processed）— 制作级 RGBA（直接 Alpha，线性色彩空间）。
适用于 Resolve、Premiere 及合成软件。
预览时将存储图像叠加在黑色背景上。
最终合成请在所选合成软件中完成。

快捷键：F7</translation>
    </message>
</context>
<context>
    <name>VolumeControl</name>
    <message>
        <location filename="../widgets/volume_control.py" line="32"/>
        <source>Click to mute / unmute</source>
        <translation>点击静音 / 取消静音</translation>
    </message>
    <message>
        <location filename="../widgets/volume_control.py" line="46"/>
        <source>Volume</source>
        <translation>音量</translation>
    </message>
</context>
<context>
    <name>WelcomeScreen</name>
    <message>
        <location filename="../widgets/welcome_screen.py" line="175"/>
        <source>Select Media Files</source>
        <translation>选择媒体文件</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../widgets/welcome_screen.py" line="85"/>
        <source>Drop Videos, Image Sequences, or Click to Import</source>
        <translation>拖入视频、图像序列，或点击导入</translation>
    </message>
    <message>
        <location filename="../widgets/welcome_screen.py" line="93"/>
        <source>Browse...</source>
        <translation>浏览...</translation>
    </message>
</context>
<context>
    <name>_ModelRow</name>
    <message>
        <location filename="../widgets/setup_wizard.py" line="603"/>
        <source>  — Installed</source>
        <translation>  — 已安装</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="625"/>
        <source>Downloading...</source>
        <translation>正在下载...</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="632"/>
        <source>%d / %d MB</source>
        <translation>%d / %d MB</translation>
    </message>
</context>
</TS>

<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW" sourcelanguage="en_US">
    <context>
        <name>BackendStatus</name>
        <message>
            <location filename="../state_labels.py" line="58" />
            <source>Loading model...</source>
            <translation>正在載入模型...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="59" />
            <source>Loading frames...</source>
            <translation>正在載入影格...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="60" />
            <source>Loading masks...</source>
            <translation>正在載入遮罩...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="61" />
            <source>Loading preview frame...</source>
            <translation>正在載入預覽影格...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="62" />
            <source>Loading first-frame mask...</source>
            <translation>正在載入第一影格遮罩...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="63" />
            <source>Loading state dict...</source>
            <translation>正在載入權重字典...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="64" />
            <source>Loading checkpoint weights...</source>
            <translation>正在載入檢查點權重...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="65" />
            <source>Loading MatAnyone2 checkpoint...</source>
            <translation>正在載入 MatAnyone2 檢查點...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="66" />
            <source>Loading MatAnyone2 model...</source>
            <translation>正在載入 MatAnyone2 模型...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="67" />
            <source>Initializing model backbone...</source>
            <translation>正在初始化模型骨幹...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="68" />
            <source>Moving model to GPU...</source>
            <translation>正在將模型移至 GPU...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="69" />
            <source>Patching attention blocks...</source>
            <translation>正在修補注意力區塊...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="70" />
            <source>Compiling model (first run may take a minute)...</source>
            <translation>正在編譯模型（首次執行可能需要一分鐘）...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="71" />
            <source>Compiling (first frame may take a minute)...</source>
            <translation>正在編譯（第一影格可能需要一分鐘）...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="72" />
            <source>Model ready</source>
            <translation>模型就緒</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="73" />
            <source>BiRefNet model ready</source>
            <translation>BiRefNet 模型就緒</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="74" />
            <source>MatAnyone2 model ready</source>
            <translation>MatAnyone2 模型就緒</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="75" />
            <source>Running SAM2 tracker...</source>
            <translation>正在執行 SAM2 追蹤器...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="76" />
            <source>Running BiRefNet inference...</source>
            <translation>正在執行 BiRefNet 推理...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="77" />
            <source>Running MatAnyone2 inference...</source>
            <translation>正在執行 MatAnyone2 推理...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="78" />
            <source>Previewing SAM2 on annotated frame...</source>
            <translation>正在標註影格上預覽 SAM2...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="79" />
            <source>Finalizing alpha hints...</source>
            <translation>正在完成 Alpha 提示...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="80" />
            <source>Releasing Python references...</source>
            <translation>正在釋放 Python 參照...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="81" />
            <source>Waiting for CUDA to finish...</source>
            <translation>正在等待 CUDA 完成...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="82" />
            <source>Clearing CUDA cache...</source>
            <translation>正在清除 CUDA 快取...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="83" />
            <source>UNet forward pass</source>
            <translation>UNet 前向推理</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="84" />
            <source>VAE encode</source>
            <translation>VAE 編碼</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="85" />
            <source>CLIP encode</source>
            <translation>CLIP 編碼</translation>
        </message>
    </context>
    <context>
        <name>BatchPipelineDialog</name>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="69" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="96" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="500" />
            <source>Batch Pipeline</source>
            <translation>批次流程</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="101" />
            <source>Select a folder containing video clips. Files with "alphahint" or "maskhint" in the name are automatically paired as hints.</source>
            <translation>選擇包含影片片段的資料夾。檔案名稱中含有「alphahint」或「maskhint」的檔案將自動配對為提示。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="111" />
            <source>Select Folder...</source>
            <translation>選擇資料夾...</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="115" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="462" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="501" />
            <source>No folder selected</source>
            <translation>未選擇資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="122" />
            <source>Global Settings</source>
            <translation>全域設定</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="127" />
            <source>No-hint clips:</source>
            <translation>無提示片段：</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="130" />
            <source>Alpha generation method for clips with no companion hint file.
GVM: fast automatic alpha.
BiRefNet: higher quality, select a model variant.</source>
            <translation>用於無配套提示檔案片段的 Alpha 產生方式。
GVM：快速自動 Alpha。
BiRefNet：品質更高，需選擇模型變體。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="157" />
            <source>MaskHint clips:</source>
            <translation>MaskHint 片段：</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="160" />
            <source>Mask refinement method for clips with a companion MaskHint file.
VideoMaMa: temporal consistency, best for video.
MatAnyone2: single-frame matting with mask guidance.</source>
            <translation>有配套 MaskHint 檔案片段的遮罩精修方式。
VideoMaMa：時間一致性，最適合影片。
MatAnyone2：單影格去背搭配遮罩引導。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="176" />
            <source>Per-clip overrides</source>
            <translation>逐片段覆蓋設定</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Clip</source>
            <translation>片段</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Detected</source>
            <translation>已偵測</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Pipeline</source>
            <translation>流程</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Status</source>
            <translation>狀態</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="204" />
            <source>Clear Pipeline</source>
            <translation>清除流程</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="205" />
            <source>Cancel all pending batch jobs and reset.</source>
            <translation>取消所有待執行的批次工作並重置。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="210" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="509" />
            <source>Cancel</source>
            <translation>取消</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="213" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="508" />
            <source>Run Batch</source>
            <translation>執行批次</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="216" />
            <source>Inference settings (despill, refiner, edge, color space, etc.) are inherited from the right panel. Adjust them there before running.</source>
            <translation>推理設定（去溢色、精修、邊緣、色彩空間等）繼承自右側面板。執行前請先在該處調整。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="234" />
            <source>Select Batch Folder</source>
            <translation>選擇批次資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="267" />
            <source>No hint</source>
            <translation>無提示</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="268" />
            <source>AlphaHint</source>
            <translation>AlphaHint</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="269" />
            <source>MaskHint</source>
            <translation>MaskHint</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="281" />
            <source>CK Inference</source>
            <translation>CK 推理</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="299" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="330" />
            <source>→ CK</source>
            <translation>→ CK</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="355" />
            <source>Found %d clip(s): %s</source>
            <translation>找到 %d 個片段：%s</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="356" />
            <source>No video clips found in this folder.</source>
            <translation>此資料夾中未找到影片片段。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="476" />
            <source>Batch Pipeline - Processing</source>
            <translation>批次流程 - 處理中</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="478" />
            <source>Running...</source>
            <translation>執行中...</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="569" />
            <source>Processing failed</source>
            <translation>處理失敗</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="576" />
            <source>Batch Pipeline - Complete</source>
            <translation>批次流程 - 完成</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="577" />
            <source>Done</source>
            <translation>完成</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="579" />
            <source>Close</source>
            <translation>關閉</translation>
        </message>
    </context>
    <context>
        <name>ClipListModel</name>
        <message>
            <location filename="../models/clip_model.py" line="73" />
            <source>State: %s</source>
            <translation>狀態：%s</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="76" />
            <source>Input: %d frames (%s)</source>
            <translation>輸入：%d 影格（%s）</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="81" />
            <source>Alpha: %d frames</source>
            <translation>Alpha：%d 影格</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="86" />
            <source>Warnings: %d</source>
            <translation>警告：%d</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="90" />
            <source>Error: %s</source>
            <translation>錯誤：%s</translation>
        </message>
    </context>
    <context>
        <name>ClipState</name>
        <message>
            <location filename="../state_labels.py" line="18" />
            <source>EXTRACTING</source>
            <translation>擷取中</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="19" />
            <source>RAW</source>
            <translation>原始</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="20" />
            <source>MASKED</source>
            <translation>已遮罩</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="21" />
            <source>READY</source>
            <translation>就緒</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="22" />
            <source>COMPLETE</source>
            <translation>完成</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="23" />
            <source>ERROR</source>
            <translation>錯誤</translation>
        </message>
    </context>
    <context>
        <name>DebugConsoleWidget</name>
        <message>
            <location filename="../widgets/debug_console.py" line="86" />
            <source>Console</source>
            <translation>主控台</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="129" />
            <source>CONSOLE</source>
            <translation>主控台</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="172" />
            <source>Level:</source>
            <translation>層級：</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="178" />
            <location filename="../widgets/debug_console.py" line="334" />
            <source>Pause</source>
            <translation>暫停</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="185" />
            <source>Clear</source>
            <translation>清除</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="334" />
            <source>Resume</source>
            <translation>繼續</translation>
        </message>
    </context>
    <context>
        <name>DiagnosticDialog</name>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="47" />
            <source>Diagnostic: %s</source>
            <translation>診斷：%s</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="108" />
            <source>Error: %s</source>
            <translation>錯誤：%s</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="122" />
            <source>Report Issue on GitHub</source>
            <translation>在 GitHub 回報問題</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="129" />
            <source>OK</source>
            <translation>確定</translation>
        </message>
    </context>
    <context>
        <name>FrameScrubber</name>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="52" />
            <source>Go to first frame</source>
            <translation>跳至第一影格</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="60" />
            <source>Previous frame</source>
            <translation>上一影格</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="68" />
            <source>Play / Pause (Space)</source>
            <translation>播放 / 暫停（Space）</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="82" />
            <source>Coverage bar — shows which frames have been processed.
Green lane: painted frames (brush strokes).
White lane: alpha hint coverage.
Yellow lane: inference output coverage.</source>
            <translation>覆蓋進度條——顯示哪些影格已處理。
綠色軌道：已繪製影格（筆觸）。
白色軌道：Alpha 提示覆蓋範圍。
黃色軌道：推理輸出覆蓋範圍。</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="95" />
            <source>Scrub through frames. Scroll wheel or Left/Right to step.</source>
            <translation>拖曳瀏覽影格。滾輪或 Left/Right 鍵逐格移動。</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="118" />
            <source>Next frame</source>
            <translation>下一影格</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="126" />
            <source>Go to last frame</source>
            <translation>跳至最後影格</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="310" />
            <source>Pause (Space)</source>
            <translation>暫停（Space）</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="317" />
            <source>Play (Space)</source>
            <translation>播放（Space）</translation>
        </message>
    </context>
    <context>
        <name>HotkeysDialog</name>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="133" />
            <source>Hotkeys</source>
            <translation>快速鍵</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="145" />
            <source>Filter shortcuts...</source>
            <translation>篩選快速鍵...</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="212" />
            <source>Reset</source>
            <translation>重置</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="220" />
            <source>Reset to default: %s</source>
            <translation>重置為預設值：%s</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="240" />
            <source>Reset All to Defaults</source>
            <translation>全部重置為預設值</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="252" />
            <source>Cancel</source>
            <translation>取消</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="263" />
            <source>OK</source>
            <translation>確定</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="309" />
            <source>Reset All Shortcuts</source>
            <translation>重置所有快速鍵</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="310" />
            <source>Reset all shortcuts to their default values?</source>
            <translation>將所有快速鍵重置為預設值？</translation>
        </message>
    </context>
    <context>
        <name>IOTrayActionsMixin</name>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="63" />
            <source>Run Extraction (%d clips)</source>
            <translation>執行擷取（%d 個片段）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="64" />
            <source>Run Extraction</source>
            <translation>執行擷取</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="72" />
            <source>Rename...</source>
            <translation>重新命名...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="78" />
            <source>Finder</source>
            <translation>Finder</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="78" />
            <source>Explorer</source>
            <translation>檔案總管</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="79" />
            <source>Open in %s</source>
            <translation>在 %s 中開啟</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="89" />
            <source>Clear Mask (%d clips)</source>
            <translation>清除遮罩（%d 個片段）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="89" />
            <location filename="../widgets/io_tray_actions.py" line="241" />
            <source>Clear Mask</source>
            <translation>清除遮罩</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="97" />
            <source>Clear Alpha (%d clips)</source>
            <translation>清除 Alpha（%d 個片段）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="97" />
            <location filename="../widgets/io_tray_actions.py" line="352" />
            <source>Clear Alpha</source>
            <translation>清除 Alpha</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="105" />
            <source>Clear Outputs (%d clips)</source>
            <translation>清除輸出（%d 個片段）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="105" />
            <location filename="../widgets/io_tray_actions.py" line="385" />
            <source>Clear Outputs</source>
            <translation>清除輸出</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="113" />
            <source>Clear All (%d clips)</source>
            <translation>全部清除（%d 個片段）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="113" />
            <location filename="../widgets/io_tray_actions.py" line="306" />
            <source>Clear All</source>
            <translation>全部清除</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="120" />
            <source>Set Output Directory...</source>
            <translation>設定輸出目錄...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="126" />
            <source>Clear Output Directory Override</source>
            <translation>清除輸出目錄覆蓋設定</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="133" />
            <source>Remove (%d clips)...</source>
            <translation>移除（%d 個片段）...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="133" />
            <source>Remove...</source>
            <translation>移除...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="157" />
            <source>Export %s as Video...</source>
            <translation>將 %s 匯出為影片...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="169" />
            <source>Open Containing Folder</source>
            <translation>開啟所在資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="183" />
            <source>Output Directory for '%s'</source>
            <translation>「%s」的輸出目錄</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="214" />
            <source>Rename Clip</source>
            <translation>重新命名片段</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="214" />
            <source>New name:</source>
            <translation>新名稱：</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="242" />
            <source>Delete tracked masks for %d clip(s)?
%s

This will remove all SAM2 mask frames from disk.</source>
            <translation>刪除 %d 個片段的追蹤遮罩？
%s

此操作將從磁碟移除所有 SAM2 遮罩影格。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="307" />
            <source>Remove ALL generated data for %d clip(s)?
%s

This will delete masks, alpha hints, and all output frames.</source>
            <translation>移除 %d 個片段的全部產生資料？
%s

此操作將刪除遮罩、Alpha 提示及所有輸出影格。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="353" />
            <source>Delete AlphaHint for %d clip(s)?
%s

This will remove all generated alpha hint frames from disk.</source>
            <translation>刪除 %d 個片段的 AlphaHint？
%s

此操作將從磁碟移除所有產生的 Alpha 提示影格。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="386" />
            <source>Remove all output files for %d clip(s)?
%s

This will delete FG, Matte, Comp, and Processed frames.</source>
            <translation>移除 %d 個片段的全部輸出檔案？
%s

此操作將刪除 FG、Matte、Comp 及 Processed 影格。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="412" />
            <source>Remove %d clip(s)?</source>
            <translation>移除 %d 個片段？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="416" />
            <source>
... and %d more</source>
            <translation>
... 以及另外 %d 個</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="421" />
            <source>How would you like to remove %d clip(s)?</source>
            <translation>您要如何移除 %d 個片段？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="424" />
            <source>Remove from List</source>
            <translation>從列表移除</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="425" />
            <source>Delete from Disk</source>
            <translation>從磁碟刪除</translation>
        </message>
    </context>
    <context>
        <name>IOTrayPanel</name>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="81" />
            <source>INPUT (0)</source>
            <translation>輸入（0）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="86" />
            <source>RESET I/O</source>
            <translation>重置入出點</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="88" />
            <source>Clear in/out markers on all clips</source>
            <translation>清除所有片段的入出點標記</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="92" />
            <source>+ ADD</source>
            <translation>+ 新增</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="94" />
            <source>Import clips — choose a folder or video file(s)</source>
            <translation>匯入片段——選擇資料夾或影片檔案</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="122" />
            <source>EXPORTS (0)</source>
            <translation>匯出（0）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="169" />
            <source>Import Folder...</source>
            <translation>匯入資料夾...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="170" />
            <source>Import Video(s)...</source>
            <translation>匯入影片...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="171" />
            <source>Import Image Sequence...</source>
            <translation>匯入影像序列...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="180" />
            <source>No Markers</source>
            <translation>無標記</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="181" />
            <source>No clips have in/out markers set.</source>
            <translation>所有片段均未設定入出點標記。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="188" />
            <source>Reset In/Out Markers</source>
            <translation>重置入出點標記</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="189" />
            <source>This will clear in/out markers on %d clip(s).

All clips will revert to full-clip processing.
Continue?</source>
            <translation>此操作將清除 %d 個片段的入出點標記。

所有片段將恢復為完整片段處理。
是否繼續？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="199" />
            <source>Confirm Reset</source>
            <translation>確認重置</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="200" />
            <source>Are you sure? This cannot be undone.

Clearing in/out markers on %d clip(s).</source>
            <translation>確定要執行此操作？此操作無法復原。

正在清除 %d 個片段的入出點標記。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="212" />
            <source>Select Clips Directory</source>
            <translation>選擇片段目錄</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="220" />
            <source>Select Video Files</source>
            <translation>選擇影片檔案</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="228" />
            <source>Select Image Sequence Folder</source>
            <translation>選擇影像序列資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="359" />
            <source>INPUT (%d)</source>
            <translation>輸入（%d）</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="360" />
            <source>EXPORTS (%d)</source>
            <translation>匯出（%d）</translation>
        </message>
    </context>
    <context>
        <name>JobType</name>
        <message>
            <location filename="../state_labels.py" line="37" />
            <source>Inference</source>
            <translation>推理</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="40" />
            <source>Track Preview</source>
            <translation>追蹤預覽</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="41" />
            <source>Track Mask</source>
            <translation>追蹤遮罩</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="44" />
            <source>Preview</source>
            <translation>預覽</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="46" />
            <source>Pipeline</source>
            <translation>流程</translation>
        </message>
    </context>
    <context>
        <name>KeyBindButton</name>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="56" />
            <source>(none)</source>
            <translation>（無）</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="62" />
            <source>Press a key...</source>
            <translation>按下按鍵...</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="94" />
            <source>Shortcut Conflict</source>
            <translation>快速鍵衝突</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="95" />
            <source>"%s" is already assigned to:
%s

Reassign anyway? The conflicting binding will be cleared.</source>
            <translation>「%s」已指定給：
%s

仍要重新指定？衝突的綁定將被清除。</translation>
        </message>
    </context>
    <context>
        <name>MainWindow</name>
        <message>
            <location filename="../main_window.py" line="291" />
            <source>%s — Mac Performance Warning</source>
            <translation>%s — Mac 效能警告</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="293" />
            <source>GPU-intensive features (SAM2, GVM, VideoMaMa, MatAnyone2) are very slow on Mac (Apple Silicon MPS).

This may take hours for longer clips and could freeze your system.

Recommendation: Import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke instead.

Continue anyway? (This warning won't appear again this session.)</source>
            <translation>GPU 密集型功能（SAM2、GVM、VideoMaMa、MatAnyone2）在 Mac（Apple Silicon MPS）上執行非常緩慢。

較長的片段可能需要數小時，且可能導致系統停止回應。

建議：改為從 After Effects、DaVinci Resolve 或 Nuke 匯入現有的 Alpha 遮罩。

仍要繼續？（本工作階段不會再顯示此警告。）</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="311" />
            <source>EZ-CorridorKey</source>
            <translation>EZ-CorridorKey</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="469" />
            <source>Detected GPU used for inference</source>
            <translation>已偵測到用於推理的 GPU</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="472" />
            <source>VRAM</source>
            <translation>VRAM</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="483" />
            <source>GPU video memory usage — updates during inference</source>
            <translation>GPU 視訊記憶體用量——推理期間即時更新</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="489" />
            <source>Current VRAM used / total available</source>
            <translation>目前已用 VRAM / 可用總量</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="681" />
            <source>No GPU</source>
            <translation>無 GPU</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="688" />
            <source>Memory</source>
            <translation>記憶體</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="689" />
            <source>Unified memory usage — CPU and GPU share the same pool</source>
            <translation>統一記憶體用量——CPU 與 GPU 共享同一記憶體池</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="690" />
            <source>Current unified memory used / total available</source>
            <translation>目前已用統一記憶體 / 可用總量</translation>
        </message>
        <message>
            <source>%s complete for %s%s -- Ready to Run Inference</source>
            <translation>%s 已完成（%s%s）—— 可執行推理</translation>
        </message>
        <message>
            <source>'%s' is already queued.</source>
            <translation>「%s」已在佇列中。</translation>
        </message>
        <message>
            <source>A new version (v%s) is available.
Click to save your session and run the updater.</source>
            <translation>有新版本（v%s）可用。
按一下以儲存工作階段並執行更新程式。</translation>
        </message>
        <message>
            <source>AI Green Screen Keyer</source>
            <translation>AI 綠幕去背工具</translation>
        </message>
        <message>
            <source>About</source>
            <translation>關於</translation>
        </message>
        <message>
            <source>About EZ-CorridorKey</source>
            <translation>關於 EZ-CorridorKey</translation>
        </message>
        <message>
            <source>All selected videos are already in the project (%s).</source>
            <translation>所有已選取的影片均已在專案（%s）中。</translation>
        </message>
        <message>
            <source>Alpha</source>
            <translation>Alpha</translation>
        </message>
        <message>
            <source>Alpha hints cover %d of %d frames.

You can process the available range, re-run GVM to
regenerate all alpha frames, or cancel.</source>
            <translation>Alpha 提示覆蓋 %d / %d 影格。

您可以處理可用範圍、重新執行 GVM 以
重新產生所有 Alpha 影格，或取消。</translation>
        </message>
        <message>
            <source>Already Imported</source>
            <translation>已匯入</translation>
        </message>
        <message>
            <source>Automatic updates are not supported on this platform.
Please download the latest release from GitHub.</source>
            <translation>此平台不支援自動更新。
請從 GitHub 下載最新版本。</translation>
        </message>
        <message>
            <source>Batch Export</source>
            <translation>批次匯出</translation>
        </message>
        <message>
            <source>Batch Export Complete</source>
            <translation>批次匯出完成</translation>
        </message>
        <message>
            <source>Batch Pipeline...</source>
            <translation>批次流程...</translation>
        </message>
        <message>
            <source>Blue Screen Model Required</source>
            <translation>需要藍幕模型</translation>
        </message>
        <message>
            <source>Cancel</source>
            <translation>取消</translation>
        </message>
        <message>
            <source>Cancel %s?</source>
            <translation>取消 %s？</translation>
        </message>
        <message>
            <source>Cancel all pending batch jobs and clear the pipeline?</source>
            <translation>取消所有待執行的批次工作並清除流程？</translation>
        </message>
        <message>
            <source>Cancel processing?</source>
            <translation>取消處理？</translation>
        </message>
        <message>
            <source>Cancelled queued work.</source>
            <translation>已取消佇列中的工作。</translation>
        </message>
        <message>
            <source>Cancelled: %s</source>
            <translation>已取消：%s</translation>
        </message>
        <message>
            <source>Clear Batch Pipeline</source>
            <translation>清除批次流程</translation>
        </message>
        <message>
            <source>Clear Holdout Strokes</source>
            <translation>清除遮擋筆觸</translation>
        </message>
        <message>
            <source>Clear Paint Strokes</source>
            <translation>清除繪製筆觸</translation>
        </message>
        <message>
            <source>Clear Project Output Folder</source>
            <translation>清除專案輸出資料夾</translation>
        </message>
        <message>
            <source>Clear all holdout mask strokes for this clip?</source>
            <translation>清除此片段的所有遮擋遮罩筆觸？</translation>
        </message>
        <message>
            <source>Clear the current batch folder and clip list?</source>
            <translation>清除目前的批次資料夾和片段列表？</translation>
        </message>
        <message>
            <source>Clip '%s' already has alpha hint images.

Do you want to replace them with chroma key hints?</source>
            <translation>片段「%s」已有 Alpha 提示影像。

是否要以色鍵提示取代？</translation>
        </message>
        <message>
            <source>Clip '%s' already has alpha hint images.

Do you want to replace them with new ones?</source>
            <translation>片段「%s」已有 Alpha 提示影像。

是否要以新影像取代？</translation>
        </message>
        <message>
            <source>Clip '%s' has %d input frames but you selected %d alpha hints.

Each input frame needs a matching alpha hint.
Only %d frames will be paired.</source>
            <translation>片段「%s」有 %d 個輸入影格，但您選取了 %d 個 Alpha 提示。

每個輸入影格需要對應的 Alpha 提示。
僅 %d 個影格將配對。</translation>
        </message>
        <message>
            <source>Clip '%s' is in %s state.
Only READY or COMPLETE clips can be processed.</source>
            <translation>片段「%s」目前為 %s 狀態。
只有「就緒」或「完成」狀態的片段才能處理。</translation>
        </message>
        <message>
            <source>Clip '%s' must be COMPLETE to export video.</source>
            <translation>片段「%s」必須為「完成」狀態才能匯出影片。</translation>
        </message>
        <message>
            <source>Clip / Output</source>
            <translation>片段 / 輸出</translation>
        </message>
        <message>
            <source>Clip: %s

%s</source>
            <translation>片段：%s

%s</translation>
        </message>
        <message>
            <source>Console</source>
            <translation>主控台</translation>
        </message>
        <message>
            <source>Copy Just These %d</source>
            <translation>僅複製這 %d 個</translation>
        </message>
        <message>
            <source>Could not load the selected language. The interface stays in English.</source>
            <translation>無法載入所選語言，介面將維持英文。</translation>
        </message>
        <message>
            <source>Could not read frame count from the selected alpha video.</source>
            <translation>無法從選取的 Alpha 影片讀取影格數。</translation>
        </message>
        <message>
            <source>Could not relaunch the app automatically.

Please close and reopen EZ-CorridorKey manually.</source>
            <translation>無法自動重新啟動應用程式。

請手動關閉並重新開啟 EZ-CorridorKey。</translation>
        </message>
        <message>
            <source>Could not update automatically:

%s

Please download the latest release manually from GitHub.</source>
            <translation>無法自動更新：

%s

請從 GitHub 手動下載最新版本。</translation>
        </message>
        <message>
            <source>Download Manager</source>
            <translation>下載管理員</translation>
        </message>
        <message>
            <source>Download Manager...</source>
            <translation>下載管理員...</translation>
        </message>
        <message>
            <source>Downloading update...</source>
            <translation>正在下載更新...</translation>
        </message>
        <message>
            <source>Duplicate</source>
            <translation>重複項目</translation>
        </message>
        <message>
            <source>Duplicate Filenames</source>
            <translation>重複檔案名稱</translation>
        </message>
        <message>
            <source>EZ-CorridorKey is already updated to v%s.

Restart the app to load the new version.</source>
            <translation>EZ-CorridorKey 已更新至 v%s。

請重新啟動應用程式以載入新版本。</translation>
        </message>
        <message>
            <source>Edit</source>
            <translation>編輯</translation>
        </message>
        <message>
            <source>Entire Clip</source>
            <translation>完整片段</translation>
        </message>
        <message>
            <source>Exit</source>
            <translation>結束</translation>
        </message>
        <message>
            <source>Export All Videos</source>
            <translation>匯出所有影片</translation>
        </message>
        <message>
            <source>Export Complete</source>
            <translation>匯出完成</translation>
        </message>
        <message>
            <source>Export Failed</source>
            <translation>匯出失敗</translation>
        </message>
        <message>
            <source>Export Video</source>
            <translation>匯出影片</translation>
        </message>
        <message>
            <source>Export Video...</source>
            <translation>匯出影片...</translation>
        </message>
        <message>
            <source>Exporting %s / %s...</source>
            <translation>正在匯出 %s / %s...</translation>
        </message>
        <message>
            <source>Exporting %s...</source>
            <translation>正在匯出 %s...</translation>
        </message>
        <message>
            <source>Exporting videos...</source>
            <translation>正在匯出影片...</translation>
        </message>
        <message>
            <source>FFmpeg Unavailable</source>
            <translation>FFmpeg 無法使用</translation>
        </message>
        <message>
            <source>Failed to export video:
%s</source>
            <translation>匯出影片失敗：
%s</translation>
        </message>
        <message>
            <source>Failed to import alpha hints:
%s</source>
            <translation>匯入 Alpha 提示失敗：
%s</translation>
        </message>
        <message>
            <source>Failed to scan clips directory:
%s</source>
            <translation>掃描片段目錄失敗：
%s</translation>
        </message>
        <message>
            <source>File</source>
            <translation>檔案</translation>
        </message>
        <message>
            <source>Force Stop</source>
            <translation>強制停止</translation>
        </message>
        <message>
            <source>Force Stop Failed</source>
            <translation>強制停止失敗</translation>
        </message>
        <message>
            <source>Force restarting...</source>
            <translation>強制重新啟動中...</translation>
        </message>
        <message>
            <source>Foreground color: %s</source>
            <translation>前景色：%s</translation>
        </message>
        <message>
            <source>Format</source>
            <translation>格式</translation>
        </message>
        <message>
            <source>Found %d/%d alpha frames from a previous run.</source>
            <translation>從上次執行中找到 %d / %d 個 Alpha 影格。</translation>
        </message>
        <message>
            <source>Found files with the same name but different extensions:
%s

This would cause output file conflicts. Please use one format per sequence folder.</source>
            <translation>找到同名但副檔名不同的檔案：
%s

這將導致輸出檔案衝突。每個序列資料夾請只使用一種格式。</translation>
        </message>
        <message>
            <source>Frame Count Mismatch</source>
            <translation>影格數不符</translation>
        </message>
        <message>
            <source>Frames</source>
            <translation>影格</translation>
        </message>
        <message>
            <source>GPU is finishing the current chunk.
VideoMaMa will stop after it completes.</source>
            <translation>GPU 正在完成目前的區塊。
VideoMaMa 將在完成後停止。</translation>
        </message>
        <message>
            <source>Give your project a name:</source>
            <translation>為您的專案命名：</translation>
        </message>
        <message>
            <source>Help</source>
            <translation>說明</translation>
        </message>
        <message>
            <source>Hotkeys...</source>
            <translation>快速鍵...</translation>
        </message>
        <message>
            <source>How would you like to import?</source>
            <translation>您要如何匯入？</translation>
        </message>
        <message>
            <source>Image Folder</source>
            <translation>影像資料夾</translation>
        </message>
        <message>
            <source>Import Alpha</source>
            <translation>匯入 Alpha</translation>
        </message>
        <message>
            <source>Import Alpha Failed</source>
            <translation>匯入 Alpha 失敗</translation>
        </message>
        <message>
            <source>Import Clips</source>
            <translation>匯入片段</translation>
        </message>
        <message>
            <source>Import Folder...</source>
            <translation>匯入資料夾...</translation>
        </message>
        <message>
            <source>Import Full Sequence</source>
            <translation>匯入完整序列</translation>
        </message>
        <message>
            <source>Import Image Frames</source>
            <translation>匯入影像影格</translation>
        </message>
        <message>
            <source>Import Image Sequence...</source>
            <translation>匯入影像序列...</translation>
        </message>
        <message>
            <source>Import Video(s)...</source>
            <translation>匯入影片...</translation>
        </message>
        <message>
            <source>Import alpha from an image folder or a video file?</source>
            <translation>要從影像資料夾或影片檔案匯入 Alpha？</translation>
        </message>
        <message>
            <source>Imported %d/%d %s from video.
Clip is now %s.</source>
            <translation>已從影片匯入 %d / %d %s。
片段目前狀態：%s。</translation>
        </message>
        <message>
            <source>Imported %d/%d %s.
Clip is now %s.</source>
            <translation>已匯入 %d / %d %s。
片段目前狀態：%s。</translation>
        </message>
        <message>
            <source>Incomplete Alpha</source>
            <translation>不完整的 Alpha</translation>
        </message>
        <message>
            <source>Inference complete: %s</source>
            <translation>推理完成：%s</translation>
        </message>
        <message>
            <source>Installing update...</source>
            <translation>正在安裝更新...</translation>
        </message>
        <message>
            <source>Language</source>
            <translation>語言</translation>
        </message>
        <message>
            <source>MatAnyone2 requires a tracked mask on frame 0.

Paint prompts and run Track Mask before using MatAnyone2.</source>
            <translation>MatAnyone2 需要在第 0 影格有追蹤遮罩。

請先繪製提示並執行「追蹤遮罩」，再使用 MatAnyone2。</translation>
        </message>
        <message>
            <source>Missing</source>
            <translation>遺失</translation>
        </message>
        <message>
            <source>Name Your Project</source>
            <translation>為您的專案命名</translation>
        </message>
        <message>
            <source>New Project</source>
            <translation>新增專案</translation>
        </message>
        <message>
            <source>No %s found in the latest release.
Release: %s

Please download manually from GitHub.</source>
            <translation>最新版本中未找到 %s。
版本：%s

請從 GitHub 手動下載。</translation>
        </message>
        <message>
            <source>No COMPLETE clips to export.</source>
            <translation>沒有「完成」狀態的片段可供匯出。</translation>
        </message>
        <message>
            <source>No Clip</source>
            <translation>無片段</translation>
        </message>
        <message>
            <source>No Clips</source>
            <translation>無片段</translation>
        </message>
        <message>
            <source>No Folder</source>
            <translation>無資料夾</translation>
        </message>
        <message>
            <source>No Frames</source>
            <translation>無影格</translation>
        </message>
        <message>
            <source>No Images</source>
            <translation>無影像</translation>
        </message>
        <message>
            <source>No Media</source>
            <translation>無媒體</translation>
        </message>
        <message>
            <source>No Output</source>
            <translation>無輸出</translation>
        </message>
        <message>
            <source>No Paint Strokes</source>
            <translation>無繪製筆觸</translation>
        </message>
        <message>
            <source>No Project</source>
            <translation>無專案</translation>
        </message>
        <message>
            <source>No READY clips to process.</source>
            <translation>沒有「就緒」狀態的片段可供處理。</translation>
        </message>
        <message>
            <source>No clip selected</source>
            <translation>未選取片段</translation>
        </message>
        <message>
            <source>No image files found in that folder.

Supported formats: PNG, JPG, EXR, TIF, TIFF, BMP, DPX</source>
            <translation>在該資料夾中未找到影像檔案。

支援格式：PNG、JPG、EXR、TIF、TIFF、BMP、DPX</translation>
        </message>
        <message>
            <source>No image files found in the selected folder.
Expected grayscale images (white=foreground, black=background).</source>
            <translation>在選取的資料夾中未找到影像檔案。
預期為灰階影像（白色＝前景，黑色＝背景）。</translation>
        </message>
        <message>
            <source>No image frames found in output directory.</source>
            <translation>輸出目錄中未找到影像影格。</translation>
        </message>
        <message>
            <source>No output frames found to export.</source>
            <translation>未找到可供匯出的輸出影格。</translation>
        </message>
        <message>
            <source>No output frames found.</source>
            <translation>未找到輸出影格。</translation>
        </message>
        <message>
            <source>No selected clips are in a processable state.</source>
            <translation>所有已選取的片段均未處於可處理狀態。</translation>
        </message>
        <message>
            <source>No video files or image sequences found in that folder.</source>
            <translation>在該資料夾中未找到影片檔案或影像序列。</translation>
        </message>
        <message>
            <source>Not Complete</source>
            <translation>未完成</translation>
        </message>
        <message>
            <source>Not Ready</source>
            <translation>未就緒</translation>
        </message>
        <message>
            <source>Nothing to Export</source>
            <translation>無可匯出的內容</translation>
        </message>
        <message>
            <source>Nothing to Process</source>
            <translation>無可處理的內容</translation>
        </message>
        <message>
            <source>Open Project...</source>
            <translation>開啟專案...</translation>
        </message>
        <message>
            <source>Open a clips folder first.</source>
            <translation>請先開啟片段資料夾。</translation>
        </message>
        <message>
            <source>Open a project first.</source>
            <translation>請先開啟專案。</translation>
        </message>
        <message>
            <source>Paint green (1) and red (2) strokes on frames first.</source>
            <translation>請先在影格上繪製綠色（1）和紅色（2）筆觸。</translation>
        </message>
        <message>
            <source>Paint prompts and run Track Mask before using VideoMaMa.</source>
            <translation>使用 VideoMaMa 前，請先繪製提示並執行「追蹤遮罩」。</translation>
        </message>
        <message>
            <source>Partial Alpha Found</source>
            <translation>找到部分 Alpha</translation>
        </message>
        <message>
            <source>Preferences...</source>
            <translation>偏好設定...</translation>
        </message>
        <message>
            <source>Process Available</source>
            <translation>可供處理</translation>
        </message>
        <message>
            <source>Processing Error</source>
            <translation>處理錯誤</translation>
        </message>
        <message>
            <source>Project name for this batch:</source>
            <translation>此批次的專案名稱：</translation>
        </message>
        <message>
            <source>Re-run GVM</source>
            <translation>重新執行 GVM</translation>
        </message>
        <message>
            <source>Regenerate</source>
            <translation>重新產生</translation>
        </message>
        <message>
            <source>Replace Alpha Hints?</source>
            <translation>取代 Alpha 提示？</translation>
        </message>
        <message>
            <source>Replace Existing Alpha?</source>
            <translation>取代現有 Alpha？</translation>
        </message>
        <message>
            <source>Report Issue...</source>
            <translation>回報問題...</translation>
        </message>
        <message>
            <source>Reset Layout</source>
            <translation>重置版面配置</translation>
        </message>
        <message>
            <source>Reset Zoom</source>
            <translation>重置縮放</translation>
        </message>
        <message>
            <source>Resume</source>
            <translation>繼續</translation>
        </message>
        <message>
            <source>Resume will skip completed frames.
Regenerate will redo all frames from scratch.</source>
            <translation>「繼續」將略過已完成的影格。
「重新產生」將從頭重做所有影格。</translation>
        </message>
        <message>
            <source>Return to Home</source>
            <translation>返回首頁</translation>
        </message>
        <message>
            <source>SAM2 preview on frame %d covers %.1f%% of the frame.

If this looks right, continue with full Track Mask.
If not, keep painting corrections on this frame and run Track Mask again.</source>
            <translation>第 %d 影格的 SAM2 預覽覆蓋了 %.1f%% 的畫面。

如果效果正確，請繼續執行完整「追蹤遮罩」。
若不正確，請繼續在此影格修正筆觸並再次執行「追蹤遮罩」。</translation>
        </message>
        <message>
            <source>Save Session</source>
            <translation>儲存工作階段</translation>
        </message>
        <message>
            <source>Scan Error</source>
            <translation>掃描錯誤</translation>
        </message>
        <message>
            <source>Select Alpha Hint Folder</source>
            <translation>選擇 Alpha 提示資料夾</translation>
        </message>
        <message>
            <source>Select Alpha Hint Video</source>
            <translation>選擇 Alpha 提示影片</translation>
        </message>
        <message>
            <source>Select Clips Directory</source>
            <translation>選擇片段目錄</translation>
        </message>
        <message>
            <source>Select Image Sequence Folder</source>
            <translation>選擇影像序列資料夾</translation>
        </message>
        <message>
            <source>Select Video Files</source>
            <translation>選擇影片檔案</translation>
        </message>
        <message>
            <source>Select a clip first.</source>
            <translation>請先選取片段。</translation>
        </message>
        <message>
            <source>Select which outputs to export as video:</source>
            <translation>選擇要匯出為影片的輸出項目：</translation>
        </message>
        <message>
            <source>Set Project Output Folder</source>
            <translation>設定專案輸出資料夾</translation>
        </message>
        <message>
            <source>Set Project Output Folder...</source>
            <translation>設定專案輸出資料夾...</translation>
        </message>
        <message>
            <source>Special Thanks</source>
            <translation>特別感謝</translation>
        </message>
        <message>
            <source>Stop requested — waiting for current GPU step. Press FORCE STOP to relaunch if it stays stuck.</source>
            <translation>已請求停止——等待目前 GPU 步驟完成。若持續卡住，請按「強制停止」重新啟動。</translation>
        </message>
        <message>
            <source>The current GPU step has not returned to Python.

Force Stop will auto-save the session and relaunch the app to break the stuck job immediately.

Continue?</source>
            <translation>目前 GPU 步驟尚未回傳至 Python。

「強制停止」將自動儲存工作階段並重新啟動應用程式，以立即中斷卡住的工作。

是否繼續？</translation>
        </message>
        <message>
            <source>The interface language will change after the current jobs finish and the app restarts.</source>
            <translation>介面語言將在目前工作完成並重新啟動應用程式後變更。</translation>
        </message>
        <message>
            <source>The update could not be verified and was NOT installed.

%s

This may indicate a security issue. Please download the latest release manually from GitHub or Gumroad.</source>
            <translation>更新無法驗證且未安裝。

%s

這可能表示存在安全性問題。請從 GitHub 或 Gumroad 手動下載最新版本。</translation>
        </message>
        <message>
            <source>This Frame</source>
            <translation>此影格</translation>
        </message>
        <message>
            <source>This clip already has an AlphaHint (from GVM or a previous run).

Tracking a new mask sequence will replace that alpha hint.

Remove existing AlphaHint and proceed?</source>
            <translation>此片段已有 AlphaHint（來自 GVM 或上次執行）。

追蹤新的遮罩序列將取代該 Alpha 提示。

移除現有的 AlphaHint 並繼續？</translation>
        </message>
        <message>
            <source>This clip uses a blue screen background.

The blue screen keying model (401 MB) is not installed. Without it, the green model will be used as a fallback.

Download the blue screen model now?</source>
            <translation>此片段使用藍幕背景。

藍幕去背模型（401 MB）尚未安裝。若未安裝，將以綠幕模型作為備用。

立即下載藍幕模型？</translation>
        </message>
        <message>
            <source>This sequence is already in the project as "%s".</source>
            <translation>此序列已以「%s」的名稱存在於專案中。</translation>
        </message>
        <message>
            <source>This will download the latest version, replace the current app,
and relaunch automatically.

Your session will be saved. Continue?</source>
            <translation>此操作將下載最新版本、取代目前的應用程式，
並自動重新啟動。

您的工作階段將被儲存。是否繼續？</translation>
        </message>
        <message>
            <source>This will save your session, close the app, and run the updater.
The app will relaunch automatically after updating.

Continue?</source>
            <translation>此操作將儲存您的工作階段、關閉應用程式並執行更新程式。
更新完成後應用程式將自動重新啟動。

是否繼續？</translation>
        </message>
        <message>
            <source>Toggle Queue Panel</source>
            <translation>切換佇列面板</translation>
        </message>
        <message>
            <source>Track Mask First</source>
            <translation>請先追蹤遮罩</translation>
        </message>
        <message>
            <source>Track Mask Preview</source>
            <translation>追蹤遮罩預覽</translation>
        </message>
        <message>
            <source>Track Mask complete for %s</source>
            <translation>「%s」的追蹤遮罩已完成</translation>
        </message>
        <message>
            <source>Track Paint Masks</source>
            <translation>追蹤繪製遮罩</translation>
        </message>
        <message>
            <source>Track preview ready. Refine paint strokes and run Track Mask again.</source>
            <translation>追蹤預覽已就緒。請精修筆觸並再次執行「追蹤遮罩」。</translation>
        </message>
        <message>
            <source>Unreadable Video</source>
            <translation>無法讀取的影片</translation>
        </message>
        <message>
            <source>Update</source>
            <translation>更新</translation>
        </message>
        <message>
            <source>Update Available</source>
            <translation>有更新可用</translation>
        </message>
        <message>
            <source>Update Available (v%s)</source>
            <translation>有更新可用（v%s）</translation>
        </message>
        <message>
            <source>Update EZ-CorridorKey</source>
            <translation>更新 EZ-CorridorKey</translation>
        </message>
        <message>
            <source>Update Failed</source>
            <translation>更新失敗</translation>
        </message>
        <message>
            <source>Update Verification Failed</source>
            <translation>更新驗證失敗</translation>
        </message>
        <message>
            <source>Updating EZ-CorridorKey</source>
            <translation>正在更新 EZ-CorridorKey</translation>
        </message>
        <message>
            <source>Verifying update signature...</source>
            <translation>正在驗證更新簽章...</translation>
        </message>
        <message>
            <source>Video File</source>
            <translation>影片檔案</translation>
        </message>
        <message>
            <source>Video exported:
%s</source>
            <translation>影片已匯出：
%s</translation>
        </message>
        <message>
            <source>VideoMaMa masks</source>
            <translation>VideoMaMa 遮罩</translation>
        </message>
        <message>
            <source>View</source>
            <translation>檢視</translation>
        </message>
        <message>
            <source>What would you like to clear?</source>
            <translation>您要清除哪些內容？</translation>
        </message>
        <message>
            <source>Workspace no longer exists:
%s</source>
            <translation>工作區已不存在：
%s</translation>
        </message>
        <message>
            <source>You dropped %d image file(s).
The source folder contains %d image(s) total.</source>
            <translation>您放入了 %d 個影像檔案。
來源資料夾共包含 %d 個影像。</translation>
        </message>
        <message>
            <source>alpha hints</source>
            <translation>Alpha 提示</translation>
        </message>
    </context>
    <context>
        <name>ParameterPanel</name>
        <message>
            <location filename="../widgets/parameter_panel.py" line="132" />
            <source>ALPHA GENERATION</source>
            <translation>Alpha 產生</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="137" />
            <source>Manual</source>
            <translation>手動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="142" />
            <source>CHROMA KEY</source>
            <translation>色鍵</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="147" />
            <source>Generate alpha hints using a traditional chroma keyer.
Best for clean green/blue screen shots.
No GPU or AI model required — instant processing.

Click to expand parameters, then click GENERATE.
Hotkey: `</source>
            <translation>使用傳統色鍵器產生 Alpha 提示。
最適合乾淨的綠幕 / 藍幕拍攝素材。
無需 GPU 或 AI 模型——即時處理。

按一下展開參數，然後按「產生」。
快速鍵：`</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="166" />
            <source>💧 Pick Screen Color</source>
            <translation>💧 取樣螢幕色彩</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="169" />
            <source>Click on the viewer to sample the screen color.
Works on either the input or output viewport.
Hotkey: E</source>
            <translation>在檢視器上按一下以取樣螢幕色彩。
適用於輸入或輸出任一視窗。
快速鍵：E</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="181" />
            <source>Sampled screen color</source>
            <translation>已取樣的螢幕色彩</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="186" />
            <source>Key Strength: 1.0</source>
            <translation>鍵控強度：1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="191" />
            <source>How aggressively to key the screen color. Higher = more separation.</source>
            <translation>對螢幕色彩進行鍵控的力度。數值越高，分離效果越強。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="193" />
            <source>Key Strength: %s</source>
            <translation>鍵控強度：%s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="199" />
            <source>Clip Black: 0.0</source>
            <translation>截黑：0.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="204" />
            <source>Push near-transparent values to fully transparent.
Cleans up noise in background areas.</source>
            <translation>將接近透明的數值推至完全透明。
清除背景區域的雜訊。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="206" />
            <source>Clip Black: %s</source>
            <translation>截黑：%s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="212" />
            <source>Clip White: 1.0</source>
            <translation>截白：1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="217" />
            <source>Push near-opaque values to fully opaque.
Solidifies the foreground core.</source>
            <translation>將接近不透明的數值推至完全不透明。
強化前景核心。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="219" />
            <source>Clip White: %s</source>
            <translation>截白：%s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="231" />
            <source>Shrink/Grow</source>
            <translation>收縮 / 擴張</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="236" />
            <source>Erode (negative) or dilate (positive) the matte edge.
0 = no change.</source>
            <translation>侵蝕（負值）或擴張（正值）遮罩邊緣。
0 ＝ 不變更。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="240" />
            <source>Edge Blur</source>
            <translation>邊緣模糊</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="245" />
            <source>Gaussian blur radius for softening matte edges.
0 = no blur.</source>
            <translation>用於柔化遮罩邊緣的高斯模糊半徑。
0 ＝ 不模糊。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="252" />
            <source>GENERATE</source>
            <translation>產生</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="253" />
            <source>Generate alpha hint frames for the entire clip using these chroma key settings.</source>
            <translation>使用這些色鍵設定為整個片段產生 Alpha 提示影格。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="271" />
            <source>Automatic</source>
            <translation>自動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="278" />
            <source>APPLE VISION</source>
            <translation>APPLE VISION</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="282" />
            <source>Auto-generate alpha hint using Apple Vision (Neural Engine).
Detects foreground subjects automatically.
macOS 14+ only. Runs on Apple Neural Engine (fast, no GPU needed).</source>
            <translation>使用 Apple Vision（神經引擎）自動產生 Alpha 提示。
自動偵測前景主體。
僅支援 macOS 14 以上版本。在 Apple 神經引擎上執行（快速，無需 GPU）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="300" />
            <source>GVM AUTO</source>
            <translation>GVM 自動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="304" />
            <source>Auto-generate alpha hint for the entire clip.
Uses GVM to predict foreground/background separation.
Available when clip is in RAW state (frames extracted).</source>
            <translation>為整個片段自動產生 Alpha 提示。
使用 GVM 預測前景 / 背景分割。
僅在片段處於「原始」狀態（影格已擷取）時可用。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="315" />
            <source>BIREFNET</source>
            <translation>BIREFNET</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="319" />
            <source>Auto-generate alpha hint using BiRefNet.
Fully automatic — no painting or annotation needed.
Downloads the selected model variant on first use.

Matting: Best for hair/transparency detail (recommended).
Portrait: Optimized for human close-ups.
General: Balanced foreground/background separation.
HR variants: For 2K/4K footage (uses more VRAM).</source>
            <translation>使用 BiRefNet 自動產生 Alpha 提示。
全自動——無需繪製或標註。
首次使用時下載所選模型變體。

去背（Matting）：最適合髮絲 / 透明細節（推薦）。
人像（Portrait）：針對人物特寫最佳化。
通用（General）：前景 / 背景分割平衡。
HR 變體：適用於 2K/4K 素材（需較多 VRAM）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="334" />
            <source>BiRefNet model variant — changes take effect on next run.</source>
            <translation>BiRefNet 模型變體——變更將在下次執行時生效。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="355" />
            <source>Requires brushstrokes</source>
            <translation>需要筆觸</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="360" />
            <source>Paint subject with 1, background with 2</source>
            <translation>按 1 繪製主體，按 2 繪製背景</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="366" />
            <source>TRACK MASK</source>
            <translation>追蹤遮罩</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="370" />
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
            <translation>使用 SAM2 將繪製的提示轉換為密集遮罩追蹤。
在執行 MatAnyone2 或 VideoMaMa 前為必要步驟。

使用方式：
1. 按 1 選取綠色筆刷（前景——要保留的主體）
2. 按 2 選取紅色筆刷（背景——要移除的區域）
3. 在左側檢視器上的素材繪製筆觸
4. 按一下「追蹤遮罩」以在已繪製影格上預覽 SAM2
5. 若預覽效果正確，確認以傳播至所有影格

技巧：
Shift + 左鍵上下拖曳：調整筆刷大小
Alt + 左鍵拖曳：在兩點間畫直線
Ctrl+Z：復原上一筆觸</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="391" />
            <source>MATANYONE2</source>
            <translation>MATANYONE2</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="395" />
            <source>Generate alpha hints using MatAnyone2 video matting.
Requires paint strokes on the FIRST FRAME (frame 1).

1. Navigate to frame 1 (the very first frame)
2. Paint foreground (hotkey 1) and background (hotkey 2)
3. Click Track Mask to generate dense masks with SAM2
4. Click MATANYONE2 to generate temporally coherent AlphaHint</source>
            <translation>使用 MatAnyone2 影片去背產生 Alpha 提示。
需要在第一影格（影格 1）繪製筆觸。

1. 導覽至影格 1（第一影格）
2. 繪製前景（快速鍵 1）和背景（快速鍵 2）
3. 按一下「追蹤遮罩」以使用 SAM2 產生密集遮罩
4. 按一下「MATANYONE2」以產生時間一致的 AlphaHint</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="406" />
            <source>VIDEOMAMA</source>
            <translation>VIDEOMAMA</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="410" />
            <source>Generate alpha hints from a dense VideoMaMa mask track.

1. Paint sparse foreground/background prompts
2. Click Track Mask to generate dense masks with SAM2
3. Click VIDEOMAMA to generate AlphaHint</source>
            <translation>從密集的 VideoMaMa 遮罩追蹤產生 Alpha 提示。

1. 繪製稀疏的前景 / 背景提示
2. 按一下「追蹤遮罩」以使用 SAM2 產生密集遮罩
3. 按一下「VIDEOMAMA」以產生 AlphaHint</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="433" />
            <source>Import your own mask for VideoMaMa.

Bypasses the Track Mask step. Select a folder or
video of grayscale masks and they will be used as
VideoMaMa's guidance input directly.</source>
            <translation>為 VideoMaMa 匯入您自己的遮罩。

略過「追蹤遮罩」步驟。選擇灰階遮罩的資料夾或
影片，這些遮罩將直接作為 VideoMaMa 的引導輸入。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="462" />
            <source>IMPORT ALPHA</source>
            <translation>匯入 ALPHA</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="466" />
            <source>Import alpha hints from an image folder or video file.
Supports: PNG/JPG/TIF/EXR sequences, or MOV/MP4/ProRes video.
White = foreground, black = background.
Files are copied into the clip's AlphaHint/ folder
and the clip advances to READY state for inference.</source>
            <translation>從影像資料夾或影片檔案匯入 Alpha 提示。
支援：PNG/JPG/TIF/EXR 序列，或 MOV/MP4/ProRes 影片。
白色 ＝ 前景，黑色 ＝ 背景。
檔案將複製至片段的 AlphaHint/ 資料夾，
片段進入「就緒」狀態以進行推理。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="479" />
            <source>INFERENCE</source>
            <translation>推理</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="492" />
            <source>BG Color</source>
            <translation>背景色</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="495" />
            <source>Background screen color for this clip.

Auto: detected from the middle frame of the clip.
Green: force green screen processing.
Blue: force blue screen processing.

Controls which checkpoint, despill math, and spill
detection are used. Also changes the UI accent color.</source>
            <translation>此片段的背景螢幕色彩。

自動：從片段中間影格偵測。
綠色：強制綠幕處理。
藍色：強制藍幕處理。

控制所使用的檢查點、去溢色演算法及溢色偵測，並變更介面強調色。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Auto</source>
            <translation>自動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Green</source>
            <translation>綠色</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Blue</source>
            <translation>藍色</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="511" />
            <source>Color Space</source>
            <translation>色彩空間</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="515" />
            <source>sRGB</source>
            <translation>sRGB</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="515" />
            <source>Linear</source>
            <translation>線性</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="521" />
            <source>Despeckle</source>
            <translation>去雜點</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="525" />
            <source>Removes small floating noise and speckles from the
alpha by discarding isolated regions smaller than the
size threshold.</source>
            <translation>從 Alpha 中移除細小浮動雜訊和斑點，方式是丟棄小於
尺寸閾值的孤立區域。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="538" />
            <source>Minimum area (in pixels) for a region to survive.
Isolated alpha blobs smaller than this are removed.
Lower = keep more detail, higher = cleaner matte.</source>
            <translation>區域得以保留的最小面積（像素）。
小於此值的孤立 Alpha 色塊將被移除。
數值較低 ＝ 保留更多細節，數值較高 ＝ 遮罩更乾淨。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="547" />
            <source>Garbage Matte</source>
            <translation>垃圾遮罩</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="551" />
            <source>Expands the alpha hint by N pixels, then zeros out
anything in the predicted matte that falls outside
that expanded region. Removes edge-of-frame artifacts
and background gunk that inference leaves behind.</source>
            <translation>將 Alpha 提示擴展 N 像素，然後將預測遮罩中
落在擴展區域外的所有部分歸零。
移除推理留下的影格邊緣瑕疵和背景殘留。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="565" />
            <source>Pixel expansion around the alpha hint.
Higher = more breathing room around subject edges.
Lower = tighter crop to the hint boundary.</source>
            <translation>Alpha 提示周圍的像素擴展量。
數值較高 ＝ 主體邊緣周圍有更多緩衝空間。
數值較低 ＝ 更緊密裁剪至提示邊界。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="577" />
            <source>Despill: 0.5</source>
            <translation>去溢色：0.5</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="584" />
            <source>Screen spill removal strength (0.0-1.0).
Removes background color bleed from hair, skin, and edges.
1.0 = full despill, 0.0 = no despill (keep original colors).</source>
            <translation>螢幕溢色移除強度（0.0-1.0）。
移除髮絲、皮膚和邊緣的背景顏色滲漏。
1.0 ＝ 完全去溢色，0.0 ＝ 不去溢色（保留原始色彩）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="593" />
            <source>Refiner: 1.0</source>
            <translation>精修：1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="600" />
            <source>Edge refinement strength (0.0-3.0).
Scales the CNN refiner's edge corrections.
1.0 = default, 0.0 = backbone only (no refinement),
higher = sharper edges but may introduce artifacts.</source>
            <translation>邊緣精修強度（0.0-3.0）。
縮放 CNN 精修器的邊緣修正。
1.0 ＝ 預設，0.0 ＝ 僅骨幹（不精修），
數值較高 ＝ 邊緣更銳利，但可能產生瑕疵。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="610" />
            <source>Live Preview</source>
            <translation>即時預覽</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="618" />
            <source>OUTPUT</source>
            <translation>輸出</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="624" />
            <source>FG</source>
            <translation>FG</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="628" />
            <source>Foreground — despilled subject on black background.
Screen spill removed from hair and edges.
Straight alpha (not premultiplied).</source>
            <translation>前景——黑色背景上已去溢色的主體。
已從髮絲和邊緣移除螢幕溢色。
直線 Alpha（非預乘）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="637" />
            <location filename="../widgets/parameter_panel.py" line="656" />
            <source>EXR = 32-bit float (post-production).
PNG = 8-bit (general use).</source>
            <translation>EXR ＝ 32 位元浮點（後期製作）。
PNG ＝ 8 位元（一般用途）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="643" />
            <source>Matte</source>
            <translation>Matte</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="647" />
            <source>Alpha matte — grayscale transparency map.
White = fully opaque, black = fully transparent.
Use in compositing software for manual keying control.</source>
            <translation>Alpha 遮罩——灰階透明度圖。
白色 ＝ 完全不透明，黑色 ＝ 完全透明。
在合成軟體中用於手動鍵控控制。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="662" />
            <source>Comp</source>
            <translation>Comp</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="666" />
            <source>Composite — final keyed result over checkerboard.
Best representation of the key quality.
Colors match the original input faithfully.</source>
            <translation>合成——棋盤格上的最終鍵控結果。
最佳的鍵控品質呈現。
色彩忠實對應原始輸入。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="675" />
            <source>PNG = 8-bit with transparency.
EXR = 32-bit float (post-production).</source>
            <translation>PNG ＝ 8 位元含透明度。
EXR ＝ 32 位元浮點（後期製作）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="681" />
            <source>Processed</source>
            <translation>Processed</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="685" />
            <source>Processed — production-ready RGBA (straight, linear).
Designed for import into Resolve, Premiere, and compositing tools.
Includes despill + garbage matte cleanup applied.</source>
            <translation>Processed——可供製作使用的 RGBA（直線，線性）。
設計用於匯入 Resolve、Premiere 及合成工具。
已套用去溢色 + 垃圾遮罩清除。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="694" />
            <source>EXR = 32-bit float (recommended for Processed).
PNG = 8-bit (lossy for straight linear RGBA).</source>
            <translation>EXR ＝ 32 位元浮點（Processed 建議使用）。
PNG ＝ 8 位元（直線線性 RGBA 會有損失）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="701" />
            <source>PERFORMANCE</source>
            <translation>效能</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="706" />
            <source>Parallel frames</source>
            <translation>平行影格</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="712" />
            <source>Process multiple frames simultaneously using parallel engines.

Each extra engine loads a full copy of the model.
CUDA: ~6-8 GB VRAM per engine.

Default: 1 (safest). Try 2 first, then increase if stable.

EXPERIMENTAL: Values above 8 are for high-memory CUDA systems
(e.g. RTX 6000).
If you run out of memory, the app will automatically scale
back to however many engines fit.

CUDA only right now. Not currently supported on Apple Silicon.</source>
            <translation>使用平行引擎同時處理多個影格。

每個額外引擎將載入完整的模型副本。
CUDA：每個引擎約 6-8 GB VRAM。

預設：1（最安全）。先嘗試 2，穩定後再增加。

實驗性功能：超過 8 的數值適用於高記憶體 CUDA 系統
（例如 RTX 6000）。
若記憶體不足，應用程式將自動縮減至可容納的引擎數量。

目前僅支援 CUDA，不支援 Apple Silicon。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="797" />
            <source>Despill: %s</source>
            <translation>去溢色：%s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="802" />
            <source>Refiner: %s</source>
            <translation>精修：%s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="939" />
            <source>Painted: %d / %d frames</source>
            <translation>已繪製：%d / %d 影格</translation>
        </message>
    </context>
    <context>
        <name>PreferencesDialog</name>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="154" />
            <source>Preferences</source>
            <translation>偏好設定</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="187" />
            <source>User Interface</source>
            <translation>使用者介面</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="192" />
            <source>Language</source>
            <translation>語言</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="206" />
            <source>Select display language. Applies immediately.</source>
            <translation>選擇顯示語言，立即套用。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="210" />
            <source>Show tooltips on controls</source>
            <translation>在控制項上顯示工具提示</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="216" />
            <source>UI sounds</source>
            <translation>介面音效</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="222" />
            <source>Show update notifications</source>
            <translation>顯示更新通知</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="225" />
            <source>When enabled, an Update Available button appears when a newer
release exists. Turn off to never check for or show updates.</source>
            <translation>啟用時，有新版本可用時將顯示「有更新可用」按鈕。
關閉則永不檢查或顯示更新。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="237" />
            <source>Project</source>
            <translation>專案</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="240" />
            <source>Copy source videos into project folder</source>
            <translation>將來源影片複製至專案資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="243" />
            <source>When enabled, imported videos are copied into the project folder.
When disabled, the project references the original file in place.

Note: Deleting a project never touches the original source file.</source>
            <translation>啟用時，匯入的影片將複製至專案資料夾。
停用時，專案將參照原始檔案的原始位置。

注意：刪除專案永遠不會影響原始來源檔案。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="253" />
            <source>Copy imported image sequences into project folder</source>
            <translation>將匯入的影像序列複製至專案資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="256" />
            <source>When enabled, imported image sequence files are copied into the project.
When disabled (default), the project references the original files in place.

Referencing saves disk space for large EXR/TIF sequences.
Original files are never modified regardless of this setting.</source>
            <translation>啟用時，匯入的影像序列檔案將複製至專案。
停用時（預設），專案將參照原始位置的檔案。

參照可節省大型 EXR/TIF 序列的磁碟空間。
無論此設定為何，原始檔案都不會被修改。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="270" />
            <source>Output</source>
            <translation>輸出</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="273" />
            <source>EXR compression</source>
            <translation>EXR 壓縮</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="284" />
            <source>Compression used when writing EXR output files.

DWAB: Lossy wavelet, smallest files. Default.
PIZ: Lossless wavelet, preferred by compositors.
ZIP: Lossless deflate, good for clean renders.
None: No compression, fastest write, largest files.</source>
            <translation>寫入 EXR 輸出檔案時使用的壓縮方式。

DWAB：失真小波壓縮，檔案最小。預設。
PIZ：無損小波壓縮，合成師偏好。
ZIP：無損 deflate，適合乾淨的算圖。
None：無壓縮，寫入最快，檔案最大。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="294" />
            <source>Default output directory</source>
            <translation>預設輸出目錄</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="300" />
            <source>Default (inside project)</source>
            <translation>預設（在專案內部）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="306" />
            <source>Global default directory for inference output.

When set, outputs go to:
  &lt;this folder&gt;/&lt;ProjectName&gt;/&lt;ClipName&gt;/FG, Matte, etc.

Leave empty to use the default (Output/ inside each clip).
Per-clip overrides (right-click → Set Output Directory) take priority.</source>
            <translation>推理輸出的全域預設目錄。

設定後，輸出將存放於：
  &lt;此資料夾&gt;/&lt;專案名稱&gt;/&lt;片段名稱&gt;/FG、Matte 等。

留空以使用預設值（每個片段內的 Output/ 資料夾）。
逐片段覆蓋設定（右鍵 → 設定輸出目錄）具有優先權。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="315" />
            <location filename="../widgets/preferences_dialog.py" line="499" />
            <source>Browse...</source>
            <translation>瀏覽...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="320" />
            <source>Clear</source>
            <translation>清除</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="330" />
            <source>Inference</source>
            <translation>推理</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="333" />
            <source>Model resolution</source>
            <translation>模型解析度</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="337" />
            <source>2048 — Full Quality</source>
            <translation>2048 — 完整品質</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="338" />
            <source>1024 — Faster, Less Detail</source>
            <translation>1024 — 較快，細節較少</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="344" />
            <source>Resolution the model processes internally before upscaling to your frame size.
Applies to all backends (CUDA, MPS, MLX, CPU).

2048: Full quality — captures fine hair strands and edge detail.
Matches the original CorridorKey quality. Recommended for CUDA with 8GB+ VRAM.
WARNING: Very slow on Apple Silicon (needs 20GB+ memory).

1024: Faster inference with lower memory usage.
Fine hair detail may be lost. Recommended for Apple Silicon / low-VRAM GPUs.

Changing this requires an engine reload (happens automatically).</source>
            <translation>模型在放大至您的影格尺寸前，內部處理所用的解析度。
適用於所有後端（CUDA、MPS、MLX、CPU）。

2048：完整品質——捕捉細緻的髮絲和邊緣細節。
符合原始 CorridorKey 品質。建議 CUDA 搭配 8GB 以上 VRAM 使用。
警告：在 Apple Silicon 上非常緩慢（需要 20GB 以上記憶體）。

1024：更快的推理，記憶體用量較低。
細緻的髮絲細節可能遺失。建議 Apple Silicon / 低 VRAM GPU 使用。

變更此設定需要重新載入引擎（自動執行）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="359" />
            <source>Processing backend</source>
            <translation>處理後端</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="363" />
            <source>Auto — MLX if available, otherwise MPS</source>
            <translation>自動——若有 MLX 則使用，否則使用 MPS</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="364" />
            <source>MLX — Apple Metal acceleration (recommended)</source>
            <translation>MLX — Apple Metal 加速（推薦）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="365" />
            <source>MPS — PyTorch Metal Performance Shaders</source>
            <translation>MPS — PyTorch Metal Performance Shaders</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="371" />
            <source>Choose the inference backend for Apple Silicon.

MLX: Native Apple Metal — fastest on M1/M2/M3/M4.
MPS: PyTorch Metal Performance Shaders — compatible fallback.
Auto: Uses MLX if installed, otherwise falls back to MPS.

Changing this requires an engine reload (happens automatically).</source>
            <translation>選擇 Apple Silicon 的推理後端。

MLX：原生 Apple Metal——在 M1/M2/M3/M4 上最快。
MPS：PyTorch Metal Performance Shaders——相容性備用方案。
自動：若已安裝 MLX 則使用，否則退回 MPS。

變更此設定需要重新載入引擎（自動執行）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="383" />
            <source>Playback</source>
            <translation>播放</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="386" />
            <source>Loop playback within in/out range</source>
            <translation>在入出點範圍內循環播放</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="389" />
            <source>When enabled, playback loops back to the in-point
after reaching the out-point (or start/end if no range).</source>
            <translation>啟用時，播放到出點後將循環回入點
（若未設定範圍，則在影片頭尾循環）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="401" />
            <source>Tracking</source>
            <translation>追蹤</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="404" />
            <source>SAM2 model</source>
            <translation>SAM2 模型</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="410" />
            <source>%s  (%s)</source>
            <translation>%s  （%s）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="415" />
            <source>Fast: lower VRAM, lower quality.
Base+: best default tradeoff for this app.
Highest Quality: slowest, heaviest tracker.</source>
            <translation>快速：較低 VRAM，較低品質。
Base+：此應用程式最佳預設平衡。
最高品質：最慢，負載最重的追蹤器。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="424" />
            <source>Models download automatically on first use. Download progress appears in the status bar.</source>
            <translation>模型將在首次使用時自動下載。下載進度顯示於狀態列。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="432" />
            <source>Manage models</source>
            <translation>管理模型</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="444" />
            <source>Open Cache Folder</source>
            <translation>開啟快取資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="453" />
            <source>Video Tools</source>
            <translation>影片工具</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="456" />
            <source>FFmpeg status</source>
            <translation>FFmpeg 狀態</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="467" />
            <source>Windows: Repair downloads a bundled full FFmpeg build into tools/ffmpeg without changing your system install.
macOS: Repair installs FFmpeg via Homebrew.
Linux: Repair copies the install command to your clipboard.</source>
            <translation>Windows：修復功能將完整的 FFmpeg 套件下載至 tools/ffmpeg，不更改系統安裝。
macOS：修復功能透過 Homebrew 安裝 FFmpeg。
Linux：修復功能將安裝指令複製至您的剪貼簿。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="485" />
            <location filename="../widgets/preferences_dialog.py" line="741" />
            <location filename="../widgets/preferences_dialog.py" line="764" />
            <source>Repair FFmpeg</source>
            <translation>修復 FFmpeg</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="488" />
            <source>Windows: download and install a full bundled FFmpeg build into tools/ffmpeg, validate ffmpeg + ffprobe 7+, and switch CorridorKey to that local copy immediately.

macOS: install FFmpeg via Homebrew and validate ffmpeg + ffprobe 7+.

Linux: do not change system packages. CorridorKey shows the exact install commands and copies them to your clipboard instead.</source>
            <translation>Windows：下載並安裝完整的 FFmpeg 套件至 tools/ffmpeg，驗證 ffmpeg + ffprobe 7 以上版本，並立即切換 CorridorKey 使用該本機副本。

macOS：透過 Homebrew 安裝 FFmpeg 並驗證 ffmpeg + ffprobe 7 以上版本。

Linux：不更改系統套件。CorridorKey 將顯示確切的安裝指令並複製至您的剪貼簿。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="502" />
            <source>Point CorridorKey at your own FFmpeg installation.
Select the folder containing ffmpeg.exe and ffprobe.exe.</source>
            <translation>將 CorridorKey 指向您自己的 FFmpeg 安裝位置。
選擇包含 ffmpeg.exe 和 ffprobe.exe 的資料夾。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="509" />
            <source>Open FFmpeg Folder</source>
            <translation>開啟 FFmpeg 資料夾</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="512" />
            <source>Open CorridorKey's bundled FFmpeg folder.
If Repair FFmpeg has been run on Windows, this is where the local full build is stored.</source>
            <translation>開啟 CorridorKey 內建的 FFmpeg 資料夾。
若已在 Windows 上執行過「修復 FFmpeg」，本機完整套件即存放於此。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="541" />
            <source>Cancel</source>
            <translation>取消</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="545" />
            <source>OK</source>
            <translation>確定</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="622" />
            <source>Select Default Output Directory</source>
            <translation>選擇預設輸出目錄</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="637" />
            <source>Select FFmpeg Folder (containing ffmpeg.exe and ffprobe.exe)</source>
            <translation>選擇 FFmpeg 資料夾（包含 ffmpeg.exe 和 ffprobe.exe）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="660" />
            <source>FFmpeg Not Found</source>
            <translation>找不到 FFmpeg</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="661" />
            <source>Could not find ffmpeg%s in:

%s

Select the folder that contains ffmpeg.exe and ffprobe.exe (usually the 'bin' folder inside the FFmpeg download).</source>
            <translation>在以下路徑找不到 ffmpeg%s：

%s

請選擇包含 ffmpeg.exe 和 ffprobe.exe 的資料夾（通常是 FFmpeg 下載包中的「bin」資料夾）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="670" />
            <source>FFprobe Missing</source>
            <translation>找不到 FFprobe</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="671" />
            <source>Found ffmpeg%s but ffprobe%s is missing from:

%s

CorridorKey requires both. Download a full FFmpeg build.</source>
            <translation>找到 ffmpeg%s，但在以下路徑找不到 ffprobe%s：

%s

CorridorKey 需要兩者都存在。請下載完整的 FFmpeg 套件。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="684" />
            <source>FFmpeg Found</source>
            <translation>找到 FFmpeg</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="688" />
            <source>FFmpeg Issue</source>
            <translation>FFmpeg 問題</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="730" />
            <source>FFmpeg OK</source>
            <translation>FFmpeg 正常</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="731" />
            <source>%s

No repair is needed.</source>
            <translation>%s

無需修復。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="743" />
            <source>

The install command has been copied to your clipboard.
Paste it into a terminal to install.</source>
            <translation>

安裝指令已複製至剪貼簿。
貼至終端機以安裝。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="751" />
            <source>CorridorKey will download and install a full bundled FFmpeg build into:

%s

This does not modify your system-wide FFmpeg.

Continue?</source>
            <translation>CorridorKey 將下載並安裝完整的 FFmpeg 套件至：

%s

這不會修改您的系統 FFmpeg。

是否繼續？</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="757" />
            <source>CorridorKey will install FFmpeg via Homebrew:

    brew install ffmpeg

Continue?</source>
            <translation>CorridorKey 將透過 Homebrew 安裝 FFmpeg：

    brew install ffmpeg

是否繼續？</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="771" />
            <source>Preparing repair...</source>
            <translation>正在準備修復...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="774" />
            <source>Repairing FFmpeg...</source>
            <translation>正在修復 FFmpeg...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="808" />
            <source>FFmpeg Repaired</source>
            <translation>FFmpeg 已修復</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="809" />
            <source>%s

CorridorKey will use FFmpeg immediately.</source>
            <translation>%s

CorridorKey 將立即使用 FFmpeg。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="816" />
            <source>FFmpeg Repair Failed</source>
            <translation>FFmpeg 修復失敗</translation>
        </message>
    </context>
    <context>
        <name>PreviewViewport</name>
        <message>
            <location filename="../widgets/preview_viewport.py" line="235" />
            <source>Extracting frames...
%s</source>
            <translation>正在擷取影格...
%s</translation>
        </message>
        <message>
            <location filename="../widgets/preview_viewport.py" line="262" />
            <source>Selected: %s
State: %s</source>
            <translation>已選取：%s
狀態：%s</translation>
        </message>
        <message>
            <location filename="../widgets/preview_viewport.py" line="403" />
            <source>Toggle A/B wipe comparison (hotkey: A)

Overlays input (A) and current output (B) in one viewer
with a diagonal divider line.

Drag the center handle to slide the line.
Drag above or below the handle to rotate the angle.
Scroll wheel to slide the line (Shift+scroll for fine-grain).
Middle-click the line to reset to default.</source>
            <translation>切換 A/B 擦拭比較（快速鍵：A）

在單一檢視器中以對角分割線疊加輸入（A）和目前輸出（B）。

拖曳中央控制點以移動分割線。
在控制點上方或下方拖曳以旋轉角度。
滾輪移動分割線（Shift + 滾輪可微調）。
中鍵按一下分割線以重置為預設值。</translation>
        </message>
        <message>
            <location filename="../widgets/preview_viewport.py" line="562" />
            <source>No frame available for stem %d</source>
            <translation>沒有可用於音軌 %d 的影格</translation>
        </message>
    </context>
    <context>
        <name>QueuePanel</name>
        <message>
            <location filename="../widgets/queue_panel.py" line="84" />
            <source>Toggle queue panel (Q)</source>
            <translation>切換佇列面板（Q）</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="92" />
            <location filename="../widgets/queue_panel.py" line="121" />
            <source>QUEUE</source>
            <translation>佇列</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="136" />
            <source>Clear</source>
            <translation>清除</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="145" />
            <source>Clear completed and cancelled jobs</source>
            <translation>清除已完成和已取消的工作</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="288" />
            <source>QUEUED</source>
            <translation>已佇列</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="289" />
            <source>PROCESSING</source>
            <translation>處理中</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="290" />
            <source>DONE</source>
            <translation>完成</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="291" />
            <source>CANCELLED</source>
            <translation>已取消</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="292" />
            <source>FAILED</source>
            <translation>失敗</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="334" />
            <source>Dismiss</source>
            <translation>關閉</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="413" />
            <source>Processing...</source>
            <translation>處理中...</translation>
        </message>
    </context>
    <context>
        <name>RecentProjectCard</name>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="57" />
            <source>Open in Finder</source>
            <translation>在 Finder 中開啟</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="57" />
            <source>Open in Explorer</source>
            <translation>在檔案總管中開啟</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="66" />
            <source>Remove project</source>
            <translation>移除專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="110" />
            <source>Rename Project</source>
            <translation>重新命名專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="116" />
            <source>Delete Project</source>
            <translation>刪除專案</translation>
        </message>
    </context>
    <context>
        <name>RecentProjectsPanel</name>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="139" />
            <source>RECENT PROJECTS</source>
            <translation>最近的專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="161" />
            <source>No recent projects</source>
            <translation>無最近的專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="211" />
            <source>Rename Project</source>
            <translation>重新命名專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="211" />
            <source>Project name:</source>
            <translation>專案名稱：</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="238" />
            <source>Remove Project</source>
            <translation>移除專案</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="239" />
            <source>Remove "%s" from recent projects?</source>
            <translation>從最近的專案中移除「%s」？</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="241" />
            <source>Remove from List: hides it from recents (files stay on disk).
Delete from Disk: permanently deletes the project folder.</source>
            <translation>從列表移除：從最近使用中隱藏（檔案保留在磁碟）。
從磁碟刪除：永久刪除專案資料夾。</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="245" />
            <source>Remove from List</source>
            <translation>從列表移除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="246" />
            <source>Delete from Disk</source>
            <translation>從磁碟刪除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="260" />
            <source>Confirm Delete</source>
            <translation>確認刪除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="261" />
            <source>Permanently delete this project folder?

%s</source>
            <translation>永久刪除此專案資料夾？

%s</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="283" />
            <source>Delete Failed</source>
            <translation>刪除失敗</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="284" />
            <source>Could not delete project:
%s</source>
            <translation>無法刪除專案：
%s</translation>
        </message>
    </context>
    <context>
        <name>ReportIssueDialog</name>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="118" />
            <source>Report Issue</source>
            <translation>回報問題</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="130" />
            <source>Issue title:</source>
            <translation>問題標題：</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="132" />
            <source>Brief summary of the problem</source>
            <translation>問題的簡短摘要</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="140" />
            <source>What happened?</source>
            <translation>發生了什麼事？</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="144" />
            <source>Describe what you were doing and what went wrong.
Steps to reproduce are very helpful.</source>
            <translation>請描述您正在執行的操作及發生的問題。
重現步驟非常有幫助。</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="156" />
            <source>System info (auto-collected, included in report)</source>
            <translation>系統資訊（自動收集，已包含在回報中）</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="171" />
            <source>This will open GitHub in your browser. A free GitHub account is required to submit issues. Your report is also copied to the clipboard in case you need to paste it after logging in.</source>
            <translation>此操作將在您的瀏覽器中開啟 GitHub。提交問題需要免費的 GitHub 帳號。您的回報也已複製至剪貼簿，以便登入後貼上。</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="184" />
            <source>Cancel</source>
            <translation>取消</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="188" />
            <source>Open GitHub</source>
            <translation>開啟 GitHub</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="293" />
            <source>Bug Report</source>
            <translation>錯誤回報</translation>
        </message>
    </context>
    <context>
        <name>SetupWizard</name>
        <message>
            <location filename="../widgets/setup_wizard.py" line="652" />
            <source>EZ-CorridorKey Setup</source>
            <translation>EZ-CorridorKey 安裝設定</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="674" />
            <source>Select which models to download. The core CorridorKey model is required.
Optional models can be downloaded later from Edit → Download Manager.</source>
            <translation>選擇要下載的模型。核心 CorridorKey 模型為必要項目。
選用模型可稍後從「編輯」→「下載管理員」下載。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="686" />
            <source>Data directory (models, projects, frame cache):</source>
            <translation>資料目錄（模型、專案、影格快取）：</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="704" />
            <source>Browse...</source>
            <translation>瀏覽...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="713" />
            <source>Default Location</source>
            <translation>預設位置</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="716" />
            <source>Reset the data directory to the platform default (in case you changed it and want to return).</source>
            <translation>將資料目錄重置為平台預設值（若您曾更改並想返回）。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="757" />
            <source>Create Desktop shortcut</source>
            <translation>建立桌面捷徑</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="775" />
            <source>Cancel &amp;&amp; Exit</source>
            <translation>取消並結束</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="786" />
            <source>Download &amp;&amp; Install</source>
            <translation>下載並安裝</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="832" />
            <source>Choose Install Location</source>
            <translation>選擇安裝位置</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="846" />
            <source>Cancelling...</source>
            <translation>正在取消...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="872" />
            <source>Preparing downloads (0/%d)...</source>
            <translation>正在準備下載（0/%d）...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="887" />
            <source>Downloading %d/%d: %s...</source>
            <translation>正在下載 %d/%d：%s...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="913" />
            <source>All %d downloads complete!</source>
            <translation>所有 %d 個下載已完成！</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="917" />
            <source>Some downloads failed. You can retry from Edit → Download Manager.</source>
            <translation>部分下載失敗。您可從「編輯」→「下載管理員」重試。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="925" />
            <source>Continue</source>
            <translation>繼續</translation>
        </message>
    </context>
    <context>
        <name>SplitViewWidget</name>
        <message>
            <location filename="../widgets/split_view.py" line="512" />
            <source>Extracting frames...</source>
            <translation>正在擷取影格...</translation>
        </message>
        <message>
            <location filename="../widgets/split_view.py" line="539" />
            <source>%d%%  (%d/%d frames)</source>
            <translation>%d%%  （%d/%d 影格）</translation>
        </message>
    </context>
    <context>
        <name>StartupDiagnosticDialog</name>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="159" />
            <source>Startup Diagnostics</source>
            <translation>啟動診斷</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="169" />
            <source>EZ-CorridorKey detected issues with your environment that may prevent some features from working correctly.</source>
            <translation>EZ-CorridorKey 偵測到您的環境存在問題，可能導致部分功能無法正常運作。</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="197" />
            <source>Continue Anyway</source>
            <translation>仍要繼續</translation>
        </message>
    </context>
    <context>
        <name>StatusBar</name>
        <message>
            <location filename="../widgets/status_bar.py" line="88" />
            <source>Inference progress for the current job</source>
            <translation>目前工作的推理進度</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="114" />
            <location filename="../widgets/status_bar.py" line="251" />
            <source>RUN INFERENCE</source>
            <translation>執行推理</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="121" />
            <source>Run AI keying on the selected clip (Ctrl+R).
Requires a READY or COMPLETE clip with alpha hints.
Respects in/out range if set (I/O hotkeys).</source>
            <translation>對選取的片段執行 AI 去背（Ctrl+R）。
需要處於「就緒」或「完成」狀態且含 Alpha 提示的片段。
若已設定入出點範圍，將予以遵守（I/O 快速鍵）。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="139" />
            <source>RESUME</source>
            <translation>繼續</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="145" />
            <source>Resume inference — skip already-processed frames,
fill in remaining gaps across the full clip.</source>
            <translation>繼續推理——略過已處理的影格，
補充整個片段中的剩餘缺口。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="154" />
            <location filename="../widgets/status_bar.py" line="203" />
            <source>STOP</source>
            <translation>停止</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="158" />
            <location filename="../widgets/status_bar.py" line="207" />
            <source>Stop the current job (Escape).
Already-processed frames are kept on disk.</source>
            <translation>停止目前工作（Escape）。
已處理的影格將保留在磁碟。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="194" />
            <source>FORCE STOP</source>
            <translation>強制停止</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="198" />
            <source>The current GPU step is blocked.
Force Stop will relaunch the app to break the stuck job.</source>
            <translation>目前 GPU 步驟已卡住。
「強制停止」將重新啟動應用程式以中斷卡住的工作。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="235" />
            <source>RUN EXTRACTION</source>
            <translation>執行擷取</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="239" />
            <source>RUN PIPELINE</source>
            <translation>執行流程</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="243" />
            <source>RUN %d CLIPS</source>
            <translation>執行 %d 個片段</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="247" />
            <source>RUN SELECTED</source>
            <translation>執行已選取</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="345" />
            <source>1 warning</source>
            <translation>1 個警告</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="347" />
            <source>%d warnings</source>
            <translation>%d 個警告</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="353" />
            <source>Latest:
%s

Click for all warnings</source>
            <translation>最新：
%s

按一下查看所有警告</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="406" />
            <source>Warnings (%d)</source>
            <translation>警告（%d）</translation>
        </message>
    </context>
    <context>
        <name>ThumbnailCanvas</name>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="238" />
            <source>%d frames</source>
            <translation>%d 影格</translation>
        </message>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="240" />
            <source>(video)</source>
            <translation>（影片）</translation>
        </message>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="242" />
            <source>(imported)</source>
            <translation>（已匯入）</translation>
        </message>
    </context>
    <context>
        <name>ViewModeBar</name>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="85" />
            <source>Original input footage (unprocessed)

Hotkey: F1</source>
            <translation>原始輸入素材（未處理）

快速鍵：F1</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="87" />
            <source>Tracked mask — SAM2 segmentation output.
White = foreground, black = background.
This is the binary mask before MatAnyone2/VideoMaMa refinement.

Hotkey: F2</source>
            <translation>追蹤遮罩——SAM2 分割輸出。
白色 ＝ 前景，黑色 ＝ 背景。
這是 MatAnyone2/VideoMaMa 精修前的二值遮罩。

快速鍵：F2</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="93" />
            <source>Alpha hint — generated by GVM, VideoMaMa, or MatAnyone2.
White = foreground, black = background.
This is the pre-inference guide used by CorridorKey.

Hotkey: F3</source>
            <translation>Alpha 提示——由 GVM、VideoMaMa 或 MatAnyone2 產生。
白色 ＝ 前景，黑色 ＝ 背景。
這是 CorridorKey 使用的推理前引導。

快速鍵：F3</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="99" />
            <source>Foreground — subject with screen spill removed.
Colors may look shifted; this is the despilled intermediate.

Hotkey: F4</source>
            <translation>前景——已移除螢幕溢色的主體。
色彩可能看起來有所偏移，這是去溢色後的中間結果。

快速鍵：F4</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="104" />
            <source>Alpha matte — white = opaque, black = transparent.
Shows the AI's confidence in foreground vs background.

Hotkey: F5</source>
            <translation>Alpha 遮罩——白色 ＝ 不透明，黑色 ＝ 透明。
顯示 AI 對前景與背景判斷的信心程度。

快速鍵：F5</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="109" />
            <source>Composite — final keyed result over checkerboard.
Best preview of key quality with faithful colors.

Hotkey: F6</source>
            <translation>合成——棋盤格上的最終鍵控結果。
忠實色彩的最佳鍵控品質預覽。

快速鍵：F6</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="114" />
            <source>Processed — production RGBA (straight, linear).
For Resolve, Premiere, and compositing tools.
Preview composites the stored image over black.
Final compositing should happen in your compositor of choice.

Hotkey: F7</source>
            <translation>Processed——製作用 RGBA（直線，線性）。
適用於 Resolve、Premiere 及合成工具。
預覽將儲存的影像合成於黑色背景上。
最終合成應在您選擇的合成軟體中進行。

快速鍵：F7</translation>
        </message>
    </context>
    <context>
        <name>VolumeControl</name>
        <message>
            <location filename="../widgets/volume_control.py" line="32" />
            <source>Click to mute / unmute</source>
            <translation>按一下以靜音 / 取消靜音</translation>
        </message>
        <message>
            <location filename="../widgets/volume_control.py" line="46" />
            <source>Volume</source>
            <translation>音量</translation>
        </message>
    </context>
    <context>
        <name>WelcomeScreen</name>
        <message>
            <location filename="../widgets/welcome_screen.py" line="175" />
            <source>Select Media Files</source>
            <translation>選擇媒體檔案</translation>
        </message>
    </context>
    <context>
        <name>_DropZone</name>
        <message>
            <location filename="../widgets/welcome_screen.py" line="85" />
            <source>Drop Videos, Image Sequences, or Click to Import</source>
            <translation>拖放影片、影像序列，或按一下以匯入</translation>
        </message>
        <message>
            <location filename="../widgets/welcome_screen.py" line="93" />
            <source>Browse...</source>
            <translation>瀏覽...</translation>
        </message>
    </context>
    <context>
        <name>_ModelRow</name>
        <message>
            <location filename="../widgets/setup_wizard.py" line="603" />
            <source>  — Installed</source>
            <translation>  — 已安裝</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="625" />
            <source>Downloading...</source>
            <translation>下載中...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="632" />
            <source>%d / %d MB</source>
            <translation>%d / %d MB</translation>
        </message>
    </context>
</TS>

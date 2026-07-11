<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ja_JP" sourcelanguage="en_US">
    <context>
        <name>BackendStatus</name>
        <message>
            <location filename="../state_labels.py" line="58" />
            <source>Loading model...</source>
            <translation>モデルを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="59" />
            <source>Loading frames...</source>
            <translation>フレームを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="60" />
            <source>Loading masks...</source>
            <translation>マスクを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="61" />
            <source>Loading preview frame...</source>
            <translation>プレビューフレームを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="62" />
            <source>Loading first-frame mask...</source>
            <translation>最初のフレームのマスクを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="63" />
            <source>Loading state dict...</source>
            <translation>状態辞書を読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="64" />
            <source>Loading checkpoint weights...</source>
            <translation>チェックポイントの重みを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="65" />
            <source>Loading MatAnyone2 checkpoint...</source>
            <translation>MatAnyone2 チェックポイントを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="66" />
            <source>Loading MatAnyone2 model...</source>
            <translation>MatAnyone2 モデルを読み込み中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="67" />
            <source>Initializing model backbone...</source>
            <translation>モデルバックボーンを初期化中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="68" />
            <source>Moving model to GPU...</source>
            <translation>モデルを GPU に転送中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="69" />
            <source>Patching attention blocks...</source>
            <translation>アテンションブロックにパッチを適用中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="70" />
            <source>Compiling model (first run may take a minute)...</source>
            <translation>モデルをコンパイル中（初回実行は時間がかかる場合があります）...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="71" />
            <source>Compiling (first frame may take a minute)...</source>
            <translation>コンパイル中（最初のフレームは時間がかかる場合があります）...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="72" />
            <source>Model ready</source>
            <translation>モデル準備完了</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="73" />
            <source>BiRefNet model ready</source>
            <translation>BiRefNet モデル準備完了</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="74" />
            <source>MatAnyone2 model ready</source>
            <translation>MatAnyone2 モデル準備完了</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="75" />
            <source>Running SAM2 tracker...</source>
            <translation>SAM2 トラッカーを実行中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="76" />
            <source>Running BiRefNet inference...</source>
            <translation>BiRefNet 推論を実行中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="77" />
            <source>Running MatAnyone2 inference...</source>
            <translation>MatAnyone2 推論を実行中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="78" />
            <source>Previewing SAM2 on annotated frame...</source>
            <translation>アノテーション済みフレームで SAM2 をプレビュー中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="79" />
            <source>Finalizing alpha hints...</source>
            <translation>アルファヒントを確定中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="80" />
            <source>Releasing Python references...</source>
            <translation>Python 参照を解放中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="81" />
            <source>Waiting for CUDA to finish...</source>
            <translation>CUDA の完了を待機中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="82" />
            <source>Clearing CUDA cache...</source>
            <translation>CUDA キャッシュをクリア中...</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="83" />
            <source>UNet forward pass</source>
            <translation>UNet フォワードパス</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="84" />
            <source>VAE encode</source>
            <translation>VAE エンコード</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="85" />
            <source>CLIP encode</source>
            <translation>CLIP エンコード</translation>
        </message>
    </context>
    <context>
        <name>BatchPipelineDialog</name>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="69" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="96" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="500" />
            <source>Batch Pipeline</source>
            <translation>バッチパイプライン</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="101" />
            <source>Select a folder containing video clips. Files with "alphahint" or "maskhint" in the name are automatically paired as hints.</source>
            <translation>動画クリップが入ったフォルダーを選択してください。名前に "alphahint" または "maskhint" を含むファイルは、ヒントとして自動的に対応付けられます。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="111" />
            <source>Select Folder...</source>
            <translation>フォルダーを選択...</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="115" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="462" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="501" />
            <source>No folder selected</source>
            <translation>フォルダーが選択されていません</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="122" />
            <source>Global Settings</source>
            <translation>全体設定</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="127" />
            <source>No-hint clips:</source>
            <translation>ヒントなしクリップ:</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="130" />
            <source>Alpha generation method for clips with no companion hint file.
GVM: fast automatic alpha.
BiRefNet: higher quality, select a model variant.</source>
            <translation>対応するヒントファイルがないクリップのアルファ生成方法です。
GVM: 高速な自動アルファ生成。
BiRefNet: より高品質。モデルバリエーションを選択してください。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="157" />
            <source>MaskHint clips:</source>
            <translation>MaskHint クリップ:</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="160" />
            <source>Mask refinement method for clips with a companion MaskHint file.
VideoMaMa: temporal consistency, best for video.
MatAnyone2: single-frame matting with mask guidance.</source>
            <translation>対応する MaskHint ファイルがあるクリップのマスク補正方法です。
VideoMaMa: 時間的一貫性を重視し、動画に最適です。
MatAnyone2: マスクガイドを使ったシングルフレームマッティングです。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="176" />
            <source>Per-clip overrides</source>
            <translation>クリップ別の設定上書き</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Clip</source>
            <translation>クリップ</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Detected</source>
            <translation>検出</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Pipeline</source>
            <translation>パイプライン</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="183" />
            <source>Status</source>
            <translation>状態</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="204" />
            <source>Clear Pipeline</source>
            <translation>パイプラインを消去</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="205" />
            <source>Cancel all pending batch jobs and reset.</source>
            <translation>保留中のバッチジョブをすべてキャンセルしてリセットします。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="210" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="509" />
            <source>Cancel</source>
            <translation>キャンセル</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="213" />
            <location filename="../widgets/batch_pipeline_dialog.py" line="508" />
            <source>Run Batch</source>
            <translation>バッチを実行</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="216" />
            <source>Inference settings (despill, refiner, edge, color space, etc.) are inherited from the right panel. Adjust them there before running.</source>
            <translation>推論設定（スピル除去、リファイナー、エッジ、カラースペースなど）は右パネルの設定を引き継ぎます。実行前にそちらで調整してください。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="234" />
            <source>Select Batch Folder</source>
            <translation>バッチフォルダーを選択</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="267" />
            <source>No hint</source>
            <translation>ヒントなし</translation>
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
            <translation>CK 推論</translation>
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
            <translation>%d 個のクリップが見つかりました: %s</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="356" />
            <source>No video clips found in this folder.</source>
            <translation>このフォルダーに動画クリップが見つかりませんでした。</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="476" />
            <source>Batch Pipeline - Processing</source>
            <translation>バッチパイプライン - 処理中</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="478" />
            <source>Running...</source>
            <translation>実行中...</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="569" />
            <source>Processing failed</source>
            <translation>処理に失敗しました</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="576" />
            <source>Batch Pipeline - Complete</source>
            <translation>バッチパイプライン - 完了</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="577" />
            <source>Done</source>
            <translation>完了</translation>
        </message>
        <message>
            <location filename="../widgets/batch_pipeline_dialog.py" line="579" />
            <source>Close</source>
            <translation>閉じる</translation>
        </message>
    </context>
    <context>
        <name>ClipListModel</name>
        <message>
            <location filename="../models/clip_model.py" line="73" />
            <source>State: %s</source>
            <translation>状態: %s</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="76" />
            <source>Input: %d frames (%s)</source>
            <translation>入力: %d フレーム (%s)</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="81" />
            <source>Alpha: %d frames</source>
            <translation>アルファ: %d フレーム</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="86" />
            <source>Warnings: %d</source>
            <translation>警告: %d</translation>
        </message>
        <message>
            <location filename="../models/clip_model.py" line="90" />
            <source>Error: %s</source>
            <translation>エラー: %s</translation>
        </message>
    </context>
    <context>
        <name>ClipState</name>
        <message>
            <location filename="../state_labels.py" line="18" />
            <source>EXTRACTING</source>
            <translation>抽出中</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="19" />
            <source>RAW</source>
            <translation>未処理</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="20" />
            <source>MASKED</source>
            <translation>マスク済</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="21" />
            <source>READY</source>
            <translation>準備完了</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="22" />
            <source>COMPLETE</source>
            <translation>完了</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="23" />
            <source>ERROR</source>
            <translation>エラー</translation>
        </message>
    </context>
    <context>
        <name>DebugConsoleWidget</name>
        <message>
            <location filename="../widgets/debug_console.py" line="86" />
            <source>Console</source>
            <translation>コンソール</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="129" />
            <source>CONSOLE</source>
            <translation>コンソール</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="172" />
            <source>Level:</source>
            <translation>レベル:</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="178" />
            <location filename="../widgets/debug_console.py" line="334" />
            <source>Pause</source>
            <translation>一時停止</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="185" />
            <source>Clear</source>
            <translation>消去</translation>
        </message>
        <message>
            <location filename="../widgets/debug_console.py" line="334" />
            <source>Resume</source>
            <translation>再開</translation>
        </message>
    </context>
    <context>
        <name>DiagnosticDialog</name>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="47" />
            <source>Diagnostic: %s</source>
            <translation>診断: %s</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="115" />
            <source>Error: %s</source>
            <translation>エラー: %s</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="129" />
            <source>Report Issue on GitHub</source>
            <translation>GitHub で問題を報告</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="136" />
            <source>OK</source>
            <translation>OK</translation>
        </message>
    </context>
    <context>
        <name>FrameScrubber</name>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="52" />
            <source>Go to first frame</source>
            <translation>最初のフレームへ移動</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="60" />
            <source>Previous frame</source>
            <translation>前のフレーム</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="68" />
            <source>Play / Pause (Space)</source>
            <translation>再生 / 一時停止 (Space)</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="82" />
            <source>Coverage bar — shows which frames have been processed.
Green lane: painted frames (brush strokes).
White lane: alpha hint coverage.
Yellow lane: inference output coverage.</source>
            <translation>カバレッジバー — 処理済みフレームを表示します。
緑のレーン: ペイント済みフレーム（ブラシストローク）。
白のレーン: アルファヒントのカバレッジ。
黄のレーン: 推論結果のカバレッジ。</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="95" />
            <source>Scrub through frames. Scroll wheel or Left/Right to step.</source>
            <translation>フレームをスクラブします。スクロールホイールまたは左右キーでステップ移動できます。</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="118" />
            <source>Next frame</source>
            <translation>次のフレーム</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="126" />
            <source>Go to last frame</source>
            <translation>最後のフレームへ移動</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="310" />
            <source>Pause (Space)</source>
            <translation>一時停止 (Space)</translation>
        </message>
        <message>
            <location filename="../widgets/frame_scrubber.py" line="317" />
            <source>Play (Space)</source>
            <translation>再生 (Space)</translation>
        </message>
    </context>
    <context>
        <name>HotkeysDialog</name>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="133" />
            <source>Hotkeys</source>
            <translation>ショートカット</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="145" />
            <source>Filter shortcuts...</source>
            <translation>ショートカットを検索...</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="212" />
            <source>Reset</source>
            <translation>リセット</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="220" />
            <source>Reset to default: %s</source>
            <translation>デフォルトにリセット: %s</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="240" />
            <source>Reset All to Defaults</source>
            <translation>すべてデフォルトにリセット</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="252" />
            <source>Cancel</source>
            <translation>キャンセル</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="263" />
            <source>OK</source>
            <translation>OK</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="309" />
            <source>Reset All Shortcuts</source>
            <translation>すべてのショートカットをリセット</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="310" />
            <source>Reset all shortcuts to their default values?</source>
            <translation>すべてのショートカットをデフォルト値にリセットしますか？</translation>
        </message>
    </context>
    <context>
        <name>IOTrayActionsMixin</name>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="63" />
            <source>Run Extraction (%d clips)</source>
            <translation>抽出を実行 (%d 個のクリップ)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="64" />
            <source>Run Extraction</source>
            <translation>抽出を実行</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="72" />
            <source>Rename...</source>
            <translation>名前を変更...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="78" />
            <source>Finder</source>
            <translation>Finder</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="78" />
            <source>Explorer</source>
            <translation>エクスプローラー</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="79" />
            <source>Open in %s</source>
            <translation>%s で開く</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="89" />
            <source>Clear Mask (%d clips)</source>
            <translation>マスクを消去 (%d 個のクリップ)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="89" />
            <location filename="../widgets/io_tray_actions.py" line="241" />
            <source>Clear Mask</source>
            <translation>マスクを消去</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="97" />
            <source>Clear Alpha (%d clips)</source>
            <translation>アルファを消去 (%d 個のクリップ)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="97" />
            <location filename="../widgets/io_tray_actions.py" line="352" />
            <source>Clear Alpha</source>
            <translation>アルファを消去</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="105" />
            <source>Clear Outputs (%d clips)</source>
            <translation>出力を消去 (%d 個のクリップ)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="105" />
            <location filename="../widgets/io_tray_actions.py" line="385" />
            <source>Clear Outputs</source>
            <translation>出力を消去</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="113" />
            <source>Clear All (%d clips)</source>
            <translation>すべて消去 (%d 個のクリップ)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="113" />
            <location filename="../widgets/io_tray_actions.py" line="306" />
            <source>Clear All</source>
            <translation>すべて消去</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="120" />
            <source>Set Output Directory...</source>
            <translation>出力フォルダーを設定...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="126" />
            <source>Clear Output Directory Override</source>
            <translation>出力フォルダーの上書き設定を消去</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="133" />
            <source>Remove (%d clips)...</source>
            <translation>削除 (%d 個のクリップ)...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="133" />
            <source>Remove...</source>
            <translation>削除...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="157" />
            <source>Export %s as Video...</source>
            <translation>%s をビデオとして書き出し...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="169" />
            <source>Open Containing Folder</source>
            <translation>親フォルダーを開く</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="183" />
            <source>Output Directory for '%s'</source>
            <translation>「%s」の出力フォルダー</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="214" />
            <source>Rename Clip</source>
            <translation>クリップ名を変更</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="214" />
            <source>New name:</source>
            <translation>新しい名前:</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="242" />
            <source>Delete tracked masks for %d clip(s)?
%s

This will remove all SAM2 mask frames from disk.</source>
            <translation>%d 個のクリップのトラッキング済みマスクを削除しますか？
%s

ディスクからすべての SAM2 マスクフレームが削除されます。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="307" />
            <source>Remove ALL generated data for %d clip(s)?
%s

This will delete masks, alpha hints, and all output frames.</source>
            <translation>%d 個のクリップの生成データをすべて削除しますか？
%s

マスク、アルファヒント、およびすべての出力フレームが削除されます。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="353" />
            <source>Delete AlphaHint for %d clip(s)?
%s

This will remove all generated alpha hint frames from disk.</source>
            <translation>%d 個のクリップの AlphaHint を削除しますか？
%s

ディスクからすべての生成済みアルファヒントフレームが削除されます。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="386" />
            <source>Remove all output files for %d clip(s)?
%s

This will delete FG, Matte, Comp, and Processed frames.</source>
            <translation>%d 個のクリップの出力ファイルをすべて削除しますか？
%s

FG、Matte、Comp、および Processed のフレームが削除されます。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="412" />
            <source>Remove %d clip(s)?</source>
            <translation>%d 個のクリップを削除しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="416" />
            <source>
... and %d more</source>
            <translation>
... 他 %d 件</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="421" />
            <source>How would you like to remove %d clip(s)?</source>
            <translation>%d 個のクリップをどのように削除しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="424" />
            <source>Remove from List</source>
            <translation>リストから削除</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_actions.py" line="425" />
            <source>Delete from Disk</source>
            <translation>ディスクから削除</translation>
        </message>
    </context>
    <context>
        <name>IOTrayPanel</name>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="81" />
            <source>INPUT (0)</source>
            <translation>入力 (0)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="86" />
            <source>RESET I/O</source>
            <translation>イン/アウトをリセット</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="88" />
            <source>Clear in/out markers on all clips</source>
            <translation>すべてのクリップのイン/アウトマーカーを消去します。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="92" />
            <source>+ ADD</source>
            <translation>+ 追加</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="94" />
            <source>Import clips — choose a folder or video file(s)</source>
            <translation>クリップを読み込みます。フォルダーまたは動画ファイルを選択してください。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="122" />
            <source>EXPORTS (0)</source>
            <translation>書き出し (0)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="169" />
            <source>Import Folder...</source>
            <translation>フォルダーを読み込み...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="170" />
            <source>Import Video(s)...</source>
            <translation>動画を読み込み...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="171" />
            <source>Import Image Sequence...</source>
            <translation>連番画像を読み込み...</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="180" />
            <source>No Markers</source>
            <translation>マーカーなし</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="181" />
            <source>No clips have in/out markers set.</source>
            <translation>イン/アウトマーカーが設定されているクリップはありません。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="188" />
            <source>Reset In/Out Markers</source>
            <translation>イン/アウトマーカーをリセット</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="189" />
            <source>This will clear in/out markers on %d clip(s).

All clips will revert to full-clip processing.
Continue?</source>
            <translation>%d 個のクリップのイン/アウトマーカーが消去されます。

すべてのクリップがクリップ全体の処理に戻ります。
続行しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="199" />
            <source>Confirm Reset</source>
            <translation>リセットの確認</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="200" />
            <source>Are you sure? This cannot be undone.

Clearing in/out markers on %d clip(s).</source>
            <translation>本当によろしいですか？この操作は元に戻せません。

%d 個のクリップのイン/アウトマーカーを消去します。</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="212" />
            <source>Select Clips Directory</source>
            <translation>クリップのフォルダーを選択</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="220" />
            <source>Select Video Files</source>
            <translation>動画ファイルを選択</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="228" />
            <source>Select Image Sequence Folder</source>
            <translation>連番画像フォルダーを選択</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="359" />
            <source>INPUT (%d)</source>
            <translation>入力 (%d)</translation>
        </message>
        <message>
            <location filename="../widgets/io_tray_panel.py" line="360" />
            <source>EXPORTS (%d)</source>
            <translation>書き出し (%d)</translation>
        </message>
    </context>
    <context>
        <name>JobType</name>
        <message>
            <location filename="../state_labels.py" line="37" />
            <source>Inference</source>
            <translation>推論</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="40" />
            <source>Track Preview</source>
            <translation>トラックプレビュー</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="41" />
            <source>Track Mask</source>
            <translation>トラックマスク</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="44" />
            <source>Preview</source>
            <translation>プレビュー</translation>
        </message>
        <message>
            <location filename="../state_labels.py" line="46" />
            <source>Pipeline</source>
            <translation>パイプライン</translation>
        </message>
    </context>
    <context>
        <name>KeyBindButton</name>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="56" />
            <source>(none)</source>
            <translation>（なし）</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="62" />
            <source>Press a key...</source>
            <translation>キーを押してください...</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="94" />
            <source>Shortcut Conflict</source>
            <translation>ショートカットの競合</translation>
        </message>
        <message>
            <location filename="../widgets/hotkeys_dialog.py" line="95" />
            <source>"%s" is already assigned to:
%s

Reassign anyway? The conflicting binding will be cleared.</source>
            <translation>「%s」は既に次の操作に割り当てられています:
%s

それでも再割り当てしますか？競合するバインドは消去されます。</translation>
        </message>
    </context>
    <context>
        <name>MainWindow</name>
        <message>
            <location filename="../main_window.py" line="292" />
            <source>%s — Mac Performance Warning</source>
            <translation>%s — Mac パフォーマンス警告</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="294" />
            <source>GPU-intensive features (SAM2, GVM, VideoMaMa, MatAnyone2) are very slow on Mac (Apple Silicon MPS).

This may take hours for longer clips and could freeze your system.

Recommendation: Import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke instead.

Continue anyway? (This warning won't appear again this session.)</source>
            <translation>GPU を多用する機能（SAM2、GVM、VideoMaMa、MatAnyone2）は Mac（Apple Silicon MPS）では非常に低速です。

長いクリップでは数時間かかることがあり、システムがフリーズする可能性があります。

推奨: After Effects、DaVinci Resolve、または Nuke で作成済みのアルファマットを読み込んでご使用ください。

それでも続行しますか？（この警告はセッション中に再表示されません。）</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="312" />
            <source>EZ-CorridorKey</source>
            <translation>EZ-CorridorKey</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="470" />
            <source>Detected GPU used for inference</source>
            <translation>推論に使用する GPU を検出しました</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="473" />
            <source>VRAM</source>
            <translation>VRAM</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="484" />
            <source>GPU video memory usage — updates during inference</source>
            <translation>GPU ビデオメモリ使用量 — 推論中に更新されます。</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="490" />
            <source>Current VRAM used / total available</source>
            <translation>現在の VRAM 使用量 / 利用可能な合計</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="682" />
            <source>No GPU</source>
            <translation>GPU なし</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="689" />
            <source>Memory</source>
            <translation>メモリ</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="690" />
            <source>Unified memory usage — CPU and GPU share the same pool</source>
            <translation>ユニファイドメモリ使用量 — CPU と GPU が同じメモリプールを共有します。</translation>
        </message>
        <message>
            <location filename="../main_window.py" line="691" />
            <source>Current unified memory used / total available</source>
            <translation>現在のユニファイドメモリ使用量 / 利用可能な合計</translation>
        </message>
        <message>
            <source>%s complete for %s%s -- Ready to Run Inference</source>
            <translation>%s が %s%s で完了 -- 推論実行の準備ができました</translation>
        </message>
        <message>
            <source>'%s' is already queued.</source>
            <translation>'%s' はすでにキューに追加されています。</translation>
        </message>
        <message>
            <source>A new version (v%s) is available.
Click to save your session and run the updater.</source>
            <translation>新しいバージョン (v%s) が利用可能です。
クリックしてセッションを保存し、アップデーターを実行してください。</translation>
        </message>
        <message>
            <source>AI Green Screen Keyer</source>
            <translation>AI グリーンスクリーン キーヤー</translation>
        </message>
        <message>
            <source>About</source>
            <translation>情報</translation>
        </message>
        <message>
            <source>About EZ-CorridorKey</source>
            <translation>EZ-CorridorKey について</translation>
        </message>
        <message>
            <source>All selected videos are already in the project (%s).</source>
            <translation>選択したすべての動画はすでにプロジェクトに含まれています (%s)。</translation>
        </message>
        <message>
            <source>Alpha hints cover %d of %d frames.

You can process the available range, re-run GVM to
regenerate all alpha frames, or cancel.</source>
            <translation>アルファヒントは %d / %d フレームをカバーしています。

利用可能な範囲を処理するか、GVM を再実行して
すべてのアルファフレームを再生成するか、キャンセルできます。</translation>
        </message>
        <message>
            <source>Already Imported</source>
            <translation>インポート済み</translation>
        </message>
        <message>
            <source>Automatic updates are not supported on this platform.
Please download the latest release from GitHub.</source>
            <translation>このプラットフォームでは自動更新はサポートされていません。
最新リリースを GitHub からダウンロードしてください。</translation>
        </message>
        <message>
            <source>Batch Export</source>
            <translation>一括書き出し</translation>
        </message>
        <message>
            <source>Batch Export Complete</source>
            <translation>一括書き出し完了</translation>
        </message>
        <message>
            <source>Batch Pipeline...</source>
            <translation>バッチパイプライン...</translation>
        </message>
        <message>
            <source>Blue Screen Model Required</source>
            <translation>ブルースクリーンモデルが必要です</translation>
        </message>
        <message>
            <source>Cancel</source>
            <translation>キャンセル</translation>
        </message>
        <message>
            <source>Cancel %s?</source>
            <translation>%s をキャンセルしますか?</translation>
        </message>
        <message>
            <source>Cancel all pending batch jobs and clear the pipeline?</source>
            <translation>保留中のすべてのバッチジョブをキャンセルし、パイプラインをクリアしますか?</translation>
        </message>
        <message>
            <source>Cancel processing?</source>
            <translation>処理をキャンセルしますか?</translation>
        </message>
        <message>
            <source>Cancelled queued work.</source>
            <translation>キューの作業をキャンセルしました。</translation>
        </message>
        <message>
            <source>Cancelled: %s</source>
            <translation>キャンセル済み: %s</translation>
        </message>
        <message>
            <source>Clear Batch Pipeline</source>
            <translation>バッチパイプラインをクリア</translation>
        </message>
        <message>
            <source>Clear Holdout Strokes</source>
            <translation>除外ストロークをクリア</translation>
        </message>
        <message>
            <source>Clear Paint Strokes</source>
            <translation>ペイントストロークをクリア</translation>
        </message>
        <message>
            <source>Clear Project Output Folder</source>
            <translation>プロジェクト出力フォルダーをクリア</translation>
        </message>
        <message>
            <source>Clear all holdout mask strokes for this clip?</source>
            <translation>このクリップのすべての除外マスクストロークをクリアしますか?</translation>
        </message>
        <message>
            <source>Clear the current batch folder and clip list?</source>
            <translation>現在のバッチフォルダーとクリップリストをクリアしますか?</translation>
        </message>
        <message>
            <source>Clip '%s' already has alpha hint images.

Do you want to replace them with chroma key hints?</source>
            <translation>クリップ '%s' にはすでにアルファヒント画像があります。

クロマキーヒントに置き換えますか?</translation>
        </message>
        <message>
            <source>Clip '%s' already has alpha hint images.

Do you want to replace them with new ones?</source>
            <translation>クリップ '%s' にはすでにアルファヒント画像があります。

新しい画像に置き換えますか?</translation>
        </message>
        <message>
            <source>Clip '%s' has %d input frames but you selected %d alpha hints.

Each input frame needs a matching alpha hint.
Only %d frames will be paired.</source>
            <translation>クリップ '%s' の入力フレーム数は %d ですが、%d 個のアルファヒントが選択されました。

各入力フレームには対応するアルファヒントが必要です。
%d フレームのみがペアになります。</translation>
        </message>
        <message>
            <source>Clip '%s' is in %s state.
Only READY or COMPLETE clips can be processed.</source>
            <translation>クリップ '%s' は %s 状態です。
処理できるのは READY または COMPLETE のクリップのみです。</translation>
        </message>
        <message>
            <source>Clip '%s' must be COMPLETE to export video.</source>
            <translation>動画を書き出すには、クリップ '%s' が COMPLETE 状態である必要があります。</translation>
        </message>
        <message>
            <source>Clip / Output</source>
            <translation>クリップ / 出力</translation>
        </message>
        <message>
            <source>Clip: %s

%s</source>
            <translation>クリップ: %s

%s</translation>
        </message>
        <message>
            <source>Console</source>
            <translation>コンソール</translation>
        </message>
        <message>
            <source>Copy Just These %d</source>
            <translation>この %d 件のみコピー</translation>
        </message>
        <message>
            <source>Could not load the selected language. The interface stays in English.</source>
            <translation>選択した言語を読み込めませんでした。インターフェースは英語のままになります。</translation>
        </message>
        <message>
            <source>Could not read frame count from the selected alpha video.</source>
            <translation>選択したアルファ動画からフレーム数を読み取れませんでした。</translation>
        </message>
        <message>
            <source>Could not relaunch the app automatically.

Please close and reopen EZ-CorridorKey manually.</source>
            <translation>アプリを自動的に再起動できませんでした。

EZ-CorridorKey を手動で閉じて再度開いてください。</translation>
        </message>
        <message>
            <source>Could not update automatically:

%s

Please download the latest release manually from GitHub.</source>
            <translation>自動更新できませんでした:

%s

最新リリースを GitHub から手動でダウンロードしてください。</translation>
        </message>
        <message>
            <source>Download Manager</source>
            <translation>ダウンロードマネージャー</translation>
        </message>
        <message>
            <source>Download Manager...</source>
            <translation>ダウンロードマネージャー...</translation>
        </message>
        <message>
            <source>Downloading update...</source>
            <translation>アップデートをダウンロード中...</translation>
        </message>
        <message>
            <source>Duplicate</source>
            <translation>重複</translation>
        </message>
        <message>
            <source>Duplicate Filenames</source>
            <translation>ファイル名の重複</translation>
        </message>
        <message>
            <source>EZ-CorridorKey is already updated to v%s.

Restart the app to load the new version.</source>
            <translation>EZ-CorridorKey はすでに v%s に更新されています。

アプリを再起動して新しいバージョンを読み込んでください。</translation>
        </message>
        <message>
            <source>Edit</source>
            <translation>編集</translation>
        </message>
        <message>
            <source>Entire Clip</source>
            <translation>クリップ全体</translation>
        </message>
        <message>
            <source>Exit</source>
            <translation>終了</translation>
        </message>
        <message>
            <source>Export All Videos</source>
            <translation>すべての動画を書き出す</translation>
        </message>
        <message>
            <source>Export Complete</source>
            <translation>書き出し完了</translation>
        </message>
        <message>
            <source>Export Failed</source>
            <translation>書き出し失敗</translation>
        </message>
        <message>
            <source>Export Video</source>
            <translation>動画を書き出す</translation>
        </message>
        <message>
            <source>Export Video...</source>
            <translation>動画を書き出す...</translation>
        </message>
        <message>
            <source>Exporting %s / %s...</source>
            <translation>%s / %s を書き出し中...</translation>
        </message>
        <message>
            <source>Exporting %s...</source>
            <translation>%s を書き出し中...</translation>
        </message>
        <message>
            <source>Exporting videos...</source>
            <translation>動画を書き出し中...</translation>
        </message>
        <message>
            <source>FFmpeg Unavailable</source>
            <translation>FFmpeg が利用できません</translation>
        </message>
        <message>
            <source>Failed to export video:
%s</source>
            <translation>動画の書き出しに失敗しました:
%s</translation>
        </message>
        <message>
            <source>Failed to import alpha hints:
%s</source>
            <translation>アルファヒントのインポートに失敗しました:
%s</translation>
        </message>
        <message>
            <source>Failed to scan clips directory:
%s</source>
            <translation>クリップディレクトリのスキャンに失敗しました:
%s</translation>
        </message>
        <message>
            <source>File</source>
            <translation>ファイル</translation>
        </message>
        <message>
            <source>Force Stop</source>
            <translation>強制停止</translation>
        </message>
        <message>
            <source>Force Stop Failed</source>
            <translation>強制停止に失敗しました</translation>
        </message>
        <message>
            <source>Force restarting...</source>
            <translation>強制的に再起動中...</translation>
        </message>
        <message>
            <source>Foreground color: %s</source>
            <translation>前景色: %s</translation>
        </message>
        <message>
            <source>Format</source>
            <translation>フォーマット</translation>
        </message>
        <message>
            <source>Found %d/%d alpha frames from a previous run.</source>
            <translation>前回の実行から %d / %d のアルファフレームが見つかりました。</translation>
        </message>
        <message>
            <source>Found files with the same name but different extensions:
%s

This would cause output file conflicts. Please use one format per sequence folder.</source>
            <translation>同じ名前で異なる拡張子のファイルが見つかりました:
%s

これにより出力ファイルが競合します。シーケンスフォルダーごとに1つのフォーマットをご使用ください。</translation>
        </message>
        <message>
            <source>Frame Count Mismatch</source>
            <translation>フレーム数の不一致</translation>
        </message>
        <message>
            <source>Frames</source>
            <translation>フレーム</translation>
        </message>
        <message>
            <source>GPU is finishing the current chunk.
VideoMaMa will stop after it completes.</source>
            <translation>GPU が現在のチャンクを処理中です。
VideoMaMa は完了後に停止します。</translation>
        </message>
        <message>
            <source>Give your project a name:</source>
            <translation>プロジェクト名を入力してください:</translation>
        </message>
        <message>
            <source>Help</source>
            <translation>ヘルプ</translation>
        </message>
        <message>
            <source>Hotkeys...</source>
            <translation>ホットキー...</translation>
        </message>
        <message>
            <source>How would you like to import?</source>
            <translation>インポート方法を選択してください:</translation>
        </message>
        <message>
            <source>Image Folder</source>
            <translation>画像フォルダー</translation>
        </message>
        <message>
            <source>Import Alpha</source>
            <translation>アルファをインポート</translation>
        </message>
        <message>
            <source>Import Alpha Failed</source>
            <translation>アルファのインポートに失敗しました</translation>
        </message>
        <message>
            <source>Import Clips</source>
            <translation>クリップをインポート</translation>
        </message>
        <message>
            <source>Import Folder...</source>
            <translation>フォルダーをインポート...</translation>
        </message>
        <message>
            <source>Import Full Sequence</source>
            <translation>シーケンス全体をインポート</translation>
        </message>
        <message>
            <source>Import Image Frames</source>
            <translation>画像フレームをインポート</translation>
        </message>
        <message>
            <source>Import Image Sequence...</source>
            <translation>画像シーケンスをインポート...</translation>
        </message>
        <message>
            <source>Import Video(s)...</source>
            <translation>動画をインポート...</translation>
        </message>
        <message>
            <source>Import alpha from an image folder or a video file?</source>
            <translation>画像フォルダーまたは動画ファイルからアルファをインポートしますか?</translation>
        </message>
        <message>
            <source>Imported %d/%d %s from video.
Clip is now %s.</source>
            <translation>動画から %d / %d の %s をインポートしました。
クリップは現在 %s です。</translation>
        </message>
        <message>
            <source>Imported %d/%d %s.
Clip is now %s.</source>
            <translation>%d / %d の %s をインポートしました。
クリップは現在 %s です。</translation>
        </message>
        <message>
            <source>Incomplete Alpha</source>
            <translation>アルファが不完全です</translation>
        </message>
        <message>
            <source>Inference complete: %s</source>
            <translation>推論完了: %s</translation>
        </message>
        <message>
            <source>Installing update...</source>
            <translation>アップデートをインストール中...</translation>
        </message>
        <message>
            <source>Language</source>
            <translation>言語</translation>
        </message>
        <message>
            <source>MatAnyone2 requires a tracked mask on frame 0.

Paint prompts and run Track Mask before using MatAnyone2.</source>
            <translation>MatAnyone2 にはフレーム0のトラックされたマスクが必要です。

MatAnyone2 を使用する前に、ペイントプロンプトを描いてトラックマスクを実行してください。</translation>
        </message>
        <message>
            <source>Missing</source>
            <translation>不足</translation>
        </message>
        <message>
            <source>Name Your Project</source>
            <translation>プロジェクト名を設定</translation>
        </message>
        <message>
            <source>New Project</source>
            <translation>新規プロジェクト</translation>
        </message>
        <message>
            <source>No %s found in the latest release.
Release: %s

Please download manually from GitHub.</source>
            <translation>最新リリースに %s が見つかりませんでした。
リリース: %s

GitHub から手動でダウンロードしてください。</translation>
        </message>
        <message>
            <source>No COMPLETE clips to export.</source>
            <translation>書き出せる COMPLETE クリップがありません。</translation>
        </message>
        <message>
            <source>No Clip</source>
            <translation>クリップなし</translation>
        </message>
        <message>
            <source>No Clips</source>
            <translation>クリップなし</translation>
        </message>
        <message>
            <source>No Folder</source>
            <translation>フォルダーなし</translation>
        </message>
        <message>
            <source>No Frames</source>
            <translation>フレームなし</translation>
        </message>
        <message>
            <source>No Images</source>
            <translation>画像なし</translation>
        </message>
        <message>
            <source>No Media</source>
            <translation>メディアなし</translation>
        </message>
        <message>
            <source>No Output</source>
            <translation>出力なし</translation>
        </message>
        <message>
            <source>No Paint Strokes</source>
            <translation>ペイントストロークなし</translation>
        </message>
        <message>
            <source>No Project</source>
            <translation>プロジェクトなし</translation>
        </message>
        <message>
            <source>No READY clips to process.</source>
            <translation>処理できる READY クリップがありません。</translation>
        </message>
        <message>
            <source>No clip selected</source>
            <translation>クリップが選択されていません</translation>
        </message>
        <message>
            <source>No image files found in that folder.

Supported formats: PNG, JPG, EXR, TIF, TIFF, BMP, DPX</source>
            <translation>そのフォルダーに画像ファイルが見つかりませんでした。

対応フォーマット: PNG, JPG, EXR, TIF, TIFF, BMP, DPX</translation>
        </message>
        <message>
            <source>No image files found in the selected folder.
Expected grayscale images (white=foreground, black=background).</source>
            <translation>選択したフォルダーに画像ファイルが見つかりませんでした。
グレースケール画像 (白=前景、黒=背景) が必要です。</translation>
        </message>
        <message>
            <source>No image frames found in output directory.</source>
            <translation>出力ディレクトリに画像フレームが見つかりませんでした。</translation>
        </message>
        <message>
            <source>No output frames found to export.</source>
            <translation>書き出すべき出力フレームが見つかりませんでした。</translation>
        </message>
        <message>
            <source>No output frames found.</source>
            <translation>出力フレームが見つかりませんでした。</translation>
        </message>
        <message>
            <source>No selected clips are in a processable state.</source>
            <translation>選択したクリップの中に処理可能な状態のものがありません。</translation>
        </message>
        <message>
            <source>No video files or image sequences found in that folder.</source>
            <translation>そのフォルダーに動画ファイルまたは画像シーケンスが見つかりませんでした。</translation>
        </message>
        <message>
            <source>Not Complete</source>
            <translation>未完了</translation>
        </message>
        <message>
            <source>Not Ready</source>
            <translation>準備未完了</translation>
        </message>
        <message>
            <source>Nothing to Export</source>
            <translation>書き出すものがありません</translation>
        </message>
        <message>
            <source>Nothing to Process</source>
            <translation>処理するものがありません</translation>
        </message>
        <message>
            <source>Open Project...</source>
            <translation>プロジェクトを開く...</translation>
        </message>
        <message>
            <source>Open a clips folder first.</source>
            <translation>最初にクリップフォルダーを開いてください。</translation>
        </message>
        <message>
            <source>Open a project first.</source>
            <translation>最初にプロジェクトを開いてください。</translation>
        </message>
        <message>
            <source>Paint green (1) and red (2) strokes on frames first.</source>
            <translation>最初にフレームに緑 (1) と赤 (2) のストロークを描いてください。</translation>
        </message>
        <message>
            <source>Paint prompts and run Track Mask before using VideoMaMa.</source>
            <translation>VideoMaMa を使用する前に、ペイントプロンプトを描いてトラックマスクを実行してください。</translation>
        </message>
        <message>
            <source>Partial Alpha Found</source>
            <translation>部分的なアルファが見つかりました</translation>
        </message>
        <message>
            <source>Preferences...</source>
            <translation>環境設定...</translation>
        </message>
        <message>
            <source>Process Available</source>
            <translation>処理を実行</translation>
        </message>
        <message>
            <source>Processing Error</source>
            <translation>処理エラー</translation>
        </message>
        <message>
            <source>Project name for this batch:</source>
            <translation>このバッチのプロジェクト名:</translation>
        </message>
        <message>
            <source>Re-run GVM</source>
            <translation>GVM を再実行</translation>
        </message>
        <message>
            <source>Regenerate</source>
            <translation>再生成</translation>
        </message>
        <message>
            <source>Replace Alpha Hints?</source>
            <translation>アルファヒントを置き換えますか?</translation>
        </message>
        <message>
            <source>Replace Existing Alpha?</source>
            <translation>既存のアルファを置き換えますか?</translation>
        </message>
        <message>
            <source>Report Issue...</source>
            <translation>問題を報告...</translation>
        </message>
        <message>
            <source>Reset Layout</source>
            <translation>レイアウトをリセット</translation>
        </message>
        <message>
            <source>Reset Zoom</source>
            <translation>ズームをリセット</translation>
        </message>
        <message>
            <source>Resume</source>
            <translation>再開</translation>
        </message>
        <message>
            <source>Resume will skip completed frames.
Regenerate will redo all frames from scratch.</source>
            <translation>再開すると完了済みのフレームはスキップされます。
再生成するとすべてのフレームを最初からやり直します。</translation>
        </message>
        <message>
            <source>Return to Home</source>
            <translation>ホームに戻る</translation>
        </message>
        <message>
            <source>SAM2 preview on frame %d covers %.1f%% of the frame.

If this looks right, continue with full Track Mask.
If not, keep painting corrections on this frame and run Track Mask again.</source>
            <translation>フレーム %d の SAM2 プレビューはフレームの %.1f%% をカバーしています。

正しければ、完全なトラックマスクに進んでください。
正しくない場合は、このフレームで修正ストロークを描き続け、トラックマスクを再実行してください。</translation>
        </message>
        <message>
            <source>Save Session</source>
            <translation>セッションを保存</translation>
        </message>
        <message>
            <source>Scan Error</source>
            <translation>スキャンエラー</translation>
        </message>
        <message>
            <source>Select Alpha Hint Folder</source>
            <translation>アルファヒントフォルダーを選択</translation>
        </message>
        <message>
            <source>Select Alpha Hint Video</source>
            <translation>アルファヒント動画を選択</translation>
        </message>
        <message>
            <source>Select Clips Directory</source>
            <translation>クリップディレクトリを選択</translation>
        </message>
        <message>
            <source>Select Image Sequence Folder</source>
            <translation>画像シーケンスフォルダーを選択</translation>
        </message>
        <message>
            <source>Select Video Files</source>
            <translation>動画ファイルを選択</translation>
        </message>
        <message>
            <source>Select a clip first.</source>
            <translation>最初にクリップを選択してください。</translation>
        </message>
        <message>
            <source>Select which outputs to export as video:</source>
            <translation>動画として書き出す出力を選択してください:</translation>
        </message>
        <message>
            <source>Set Project Output Folder</source>
            <translation>プロジェクト出力フォルダーを設定</translation>
        </message>
        <message>
            <source>Set Project Output Folder...</source>
            <translation>プロジェクト出力フォルダーを設定...</translation>
        </message>
        <message>
            <source>Special Thanks</source>
            <translation>スペシャルサンクス</translation>
        </message>
        <message>
            <source>Stop requested — waiting for current GPU step. Press FORCE STOP to relaunch if it stays stuck.</source>
            <translation>停止を要求中 -- 現在の GPU ステップを待機しています。動かない場合は FORCE STOP を押して再起動してください。</translation>
        </message>
        <message>
            <source>The current GPU step has not returned to Python.

Force Stop will auto-save the session and relaunch the app to break the stuck job immediately.

Continue?</source>
            <translation>現在の GPU ステップが Python に戻っていません。

強制停止を実行すると、セッションを自動保存してアプリを再起動し、スタックしたジョブを直ちに中断します。

続行しますか?</translation>
        </message>
        <message>
            <source>The interface language will change after the current jobs finish and the app restarts.</source>
            <translation>インターフェースの言語は、現在のジョブが完了してアプリが再起動した後に変更されます。</translation>
        </message>
        <message>
            <source>The update could not be verified and was NOT installed.

%s

This may indicate a security issue. Please download the latest release manually from GitHub or Gumroad.</source>
            <translation>アップデートを検証できなかったため、インストールされませんでした。

%s

セキュリティの問題が考えられます。最新リリースを GitHub または Gumroad から手動でダウンロードしてください。</translation>
        </message>
        <message>
            <source>This Frame</source>
            <translation>このフレーム</translation>
        </message>
        <message>
            <source>This clip already has an AlphaHint (from GVM or a previous run).

Tracking a new mask sequence will replace that alpha hint.

Remove existing AlphaHint and proceed?</source>
            <translation>このクリップにはすでに AlphaHint があります (GVM または前回の実行から)。

新しいマスクシーケンスをトラッキングすると、そのアルファヒントが置き換えられます。

既存の AlphaHint を削除して続行しますか?</translation>
        </message>
        <message>
            <source>This clip uses a blue screen background.

The blue screen keying model (401 MB) is not installed. Without it, the green model will be used as a fallback.

Download the blue screen model now?</source>
            <translation>このクリップはブルースクリーン背景を使用しています。

ブルースクリーンキーイングモデル (401 MB) がインストールされていません。これなしでは、グリーンモデルがフォールバックとして使用されます。

ブルースクリーンモデルを今すぐダウンロードしますか?</translation>
        </message>
        <message>
            <source>This sequence is already in the project as "%s".</source>
            <translation>このシーケンスはすでに "%s" としてプロジェクトに含まれています。</translation>
        </message>
        <message>
            <source>This will download the latest version, replace the current app,
and relaunch automatically.

Your session will be saved. Continue?</source>
            <translation>最新バージョンをダウンロードし、現在のアプリを置き換えて、
自動的に再起動します。

セッションは保存されます。続行しますか?</translation>
        </message>
        <message>
            <source>This will save your session, close the app, and run the updater.
The app will relaunch automatically after updating.

Continue?</source>
            <translation>セッションを保存し、アプリを閉じてアップデーターを実行します。
アップデート後、アプリは自動的に再起動します。

続行しますか?</translation>
        </message>
        <message>
            <source>Toggle Queue Panel</source>
            <translation>キューパネルの表示切替</translation>
        </message>
        <message>
            <source>Track Mask First</source>
            <translation>先にトラックマスクを実行</translation>
        </message>
        <message>
            <source>Track Mask Preview</source>
            <translation>トラックマスクプレビュー</translation>
        </message>
        <message>
            <source>Track Mask complete for %s</source>
            <translation>%s のトラックマスクが完了しました</translation>
        </message>
        <message>
            <source>Track Paint Masks</source>
            <translation>ペイントマスクをトラック</translation>
        </message>
        <message>
            <source>Track preview ready. Refine paint strokes and run Track Mask again.</source>
            <translation>トラックプレビューが準備できました。ペイントストロークを調整して、トラックマスクを再実行してください。</translation>
        </message>
        <message>
            <source>Unreadable Video</source>
            <translation>読み取れない動画</translation>
        </message>
        <message>
            <source>Update</source>
            <translation>アップデート</translation>
        </message>
        <message>
            <source>Update Available</source>
            <translation>アップデートあり</translation>
        </message>
        <message>
            <source>Update Available (v%s)</source>
            <translation>アップデートあり (v%s)</translation>
        </message>
        <message>
            <source>Update EZ-CorridorKey</source>
            <translation>EZ-CorridorKey をアップデート</translation>
        </message>
        <message>
            <source>Update Failed</source>
            <translation>アップデート失敗</translation>
        </message>
        <message>
            <source>Update Verification Failed</source>
            <translation>アップデートの検証に失敗しました</translation>
        </message>
        <message>
            <source>Updating EZ-CorridorKey</source>
            <translation>EZ-CorridorKey をアップデート中</translation>
        </message>
        <message>
            <source>Verifying update signature...</source>
            <translation>アップデートの署名を検証中...</translation>
        </message>
        <message>
            <source>Video File</source>
            <translation>動画ファイル</translation>
        </message>
        <message>
            <source>Video exported:
%s</source>
            <translation>動画を書き出しました:
%s</translation>
        </message>
        <message>
            <source>VideoMaMa masks</source>
            <translation>VideoMaMa マスク</translation>
        </message>
        <message>
            <source>View</source>
            <translation>表示</translation>
        </message>
        <message>
            <source>What would you like to clear?</source>
            <translation>何をクリアしますか?</translation>
        </message>
        <message>
            <source>Workspace no longer exists:
%s</source>
            <translation>ワークスペースが存在しなくなりました:
%s</translation>
        </message>
        <message>
            <source>You dropped %d image file(s).
The source folder contains %d image(s) total.</source>
            <translation>%d 個の画像ファイルをドロップしました。
ソースフォルダーには合計 %d 個の画像があります。</translation>
        </message>
        <message>
            <source>alpha hints</source>
            <translation>アルファヒント</translation>
        </message>
        <message>
            <source>Alpha</source>
            <translation>アルファ</translation>
        </message>
        <message>
            <source>Extraction Failed</source>
            <translation>抽出に失敗しました</translation>
        </message>
        <message>
            <source>Frame extraction failed for '%s':

%s</source>
            <translation>'%s' のフレーム抽出に失敗しました:

%s</translation>
        </message>
    </context>
    <context>
        <name>ParameterPanel</name>
        <message>
            <location filename="../widgets/parameter_panel.py" line="132" />
            <source>ALPHA GENERATION</source>
            <translation>アルファ生成</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="137" />
            <source>Manual</source>
            <translation>手動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="142" />
            <source>CHROMA KEY</source>
            <translation>クロマキー</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="147" />
            <source>Generate alpha hints using a traditional chroma keyer.
Best for clean green/blue screen shots.
No GPU or AI model required — instant processing.

Click to expand parameters, then click GENERATE.
Hotkey: `</source>
            <translation>従来のクロマキーを使用してアルファヒントを生成します。
クリーンなグリーンバック/ブルーバック素材に最適です。
GPU や AI モデル不要で即時処理できます。

パラメーターを展開してから「生成」をクリックしてください。
ショートカット: `</translation>
        </message>
        <message>
            <source> Pick Screen Color</source>
            <translation type="vanished"> スクリーンカラーを取得</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="169" />
            <source>Click on the viewer to sample the screen color.
Works on either the input or output viewport.
Hotkey: E</source>
            <translation>ビューアをクリックしてスクリーンカラーをサンプリングします。
入力ビューアと出力ビューアのどちらでも使用できます。
ショートカット: E</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="181" />
            <source>Sampled screen color</source>
            <translation>サンプリング済みスクリーンカラー</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="186" />
            <source>Key Strength: 1.0</source>
            <translation>キーの強さ: 1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="191" />
            <source>How aggressively to key the screen color. Higher = more separation.</source>
            <translation>スクリーンカラーをどれだけ積極的にキーイングするかを設定します。値が高いほど分離が強くなります。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="193" />
            <source>Key Strength: %s</source>
            <translation>キーの強さ: %s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="199" />
            <source>Clip Black: 0.0</source>
            <translation>黒クリップ: 0.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="204" />
            <source>Push near-transparent values to fully transparent.
Cleans up noise in background areas.</source>
            <translation>透明に近い値を完全な透明に押し下げます。
背景部分のノイズを除去します。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="206" />
            <source>Clip Black: %s</source>
            <translation>黒クリップ: %s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="212" />
            <source>Clip White: 1.0</source>
            <translation>白クリップ: 1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="217" />
            <source>Push near-opaque values to fully opaque.
Solidifies the foreground core.</source>
            <translation>不透明に近い値を完全な不透明に押し上げます。
前景のコア部分を強化します。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="219" />
            <source>Clip White: %s</source>
            <translation>白クリップ: %s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="231" />
            <source>Shrink/Grow</source>
            <translation>収縮／拡張</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="236" />
            <source>Erode (negative) or dilate (positive) the matte edge.
0 = no change.</source>
            <translation>マットのエッジを収縮（マイナス）または拡張（プラス）します。
0 = 変更なし。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="240" />
            <source>Edge Blur</source>
            <translation>エッジのぼかし</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="245" />
            <source>Gaussian blur radius for softening matte edges.
0 = no blur.</source>
            <translation>マットのエッジを柔らかくするガウスぼかしの半径です。
0 = ぼかしなし。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="252" />
            <source>GENERATE</source>
            <translation>生成</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="253" />
            <source>Generate alpha hint frames for the entire clip using these chroma key settings.</source>
            <translation>これらのクロマキー設定を使用して、クリップ全体のアルファヒントフレームを生成します。</translation>
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
            <translation>Apple Vision（Neural Engine）を使用してアルファヒントを自動生成します。
前景の被写体を自動的に検出します。
macOS 14 以降専用です。Apple Neural Engine 上で動作し、高速で GPU 不要です。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="300" />
            <source>GVM AUTO</source>
            <translation>GVM自動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="304" />
            <source>Auto-generate alpha hint for the entire clip.
Uses GVM to predict foreground/background separation.
Available when clip is in RAW state (frames extracted).</source>
            <translation>クリップ全体のアルファヒントを自動生成します。
GVM を使用して前景と背景の分離を予測します。
クリップが RAW 状態（フレーム抽出済み）のときに使用できます。</translation>
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
            <translation>BiRefNet を使用してアルファヒントを自動生成します。
完全自動 — ペイントやアノテーション不要です。
初回使用時に選択したモデルバリエーションをダウンロードします。

Matting: 髪や透明度のディテールに最適（推奨）。
Portrait: 人物のクローズアップに最適化。
General: 前景と背景のバランス重視の分離。
HR バリエーション: 2K/4K フッテージ向け（VRAM 使用量増加）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="334" />
            <source>BiRefNet model variant — changes take effect on next run.</source>
            <translation>BiRefNet のモデルバリエーションです。変更は次回実行時に反映されます。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="355" />
            <source>Requires brushstrokes</source>
            <translation>ブラシストロークが必要です</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="360" />
            <source>Paint subject with 1, background with 2</source>
            <translation>被写体は 1、背景は 2 で塗ってください</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="366" />
            <source>TRACK MASK</source>
            <translation>マスクをトラック</translation>
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
            <translation>SAM2 を使用してペイントしたプロンプトを高密度マスクトラックに変換します。
MatAnyone2 または VideoMaMa を実行する前に必要です。

使い方:
1. 1 キーで緑のブラシを選択（前景 — 残す被写体）
2. 2 キーで赤のブラシを選択（背景 — 除去するエリア）
3. 左のビューアでフッテージ上にストロークを描きます
4. 「マスクをトラック」をクリックして、ペイントしたフレームで SAM2 をプレビューします
5. プレビューが正しければ確認し、全フレームに伝播します

ヒント:
Shift + 左ドラッグ（上下）: ブラシサイズを変更
Alt + 左ドラッグ: 2 点間に直線を描画
Ctrl+Z: 最後のストロークを元に戻す</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="600" />
            <source>Edge refinement strength (0.0-3.0).
Scales the CNN refiner's edge corrections.
1.0 = default, 0.0 = backbone only (no refinement),
higher = sharper edges but may introduce artifacts.</source>
            <translation>エッジ補正の強さ（0.0〜3.0）です。
CNN リファイナーのエッジ補正をスケーリングします。
1.0 = デフォルト、0.0 = バックボーンのみ（補正なし）、
高くするとエッジがシャープになりますがアーティファクトが発生する場合があります。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="391" />
            <source>MATANYONE2</source>
            <translation>MATANYONE2</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="166" />
            <source>💧 Pick Screen Color</source>
            <translation>💧 スクリーンカラーを取得</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="395" />
            <source>Generate alpha hints using MatAnyone2 video matting.
Requires paint strokes on the FIRST FRAME (frame 1).

1. Navigate to frame 1 (the very first frame)
2. Paint foreground (hotkey 1) and background (hotkey 2)
3. Click Track Mask to generate dense masks with SAM2
4. Click MATANYONE2 to generate temporally coherent AlphaHint</source>
            <translation>MatAnyone2 ビデオマッティングを使用してアルファヒントを生成します。
最初のフレーム（フレーム 1）へのブラシストロークが必要です。

1. フレーム 1（最初のフレーム）に移動します
2. 前景（ショートカット 1）と背景（ショートカット 2）を塗ります
3. 「マスクをトラック」をクリックして SAM2 で高密度マスクを生成します
4. MATANYONE2 をクリックして時間的に一貫した AlphaHint を生成します</translation>
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
            <translation>高密度な VideoMaMa マスクトラックからアルファヒントを生成します。

1. 前景/背景のプロンプトを簡単に塗ります
2. 「マスクをトラック」をクリックして SAM2 で高密度マスクを生成します
3. VIDEOMAMA をクリックして AlphaHint を生成します</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="433" />
            <source>Import your own mask for VideoMaMa.

Bypasses the Track Mask step. Select a folder or
video of grayscale masks and they will be used as
VideoMaMa's guidance input directly.</source>
            <translation>VideoMaMa 用の独自マスクを読み込みます。

「マスクをトラック」のステップをスキップします。グレースケールマスクの
フォルダーまたは動画を選択すると、VideoMaMa のガイダンス入力として
直接使用されます。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="462" />
            <source>IMPORT ALPHA</source>
            <translation>アルファを読み込み</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="466" />
            <source>Import alpha hints from an image folder or video file.
Supports: PNG/JPG/TIF/EXR sequences, or MOV/MP4/ProRes video.
White = foreground, black = background.
Files are copied into the clip's AlphaHint/ folder
and the clip advances to READY state for inference.</source>
            <translation>画像フォルダーまたは動画ファイルからアルファヒントを読み込みます。
対応形式: PNG/JPG/TIF/EXR 連番、または MOV/MP4/ProRes 動画。
白 = 前景、黒 = 背景。
ファイルはクリップの AlphaHint/ フォルダーにコピーされ、
クリップは推論の READY 状態に進みます。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="479" />
            <source>INFERENCE</source>
            <translation>推論</translation>
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
            <translation>このクリップの背景スクリーンカラーです。

自動: クリップの中間フレームから自動検出します。
グリーン: グリーンバック処理を強制します。
ブルー: ブルーバック処理を強制します。

使用するチェックポイント、スピル除去の計算、およびスピル検出を制御します。
また、UI のアクセントカラーも変更されます。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Auto</source>
            <translation>自動</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Green</source>
            <translation>グリーン</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="505" />
            <source>Blue</source>
            <translation>ブルー</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="511" />
            <source>Color Space</source>
            <translation>カラースペース</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="515" />
            <source>sRGB</source>
            <translation>sRGB</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="515" />
            <source>Linear</source>
            <translation>リニア</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="525" />
            <source>Removes small floating noise and speckles from the
alpha by discarding isolated regions smaller than the
size threshold.</source>
            <translation>サイズのしきい値より小さい孤立した領域を破棄することで、
アルファから小さな浮遊ノイズやスペックルを除去します。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="547" />
            <source>Garbage Matte</source>
            <translation>ガベージマット</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="551" />
            <source>Expands the alpha hint by N pixels, then zeros out
anything in the predicted matte that falls outside
that expanded region. Removes edge-of-frame artifacts
and background gunk that inference leaves behind.</source>
            <translation>アルファヒントを N ピクセル拡張し、拡張された領域の外側に
ある予測マットの部分をゼロにします。フレーム端のアーティファクトや
推論が残した背景のノイズを除去します。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="565" />
            <source>Pixel expansion around the alpha hint.
Higher = more breathing room around subject edges.
Lower = tighter crop to the hint boundary.</source>
            <translation>アルファヒント周囲のピクセル拡張量です。
高いほど被写体エッジ周辺の余裕が増えます。
低いほどヒント境界への切り取りがタイトになります。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="577" />
            <source>Despill: 0.5</source>
            <translation>スピル除去: 0.5</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="584" />
            <source>Screen spill removal strength (0.0-1.0).
Removes background color bleed from hair, skin, and edges.
1.0 = full despill, 0.0 = no despill (keep original colors).</source>
            <translation>スクリーンのスピル除去の強さ（0.0〜1.0）です。
髪、肌、エッジへの背景色のにじみを除去します。
1.0 = 完全除去、0.0 = 除去なし（元の色を維持）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="521" />
            <source>Despeckle</source>
            <translation>スペックル除去</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="538" />
            <source>Minimum area (in pixels) for a region to survive.
Isolated alpha blobs smaller than this are removed.
Lower = keep more detail, higher = cleaner matte.</source>
            <translation>領域が残るための最小面積（ピクセル単位）です。
これより小さい孤立したアルファの塊は除去されます。
低いほど詳細を保持し、高いほどマットがクリーンになります。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="593" />
            <source>Refiner: 1.0</source>
            <translation>リファイナー: 1.0</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="610" />
            <source>Live Preview</source>
            <translation>ライブプレビュー</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="618" />
            <source>OUTPUT</source>
            <translation>出力</translation>
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
            <translation>前景 — スピル除去済みの被写体を黒背景に配置します。
髪とエッジからスクリーンのスピルが除去されます。
ストレートアルファ（乗算済みではありません）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="637" />
            <location filename="../widgets/parameter_panel.py" line="656" />
            <source>EXR = 32-bit float (post-production).
PNG = 8-bit (general use).</source>
            <translation>EXR = 32 ビット浮動小数点（ポストプロダクション用）。
PNG = 8 ビット（汎用）。</translation>
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
            <translation>アルファマット — グレースケールの透明度マップです。
白 = 完全不透明、黒 = 完全透明。
合成ソフトで手動キーイングの制御に使用します。</translation>
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
            <translation>コンポジット — チェッカーボード背景に合成した最終的なキーイング結果です。
キーの品質を確認するのに最適です。
元の入力に忠実な色を再現します。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="675" />
            <source>PNG = 8-bit with transparency.
EXR = 32-bit float (post-production).</source>
            <translation>PNG = 8 ビット（透明度付き）。
EXR = 32 ビット浮動小数点（ポストプロダクション用）。</translation>
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
            <translation>Processed — 制作に使用できる RGBA（ストレート、リニア）です。
DaVinci Resolve、Premiere、および合成ツールへの読み込み向けに設計されています。
スピル除去とガベージマットのクリーンアップが適用されています。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="694" />
            <source>EXR = 32-bit float (recommended for Processed).
PNG = 8-bit (lossy for straight linear RGBA).</source>
            <translation>EXR = 32 ビット浮動小数点（Processed に推奨）。
PNG = 8 ビット（ストレートリニア RGBA では品質劣化あり）。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="701" />
            <source>PERFORMANCE</source>
            <translation>パフォーマンス</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="706" />
            <source>Parallel frames</source>
            <translation>並列フレーム数</translation>
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
            <translation>並列エンジンを使用して複数のフレームを同時に処理します。

追加のエンジンごとにモデルのコピーが 1 つ読み込まれます。
CUDA: エンジンあたり約 6〜8 GB の VRAM が必要です。

デフォルト: 1（最も安全）。まず 2 を試し、安定していれば増やしてください。

実験的機能: 8 より大きい値は高メモリの CUDA システム向けです
（例: RTX 6000）。
メモリが不足した場合、アプリは自動的に収まるエンジン数に縮小します。

現時点では CUDA のみ対応です。Apple Silicon では現在サポートされていません。</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="797" />
            <source>Despill: %s</source>
            <translation>スピル除去: %s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="802" />
            <source>Refiner: %s</source>
            <translation>リファイナー: %s</translation>
        </message>
        <message>
            <location filename="../widgets/parameter_panel.py" line="939" />
            <source>Painted: %d / %d frames</source>
            <translation>ペイント済み: %d / %d フレーム</translation>
        </message>
    </context>
    <context>
        <name>PreferencesDialog</name>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="156" />
            <source>Preferences</source>
            <translation>環境設定</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="189" />
            <source>User Interface</source>
            <translation>ユーザーインターフェイス</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="208" />
            <source>Select display language. Applies immediately.</source>
            <translation>表示言語を選択してください。すぐに適用されます。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="212" />
            <source>Show tooltips on controls</source>
            <translation>コントロールのツールチップを表示</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="218" />
            <source>UI sounds</source>
            <translation>UI サウンド</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="194" />
            <source>Language</source>
            <translation>言語</translation>
        </message>
        <message>
            <source>English</source>
            <translation type="vanished">日本語</translation>
        </message>
        <message>
            <source>Select display language. Restart required to apply.</source>
            <translation type="vanished">表示言語を選択してください。適用するには再起動が必要です。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="224" />
            <source>Show update notifications</source>
            <translation>アップデート通知を表示</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="227" />
            <source>When enabled, an Update Available button appears when a newer
release exists. Turn off to never check for or show updates.</source>
            <translation>有効にすると、新しいリリースが存在する場合に Update Available ボタンが表示されます。
オフにすると、アップデートの確認・表示を完全に無効化します。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="239" />
            <source>Project</source>
            <translation>プロジェクト</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="242" />
            <source>Copy source videos into project folder</source>
            <translation>ソース動画をプロジェクトフォルダーにコピー</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="245" />
            <source>When enabled, imported videos are copied into the project folder.
When disabled, the project references the original file in place.

Note: Deleting a project never touches the original source file.</source>
            <translation>有効にすると、読み込んだ動画がプロジェクトフォルダーにコピーされます。
無効にすると、プロジェクトは元のファイルを直接参照します。

注意: プロジェクトを削除しても、元のソースファイルは削除されません。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="255" />
            <source>Copy imported image sequences into project folder</source>
            <translation>読み込んだ連番画像をプロジェクトフォルダーにコピー</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="258" />
            <source>When enabled, imported image sequence files are copied into the project.
When disabled (default), the project references the original files in place.

Referencing saves disk space for large EXR/TIF sequences.
Original files are never modified regardless of this setting.</source>
            <translation>有効にすると、読み込んだ連番画像ファイルがプロジェクトにコピーされます。
無効（デフォルト）にすると、プロジェクトは元のファイルを直接参照します。

参照方式は大きな EXR/TIF 連番のディスク容量を節約できます。
この設定に関わらず、元のファイルは変更されません。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="272" />
            <source>Output</source>
            <translation>出力</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="275" />
            <source>EXR compression</source>
            <translation>EXR 圧縮</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="286" />
            <source>Compression used when writing EXR output files.

DWAB: Lossy wavelet, smallest files. Default.
PIZ: Lossless wavelet, preferred by compositors.
ZIP: Lossless deflate, good for clean renders.
None: No compression, fastest write, largest files.</source>
            <translation>EXR 出力ファイルへの書き込みに使用する圧縮方式です。

DWAB: 非可逆ウェーブレット、最小ファイルサイズ。デフォルト。
PIZ: 可逆ウェーブレット、コンポジター向け推奨。
ZIP: 可逆デフレート、クリーンなレンダリングに適しています。
なし: 圧縮なし、最高速書き込み、最大ファイルサイズ。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="296" />
            <source>Default output directory</source>
            <translation>デフォルトの出力フォルダー</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="302" />
            <source>Default (inside project)</source>
            <translation>デフォルト（プロジェクト内）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="308" />
            <source>Global default directory for inference output.

When set, outputs go to:
  &lt;this folder&gt;/&lt;ProjectName&gt;/&lt;ClipName&gt;/FG, Matte, etc.

Leave empty to use the default (Output/ inside each clip).
Per-clip overrides (right-click → Set Output Directory) take priority.</source>
            <translation>推論出力のグローバルなデフォルトフォルダーです。

設定すると、出力先は以下のようになります:
  &lt;このフォルダー&gt;/&lt;プロジェクト名&gt;/&lt;クリップ名&gt;/FG、Matte など。

空のままにすると各クリップのデフォルト（Output/ フォルダー）が使用されます。
クリップ別の上書き設定（右クリック → 出力フォルダーを設定）が優先されます。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="317" />
            <location filename="../widgets/preferences_dialog.py" line="501" />
            <source>Browse...</source>
            <translation>参照...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="322" />
            <source>Clear</source>
            <translation>消去</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="332" />
            <source>Inference</source>
            <translation>推論</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="335" />
            <source>Model resolution</source>
            <translation>モデル解像度</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="339" />
            <source>2048 — Full Quality</source>
            <translation>2048 — 最高品質</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="340" />
            <source>1024 — Faster, Less Detail</source>
            <translation>1024 — 高速・低ディテール</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="346" />
            <source>Resolution the model processes internally before upscaling to your frame size.
Applies to all backends (CUDA, MPS, MLX, CPU).

2048: Full quality — captures fine hair strands and edge detail.
Matches the original CorridorKey quality. Recommended for CUDA with 8GB+ VRAM.
WARNING: Very slow on Apple Silicon (needs 20GB+ memory).

1024: Faster inference with lower memory usage.
Fine hair detail may be lost. Recommended for Apple Silicon / low-VRAM GPUs.

Changing this requires an engine reload (happens automatically).</source>
            <translation>モデルがフレームサイズにアップスケールする前に内部で処理する解像度です。
すべてのバックエンド（CUDA、MPS、MLX、CPU）に適用されます。

2048: 最高品質 — 細かい髪の毛やエッジのディテールを捉えます。
元の CorridorKey の品質に匹敵します。8 GB 以上の VRAM を搭載した CUDA に推奨。
警告: Apple Silicon では非常に低速です（20 GB 以上のメモリが必要）。

1024: 低メモリで高速な推論を実現します。
細かい髪のディテールが失われる場合があります。Apple Silicon / 低 VRAM GPU に推奨。

この設定を変更するとエンジンの再読み込みが必要です（自動的に実行されます）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="361" />
            <source>Processing backend</source>
            <translation>処理バックエンド</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="365" />
            <source>Auto — MLX if available, otherwise MPS</source>
            <translation>自動 — MLX が利用可能な場合は MLX、それ以外は MPS</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="366" />
            <source>MLX — Apple Metal acceleration (recommended)</source>
            <translation>MLX — Apple Metal アクセラレーション（推奨）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="367" />
            <source>MPS — PyTorch Metal Performance Shaders</source>
            <translation>MPS — PyTorch Metal Performance Shaders</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="373" />
            <source>Choose the inference backend for Apple Silicon.

MLX: Native Apple Metal — fastest on M1/M2/M3/M4.
MPS: PyTorch Metal Performance Shaders — compatible fallback.
Auto: Uses MLX if installed, otherwise falls back to MPS.

Changing this requires an engine reload (happens automatically).</source>
            <translation>Apple Silicon の推論バックエンドを選択します。

MLX: ネイティブ Apple Metal — M1/M2/M3/M4 で最速。
MPS: PyTorch Metal Performance Shaders — 互換性のあるフォールバック。
自動: MLX がインストールされていれば使用し、それ以外は MPS にフォールバック。

この設定を変更するとエンジンの再読み込みが必要です（自動的に実行されます）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="385" />
            <source>Playback</source>
            <translation>再生</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="388" />
            <source>Loop playback within in/out range</source>
            <translation>イン/アウト範囲内でループ再生</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="391" />
            <source>When enabled, playback loops back to the in-point
after reaching the out-point (or start/end if no range).</source>
            <translation>有効にすると、アウト点に達した後（範囲未設定の場合は末尾に達した後）、
再生がイン点に戻ります。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="403" />
            <source>Tracking</source>
            <translation>トラッキング</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="406" />
            <source>SAM2 model</source>
            <translation>SAM2 モデル</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="412" />
            <source>%s  (%s)</source>
            <translation>%s  (%s)</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="417" />
            <source>Fast: lower VRAM, lower quality.
Base+: best default tradeoff for this app.
Highest Quality: slowest, heaviest tracker.</source>
            <translation>Fast: 低 VRAM、低品質。
Base+: このアプリのデフォルトとして最適なバランス。
Highest Quality: 最も低速で最重量のトラッカー。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="426" />
            <source>Models download automatically on first use. Download progress appears in the status bar.</source>
            <translation>モデルは初回使用時に自動的にダウンロードされます。ダウンロードの進捗はステータスバーに表示されます。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="434" />
            <source>Manage models</source>
            <translation>モデルを管理</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="446" />
            <source>Open Cache Folder</source>
            <translation>キャッシュフォルダーを開く</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="455" />
            <source>Video Tools</source>
            <translation>ビデオツール</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="458" />
            <source>FFmpeg status</source>
            <translation>FFmpeg のステータス</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="469" />
            <source>Windows: Repair downloads a bundled full FFmpeg build into tools/ffmpeg without changing your system install.
macOS: Repair installs FFmpeg via Homebrew.
Linux: Repair copies the install command to your clipboard.</source>
            <translation>Windows: 修復ではシステムの FFmpeg を変更せず、完全な FFmpeg ビルドを tools/ffmpeg にダウンロードします。
macOS: 修復では Homebrew を使用して FFmpeg をインストールします。
Linux: 修復ではインストールコマンドをクリップボードにコピーします。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="487" />
            <location filename="../widgets/preferences_dialog.py" line="772" />
            <location filename="../widgets/preferences_dialog.py" line="795" />
            <source>Repair FFmpeg</source>
            <translation>FFmpeg を修復</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="490" />
            <source>Windows: download and install a full bundled FFmpeg build into tools/ffmpeg, validate ffmpeg + ffprobe 7+, and switch CorridorKey to that local copy immediately.

macOS: install FFmpeg via Homebrew and validate ffmpeg + ffprobe 7+.

Linux: do not change system packages. CorridorKey shows the exact install commands and copies them to your clipboard instead.</source>
            <translation>Windows: 完全な FFmpeg ビルドを tools/ffmpeg にダウンロードしてインストールし、ffmpeg + ffprobe 7 以降を検証してから、CorridorKey をそのローカルコピーに即時切り替えます。

macOS: Homebrew 経由で FFmpeg をインストールし、ffmpeg + ffprobe 7 以降を検証します。

Linux: システムパッケージは変更しません。CorridorKey が正確なインストールコマンドを表示し、クリップボードにコピーします。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="504" />
            <source>Point CorridorKey at your own FFmpeg installation.
Select the folder containing ffmpeg.exe and ffprobe.exe.</source>
            <translation>CorridorKey が使用する独自の FFmpeg インストール先を指定します。
ffmpeg.exe と ffprobe.exe が入ったフォルダーを選択してください。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="511" />
            <source>Open FFmpeg Folder</source>
            <translation>FFmpeg フォルダーを開く</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="514" />
            <source>Open CorridorKey's bundled FFmpeg folder.
If Repair FFmpeg has been run on Windows, this is where the local full build is stored.</source>
            <translation>CorridorKey 同梱の FFmpeg フォルダーを開きます。
Windows で FFmpeg の修復を実行済みの場合、ローカルのフルビルドがここに保存されています。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="528" />
            <source>Privacy</source>
            <translation>プライバシー</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="532" />
            <source>Help improve EZ-CorridorKey: send crash reports automatically</source>
            <translation>EZ-CorridorKey の改善にご協力ください: クラッシュレポートを自動送信する</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="542" />
            <source>Off by default. When enabled, crash details, GPU/driver info, and the app version are sent when the app hits an error. Never your media, files, or personal info. Takes effect on the next launch.</source>
            <translation>デフォルトではオフです。有効にすると、エラー発生時にクラッシュの詳細、GPU/ドライバー情報、アプリのバージョンが送信されます。メディア、ファイル、個人情報は送信されません。次回起動時に反映されます。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="569" />
            <source>Cancel</source>
            <translation>キャンセル</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="573" />
            <source>OK</source>
            <translation>OK</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="653" />
            <source>Select Default Output Directory</source>
            <translation>デフォルトの出力フォルダーを選択</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="668" />
            <source>Select FFmpeg Folder (containing ffmpeg.exe and ffprobe.exe)</source>
            <translation>FFmpeg フォルダーを選択（ffmpeg.exe と ffprobe.exe が入ったフォルダー）</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="691" />
            <source>FFmpeg Not Found</source>
            <translation>FFmpeg が見つかりません</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="692" />
            <source>Could not find ffmpeg%s in:

%s

Select the folder that contains ffmpeg.exe and ffprobe.exe (usually the 'bin' folder inside the FFmpeg download).</source>
            <translation>以下の場所に ffmpeg%s が見つかりませんでした:

%s

ffmpeg.exe と ffprobe.exe が入ったフォルダーを選択してください（通常は FFmpeg ダウンロードの「bin」フォルダー内）。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="701" />
            <source>FFprobe Missing</source>
            <translation>FFprobe が見つかりません</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="702" />
            <source>Found ffmpeg%s but ffprobe%s is missing from:

%s

CorridorKey requires both. Download a full FFmpeg build.</source>
            <translation>ffmpeg%s は見つかりましたが、ffprobe%s が以下の場所に見つかりません:

%s

CorridorKey では両方が必要です。完全な FFmpeg ビルドをダウンロードしてください。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="715" />
            <source>FFmpeg Found</source>
            <translation>FFmpeg が見つかりました</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="719" />
            <source>FFmpeg Issue</source>
            <translation>FFmpeg の問題</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="761" />
            <source>FFmpeg OK</source>
            <translation>FFmpeg 正常</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="762" />
            <source>%s

No repair is needed.</source>
            <translation>%s

修復は必要ありません。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="774" />
            <source>

The install command has been copied to your clipboard.
Paste it into a terminal to install.</source>
            <translation>

インストールコマンドがクリップボードにコピーされました。
ターミナルに貼り付けてインストールしてください。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="782" />
            <source>CorridorKey will download and install a full bundled FFmpeg build into:

%s

This does not modify your system-wide FFmpeg.

Continue?</source>
            <translation>CorridorKey は完全な FFmpeg ビルドを以下にダウンロードしてインストールします:

%s

システム全体の FFmpeg は変更されません。

続行しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="788" />
            <source>CorridorKey will install FFmpeg via Homebrew:

    brew install ffmpeg

Continue?</source>
            <translation>CorridorKey は Homebrew 経由で FFmpeg をインストールします:

    brew install ffmpeg

続行しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="802" />
            <source>Preparing repair...</source>
            <translation>修復を準備中...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="805" />
            <source>Repairing FFmpeg...</source>
            <translation>FFmpeg を修復中...</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="839" />
            <source>FFmpeg Repaired</source>
            <translation>FFmpeg の修復完了</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="840" />
            <source>%s

CorridorKey will use FFmpeg immediately.</source>
            <translation>%s

CorridorKey はただちに FFmpeg を使用します。</translation>
        </message>
        <message>
            <location filename="../widgets/preferences_dialog.py" line="847" />
            <source>FFmpeg Repair Failed</source>
            <translation>FFmpeg の修復に失敗しました</translation>
        </message>
    </context>
    <context>
        <name>PreviewViewport</name>
        <message>
            <location filename="../widgets/preview_viewport.py" line="235" />
            <source>Extracting frames...
%s</source>
            <translation>フレームを抽出中...
%s</translation>
        </message>
        <message>
            <location filename="../widgets/preview_viewport.py" line="262" />
            <source>Selected: %s
State: %s</source>
            <translation>選択中: %s
状態: %s</translation>
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
            <translation>A/B ワイプ比較の切り替え（ショートカット: A）

入力（A）と現在の出力（B）を 1 つのビューアに重ね、
斜めの分割線で比較します。

中央のハンドルをドラッグして分割線を移動します。
ハンドルの上下をドラッグして角度を回転させます。
スクロールホイールで分割線を移動（Shift+スクロールで微調整）。
分割線を中ボタンクリックでデフォルトにリセットします。</translation>
        </message>
        <message>
            <location filename="../widgets/preview_viewport.py" line="562" />
            <source>No frame available for stem %d</source>
            <translation>ステム %d に利用可能なフレームがありません</translation>
        </message>
    </context>
    <context>
        <name>QueuePanel</name>
        <message>
            <location filename="../widgets/queue_panel.py" line="84" />
            <source>Toggle queue panel (Q)</source>
            <translation>キューパネルの表示切り替え (Q)</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="92" />
            <location filename="../widgets/queue_panel.py" line="121" />
            <source>QUEUE</source>
            <translation>キュー</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="136" />
            <source>Clear</source>
            <translation>消去</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="145" />
            <source>Clear completed and cancelled jobs</source>
            <translation>完了およびキャンセルされたジョブを消去します。</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="288" />
            <source>QUEUED</source>
            <translation>待機中</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="289" />
            <source>PROCESSING</source>
            <translation>処理中</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="290" />
            <source>DONE</source>
            <translation>完了</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="291" />
            <source>CANCELLED</source>
            <translation>キャンセル済</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="292" />
            <source>FAILED</source>
            <translation>失敗</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="334" />
            <source>Dismiss</source>
            <translation>閉じる</translation>
        </message>
        <message>
            <location filename="../widgets/queue_panel.py" line="413" />
            <source>Processing...</source>
            <translation>処理中...</translation>
        </message>
    </context>
    <context>
        <name>RecentProjectCard</name>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="57" />
            <source>Open in Finder</source>
            <translation>Finder で開く</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="57" />
            <source>Open in Explorer</source>
            <translation>エクスプローラーで開く</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="66" />
            <source>Remove project</source>
            <translation>プロジェクトを削除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="110" />
            <source>Rename Project</source>
            <translation>プロジェクト名を変更</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="116" />
            <source>Delete Project</source>
            <translation>プロジェクトを削除</translation>
        </message>
    </context>
    <context>
        <name>RecentProjectsPanel</name>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="139" />
            <source>RECENT PROJECTS</source>
            <translation>最近のプロジェクト</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="161" />
            <source>No recent projects</source>
            <translation>最近のプロジェクトはありません</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="211" />
            <source>Rename Project</source>
            <translation>プロジェクト名を変更</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="211" />
            <source>Project name:</source>
            <translation>プロジェクト名:</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="238" />
            <source>Remove Project</source>
            <translation>プロジェクトを削除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="239" />
            <source>Remove "%s" from recent projects?</source>
            <translation>「%s」を最近のプロジェクトから削除しますか？</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="241" />
            <source>Remove from List: hides it from recents (files stay on disk).
Delete from Disk: permanently deletes the project folder.</source>
            <translation>リストから削除: 最近のプロジェクトから非表示にします（ファイルはディスクに残ります）。
ディスクから削除: プロジェクトフォルダーを完全に削除します。</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="245" />
            <source>Remove from List</source>
            <translation>リストから削除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="246" />
            <source>Delete from Disk</source>
            <translation>ディスクから削除</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="260" />
            <source>Confirm Delete</source>
            <translation>削除の確認</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="261" />
            <source>Permanently delete this project folder?

%s</source>
            <translation>このプロジェクトフォルダーを完全に削除しますか？

%s</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="283" />
            <source>Delete Failed</source>
            <translation>削除に失敗しました</translation>
        </message>
        <message>
            <location filename="../widgets/recent_projects_panel.py" line="284" />
            <source>Could not delete project:
%s</source>
            <translation>プロジェクトを削除できませんでした:
%s</translation>
        </message>
    </context>
    <context>
        <name>ReportIssueDialog</name>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="120" />
            <source>Report Issue</source>
            <translation>問題を報告</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="133" />
            <source>Issue title:</source>
            <translation>問題のタイトル:</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="135" />
            <source>Brief summary of the problem</source>
            <translation>問題の簡単な概要</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="143" />
            <source>What happened?</source>
            <translation>何が起きましたか？</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="147" />
            <source>Describe what you were doing and what went wrong.
Steps to reproduce are very helpful.</source>
            <translation>何をしていたときに何が問題になったかを説明してください。
再現手順があると大変役立ちます。</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="159" />
            <source>System info (auto-collected, included in report)</source>
            <translation>システム情報（自動収集、レポートに含まれます）</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="174" />
            <source>This will open GitHub in your browser. A free GitHub account is required to submit issues. Your report is also copied to the clipboard in case you need to paste it after logging in.</source>
            <translation>ブラウザで GitHub が開きます。問題を送信するには無料の GitHub アカウントが必要です。ログイン後に貼り付けられるよう、レポートはクリップボードにもコピーされます。</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="185" />
            <source>Also send this report directly to the developer</source>
            <translation>このレポートを開発者にも直接送信する</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="190" />
            <source>Sends the report shown above: crash details, GPU/driver info, app version. Never your media, files, or personal info.</source>
            <translation>上記のレポートを送信します: クラッシュの詳細、GPU/ドライバー情報、アプリのバージョン。メディア、ファイル、個人情報は送信されません。</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="201" />
            <source>Cancel</source>
            <translation>キャンセル</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="205" />
            <source>Open GitHub</source>
            <translation>GitHub を開く</translation>
        </message>
        <message>
            <location filename="../widgets/report_issue_dialog.py" line="328" />
            <source>Bug Report</source>
            <translation>バグレポート</translation>
        </message>
    </context>
    <context>
        <name>SetupWizard</name>
        <message>
            <location filename="../widgets/setup_wizard.py" line="652" />
            <source>EZ-CorridorKey Setup</source>
            <translation>EZ-CorridorKey セットアップ</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="674" />
            <source>Select which models to download. The core CorridorKey model is required.
Optional models can be downloaded later from Edit → Download Manager.</source>
            <translation>ダウンロードするモデルを選択してください。コアの CorridorKey モデルは必須です。
オプションのモデルは後で「編集」→「ダウンロードマネージャー」からダウンロードできます。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="704" />
            <source>Browse...</source>
            <translation>参照...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="713" />
            <source>Default Location</source>
            <translation>デフォルトの場所</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="686" />
            <source>Data directory (models, projects, frame cache):</source>
            <translation>データフォルダー（モデル、プロジェクト、フレームキャッシュ）:</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="716" />
            <source>Reset the data directory to the platform default (in case you changed it and want to return).</source>
            <translation>データフォルダーをプラットフォームの既定値にリセットします（変更後に元に戻したい場合）。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="757" />
            <source>Create Desktop shortcut</source>
            <translation>デスクトップにショートカットを作成</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="774" />
            <source>Send anonymous crash reports to help troubleshooting</source>
            <translation>匿名のクラッシュレポートを送信してトラブルシューティングに協力する</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="780" />
            <source>Crash details, GPU/driver info, app version. Never your media, files, or personal info. Change anytime in Preferences &gt; Privacy.</source>
            <translation>クラッシュの詳細、GPU/ドライバー情報、アプリのバージョン。メディア、ファイル、個人情報は送信されません。設定 &gt; プライバシーでいつでも変更できます。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="803" />
            <source>Cancel &amp;&amp; Exit</source>
            <translation>キャンセルして終了</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="814" />
            <source>Download &amp;&amp; Install</source>
            <translation>ダウンロードしてインストール</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="860" />
            <source>Choose Install Location</source>
            <translation>インストール先を選択</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="874" />
            <source>Cancelling...</source>
            <translation>キャンセル中...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="911" />
            <source>Preparing downloads (0/%d)...</source>
            <translation>ダウンロードを準備中 (0/%d)...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="926" />
            <source>Downloading %d/%d: %s...</source>
            <translation>ダウンロード中 %d/%d: %s...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="952" />
            <source>All %d downloads complete!</source>
            <translation>%d 件のダウンロードがすべて完了しました！</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="956" />
            <source>Some downloads failed. You can retry from Edit → Download Manager.</source>
            <translation>一部のダウンロードに失敗しました。「編集」→「ダウンロードマネージャー」から再試行できます。</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="964" />
            <source>Continue</source>
            <translation>続行</translation>
        </message>
    </context>
    <context>
        <name>SplitViewWidget</name>
        <message>
            <location filename="../widgets/split_view.py" line="512" />
            <source>Extracting frames...</source>
            <translation>フレームを抽出中...</translation>
        </message>
        <message>
            <location filename="../widgets/split_view.py" line="539" />
            <source>%d%%  (%d/%d frames)</source>
            <translation>%d%%  (%d/%d フレーム)</translation>
        </message>
    </context>
    <context>
        <name>StartupDiagnosticDialog</name>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="166" />
            <source>Startup Diagnostics</source>
            <translation>起動時診断</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="176" />
            <source>EZ-CorridorKey detected issues with your environment that may prevent some features from working correctly.</source>
            <translation>EZ-CorridorKey が環境の問題を検出しました。一部の機能が正しく動作しない可能性があります。</translation>
        </message>
        <message>
            <location filename="../widgets/diagnostic_dialog.py" line="212" />
            <source>Continue Anyway</source>
            <translation>このまま続行</translation>
        </message>
    </context>
    <context>
        <name>StatusBar</name>
        <message>
            <location filename="../widgets/status_bar.py" line="88" />
            <source>Inference progress for the current job</source>
            <translation>現在のジョブの推論の進捗です。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="114" />
            <location filename="../widgets/status_bar.py" line="251" />
            <source>RUN INFERENCE</source>
            <translation>推論を実行</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="121" />
            <source>Run AI keying on the selected clip (Ctrl+R).
Requires a READY or COMPLETE clip with alpha hints.
Respects in/out range if set (I/O hotkeys).</source>
            <translation>選択したクリップで AI キーイングを実行します（Ctrl+R）。
アルファヒントがある READY または COMPLETE 状態のクリップが必要です。
イン/アウト範囲が設定されている場合はその範囲を尊重します（I/O ショートカット）。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="139" />
            <source>RESUME</source>
            <translation>再開</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="145" />
            <source>Resume inference — skip already-processed frames,
fill in remaining gaps across the full clip.</source>
            <translation>推論を再開します — 処理済みフレームをスキップし、
クリップ全体の残りのフレームを処理します。</translation>
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
            <translation>現在のジョブを停止します（Escape）。
処理済みのフレームはディスクに保持されます。</translation>
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
            <translation>現在の GPU ステップがブロックされています。
強制停止するとアプリが再起動し、スタックしたジョブを中断します。</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="235" />
            <source>RUN EXTRACTION</source>
            <translation>抽出を実行</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="239" />
            <source>RUN PIPELINE</source>
            <translation>パイプラインを実行</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="243" />
            <source>RUN %d CLIPS</source>
            <translation>%d クリップを実行</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="247" />
            <source>RUN SELECTED</source>
            <translation>選択分を実行</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="345" />
            <source>1 warning</source>
            <translation>1 件の警告</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="347" />
            <source>%d warnings</source>
            <translation>%d 件の警告</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="353" />
            <source>Latest:
%s

Click for all warnings</source>
            <translation>最新:
%s

クリックしてすべての警告を表示</translation>
        </message>
        <message>
            <location filename="../widgets/status_bar.py" line="406" />
            <source>Warnings (%d)</source>
            <translation>警告 (%d)</translation>
        </message>
    </context>
    <context>
        <name>ThumbnailCanvas</name>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="238" />
            <source>%d frames</source>
            <translation>%d フレーム</translation>
        </message>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="240" />
            <source>(video)</source>
            <translation>（動画）</translation>
        </message>
        <message>
            <location filename="../widgets/thumbnail_canvas.py" line="242" />
            <source>(imported)</source>
            <translation>（インポート済）</translation>
        </message>
    </context>
    <context>
        <name>ViewModeBar</name>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="85" />
            <source>Original input footage (unprocessed)

Hotkey: F1</source>
            <translation>元の入力フッテージ（未処理）

ショートカット: F1</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="87" />
            <source>Tracked mask — SAM2 segmentation output.
White = foreground, black = background.
This is the binary mask before MatAnyone2/VideoMaMa refinement.

Hotkey: F2</source>
            <translation>トラッキング済みマスク — SAM2 セグメンテーション出力です。
白 = 前景、黒 = 背景。
MatAnyone2/VideoMaMa 補正前のバイナリマスクです。

ショートカット: F2</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="93" />
            <source>Alpha hint — generated by GVM, VideoMaMa, or MatAnyone2.
White = foreground, black = background.
This is the pre-inference guide used by CorridorKey.

Hotkey: F3</source>
            <translation>アルファヒント — GVM、VideoMaMa、または MatAnyone2 によって生成されます。
白 = 前景、黒 = 背景。
CorridorKey が使用する推論前のガイドです。

ショートカット: F3</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="99" />
            <source>Foreground — subject with screen spill removed.
Colors may look shifted; this is the despilled intermediate.

Hotkey: F4</source>
            <translation>前景 — スクリーンのスピルが除去された被写体です。
色がずれて見える場合がありますが、これはスピル除去後の中間状態です。

ショートカット: F4</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="104" />
            <source>Alpha matte — white = opaque, black = transparent.
Shows the AI's confidence in foreground vs background.

Hotkey: F5</source>
            <translation>アルファマット — 白 = 不透明、黒 = 透明。
前景と背景に対する AI の信頼度を示します。

ショートカット: F5</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="109" />
            <source>Composite — final keyed result over checkerboard.
Best preview of key quality with faithful colors.

Hotkey: F6</source>
            <translation>コンポジット — チェッカーボード背景に合成した最終的なキーイング結果です。
忠実な色でキーの品質を確認するのに最適です。

ショートカット: F6</translation>
        </message>
        <message>
            <location filename="../widgets/view_mode_bar.py" line="114" />
            <source>Processed — production RGBA (straight, linear).
For Resolve, Premiere, and compositing tools.
Preview composites the stored image over black.
Final compositing should happen in your compositor of choice.

Hotkey: F7</source>
            <translation>Processed — 制作用 RGBA（ストレート、リニア）です。
DaVinci Resolve、Premiere、および合成ツール向けです。
プレビューでは保存された画像を黒背景に合成します。
最終的な合成はお使いのコンポジターで行ってください。

ショートカット: F7</translation>
        </message>
    </context>
    <context>
        <name>VolumeControl</name>
        <message>
            <location filename="../widgets/volume_control.py" line="32" />
            <source>Click to mute / unmute</source>
            <translation>クリックでミュート / ミュート解除</translation>
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
            <translation>メディアファイルを選択</translation>
        </message>
    </context>
    <context>
        <name>_DropZone</name>
        <message>
            <location filename="../widgets/welcome_screen.py" line="85" />
            <source>Drop Videos, Image Sequences, or Click to Import</source>
            <translation>動画や連番画像をドロップするか、クリックして読み込んでください</translation>
        </message>
        <message>
            <location filename="../widgets/welcome_screen.py" line="93" />
            <source>Browse...</source>
            <translation>参照...</translation>
        </message>
    </context>
    <context>
        <name>_ModelRow</name>
        <message>
            <location filename="../widgets/setup_wizard.py" line="603" />
            <source>  — Installed</source>
            <translation>  — インストール済み</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="625" />
            <source>Downloading...</source>
            <translation>ダウンロード中...</translation>
        </message>
        <message>
            <location filename="../widgets/setup_wizard.py" line="632" />
            <source>%d / %d MB</source>
            <translation>%d / %d MB</translation>
        </message>
    </context>
</TS>

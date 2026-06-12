<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ru_RU" sourcelanguage="en_US">
<context>
    <name>BatchPipelineDialog</name>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="69"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="96"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="500"/>
        <source>Batch Pipeline</source>
        <translation>Пакетная обработка</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="101"/>
        <source>Select a folder containing video clips. Files with &quot;alphahint&quot; or &quot;maskhint&quot; in the name are automatically paired as hints.</source>
        <translation>Выберите папку с видеоклипами. Файлы, содержащие «alphahint» или «maskhint» в имени, автоматически подбираются как подсказки.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="111"/>
        <source>Select Folder...</source>
        <translation>Выбрать папку…</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="115"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="462"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="501"/>
        <source>No folder selected</source>
        <translation>Папка не выбрана</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="122"/>
        <source>Global Settings</source>
        <translation>Общие параметры</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="127"/>
        <source>No-hint clips:</source>
        <translation>Клипы без подсказки:</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="130"/>
        <source>Alpha generation method for clips with no companion hint file.
GVM: fast automatic alpha.
BiRefNet: higher quality, select a model variant.</source>
        <translation>Метод генерации альфы для клипов без сопутствующего файла подсказки.
GVM: быстрая автоматическая альфа.
BiRefNet: выше качество, выберите вариант модели.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="157"/>
        <source>MaskHint clips:</source>
        <translation>Клипы с MaskHint:</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="160"/>
        <source>Mask refinement method for clips with a companion MaskHint file.
VideoMaMa: temporal consistency, best for video.
MatAnyone2: single-frame matting with mask guidance.</source>
        <translation>Метод уточнения маски для клипов с сопутствующим файлом MaskHint.
VideoMaMa: согласованность во времени, лучший выбор для видео.
MatAnyone2: однокадровый маттинг с опорой на маску.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="176"/>
        <source>Per-clip overrides</source>
        <translation>Переопределения по клипам</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Clip</source>
        <translation>Клип</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Detected</source>
        <translation>Обнаружено</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Pipeline</source>
        <translation>Пайплайн</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="183"/>
        <source>Status</source>
        <translation>Статус</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="204"/>
        <source>Clear Pipeline</source>
        <translation>Очистить пайплайн</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="205"/>
        <source>Cancel all pending batch jobs and reset.</source>
        <translation>Отменить все ожидающие пакетные задания и выполнить сброс.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="210"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="509"/>
        <source>Cancel</source>
        <translation>Отмена</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="213"/>
        <location filename="../widgets/batch_pipeline_dialog.py" line="508"/>
        <source>Run Batch</source>
        <translation>Запустить пакет</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="216"/>
        <source>Inference settings (despill, refiner, edge, color space, etc.) are inherited from the right panel. Adjust them there before running.</source>
        <translation>Параметры инференса (деспилл, рефайнер, края, цветовое пространство и т. д.) наследуются из правой панели. Настройте их там перед запуском.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="234"/>
        <source>Select Batch Folder</source>
        <translation>Выбор папки для пакетной обработки</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="267"/>
        <source>No hint</source>
        <translation>Без подсказки</translation>
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
        <translation>Инференс CK</translation>
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
        <translation>Найдено клипов: %d (%s)</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="356"/>
        <source>No video clips found in this folder.</source>
        <translation>В этой папке не найдено видеоклипов.</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="476"/>
        <source>Batch Pipeline - Processing</source>
        <translation>Пакетная обработка: выполняется</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="478"/>
        <source>Running...</source>
        <translation>Выполняется…</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="569"/>
        <source>Processing failed</source>
        <translation>Ошибка обработки</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="576"/>
        <source>Batch Pipeline - Complete</source>
        <translation>Пакетная обработка: завершено</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="577"/>
        <source>Done</source>
        <translation>Готово</translation>
    </message>
    <message>
        <location filename="../widgets/batch_pipeline_dialog.py" line="579"/>
        <source>Close</source>
        <translation>Закрыть</translation>
    </message>
</context>
<context>
    <name>DebugConsoleWidget</name>
    <message>
        <location filename="../widgets/debug_console.py" line="86"/>
        <source>Console</source>
        <translation>Консоль</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="129"/>
        <source>CONSOLE</source>
        <translation>КОНСОЛЬ</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="172"/>
        <source>Level:</source>
        <translation>Уровень:</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="178"/>
        <location filename="../widgets/debug_console.py" line="334"/>
        <source>Pause</source>
        <translation>Пауза</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="185"/>
        <source>Clear</source>
        <translation>Очистить</translation>
    </message>
    <message>
        <location filename="../widgets/debug_console.py" line="334"/>
        <source>Resume</source>
        <translation>Возобновить</translation>
    </message>
</context>
<context>
    <name>DiagnosticDialog</name>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="47"/>
        <source>Diagnostic: %s</source>
        <translation>Диагностика: %s</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="108"/>
        <source>Error: %s</source>
        <translation>Ошибка: %s</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="122"/>
        <source>Report Issue on GitHub</source>
        <translation>Сообщить о проблеме на GitHub</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="129"/>
        <source>OK</source>
        <translation>ОК</translation>
    </message>
</context>
<context>
    <name>FrameScrubber</name>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="52"/>
        <source>Go to first frame</source>
        <translation>Перейти к первому кадру</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="60"/>
        <source>Previous frame</source>
        <translation>Предыдущий кадр</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="68"/>
        <source>Play / Pause (Space)</source>
        <translation>Воспроизведение / пауза (Пробел)</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="82"/>
        <source>Coverage bar — shows which frames have been processed.
Green lane: painted frames (brush strokes).
White lane: alpha hint coverage.
Yellow lane: inference output coverage.</source>
        <translation>Шкала покрытия: показывает, какие кадры обработаны.
Зелёная дорожка: закрашенные кадры (мазки кисти).
Белая дорожка: покрытие альфа-подсказкой.
Жёлтая дорожка: покрытие выводом инференса.</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="95"/>
        <source>Scrub through frames. Scroll wheel or Left/Right to step.</source>
        <translation>Перемещайтесь по кадрам колесом мыши или стрелками влево/вправо.</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="118"/>
        <source>Next frame</source>
        <translation>Следующий кадр</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="126"/>
        <source>Go to last frame</source>
        <translation>Перейти к последнему кадру</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="310"/>
        <source>Pause (Space)</source>
        <translation>Пауза (Пробел)</translation>
    </message>
    <message>
        <location filename="../widgets/frame_scrubber.py" line="317"/>
        <source>Play (Space)</source>
        <translation>Воспроизведение (Пробел)</translation>
    </message>
</context>
<context>
    <name>HotkeysDialog</name>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="133"/>
        <source>Hotkeys</source>
        <translation>Горячие клавиши</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="145"/>
        <source>Filter shortcuts...</source>
        <translation>Фильтр сочетаний…</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="212"/>
        <source>Reset</source>
        <translation>Сброс</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="220"/>
        <source>Reset to default: %s</source>
        <translation>Сбросить по умолчанию: %s</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="240"/>
        <source>Reset All to Defaults</source>
        <translation>Сбросить всё по умолчанию</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="252"/>
        <source>Cancel</source>
        <translation>Отмена</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="263"/>
        <source>OK</source>
        <translation>ОК</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="309"/>
        <source>Reset All Shortcuts</source>
        <translation>Сброс всех сочетаний</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="310"/>
        <source>Reset all shortcuts to their default values?</source>
        <translation>Сбросить все сочетания клавиш к значениям по умолчанию?</translation>
    </message>
</context>
<context>
    <name>IOTrayActionsMixin</name>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="63"/>
        <source>Run Extraction (%d clips)</source>
        <translation>Извлечь кадры (клипов: %d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="64"/>
        <source>Run Extraction</source>
        <translation>Извлечь кадры</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="72"/>
        <source>Rename...</source>
        <translation>Переименовать…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="78"/>
        <source>Finder</source>
        <translation>Finder</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="78"/>
        <source>Explorer</source>
        <translation>Проводник</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="79"/>
        <source>Open in %s</source>
        <translation>Открыть в %s</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="89"/>
        <source>Clear Mask (%d clips)</source>
        <translation>Очистить маску (клипов: %d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="89"/>
        <location filename="../widgets/io_tray_actions.py" line="232"/>
        <source>Clear Mask</source>
        <translation>Очистить маску</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="97"/>
        <source>Clear Alpha (%d clips)</source>
        <translation>Очистить альфу (клипов: %d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="97"/>
        <location filename="../widgets/io_tray_actions.py" line="341"/>
        <source>Clear Alpha</source>
        <translation>Очистить альфу</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="105"/>
        <source>Clear Outputs (%d clips)</source>
        <translation>Очистить выходные файлы (клипов: %d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="105"/>
        <location filename="../widgets/io_tray_actions.py" line="373"/>
        <source>Clear Outputs</source>
        <translation>Очистить выходные файлы</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="113"/>
        <source>Clear All (%d clips)</source>
        <translation>Очистить всё (клипов: %d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="113"/>
        <location filename="../widgets/io_tray_actions.py" line="296"/>
        <source>Clear All</source>
        <translation>Очистить всё</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="120"/>
        <source>Set Output Directory...</source>
        <translation>Задать папку вывода…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="126"/>
        <source>Clear Output Directory Override</source>
        <translation>Сбросить переопределение папки вывода</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="133"/>
        <source>Remove (%d clips)...</source>
        <translation>Убрать (клипов: %d)…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="133"/>
        <source>Remove...</source>
        <translation>Убрать…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="157"/>
        <source>Export %s as Video...</source>
        <translation>Экспортировать %s как видео…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="169"/>
        <source>Open Containing Folder</source>
        <translation>Открыть содержащую папку</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="183"/>
        <source>Output Directory for &apos;%s&apos;</source>
        <translation>Папка вывода для «%s»</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="214"/>
        <source>Rename Clip</source>
        <translation>Переименовать клип</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="214"/>
        <source>New name:</source>
        <translation>Новое имя:</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="233"/>
        <source>Delete tracked masks for %d clip(s)?
%s

This will remove all SAM2 mask frames from disk.</source>
        <translation>Удалить отслеженные маски для клипов (%d)?
%s

Это действие удалит все кадры масок SAM2 с диска.</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="297"/>
        <source>Remove ALL generated data for %d clip(s)?
%s

This will delete masks, alpha hints, and all output frames.</source>
        <translation>Удалить ВСЕ сгенерированные данные для клипов (%d)?
%s

Это действие удалит маски, альфа-подсказки и все выходные кадры.</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="342"/>
        <source>Delete AlphaHint for %d clip(s)?
%s

This will remove all generated alpha hint frames from disk.</source>
        <translation>Удалить AlphaHint для клипов (%d)?
%s

Это действие удалит все сгенерированные кадры альфа-подсказки с диска.</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="374"/>
        <source>Remove all output files for %d clip(s)?
%s

This will delete FG, Matte, Comp, and Processed frames.</source>
        <translation>Удалить все выходные файлы для клипов (%d)?
%s

Это действие удалит кадры FG, Matte, Comp и Processed.</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="400"/>
        <source>Remove %d clip(s)?</source>
        <translation>Убрать клипы (%d)?</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="404"/>
        <source>
... and %d more</source>
        <translation>
... и ещё %d</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="409"/>
        <source>How would you like to remove %d clip(s)?</source>
        <translation>Как убрать клипы (%d)?</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="412"/>
        <source>Remove from List</source>
        <translation>Убрать из списка</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_actions.py" line="413"/>
        <source>Delete from Disk</source>
        <translation>Удалить с диска</translation>
    </message>
</context>
<context>
    <name>IOTrayPanel</name>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="81"/>
        <source>INPUT (0)</source>
        <translation>ВВОД (0)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="86"/>
        <source>RESET I/O</source>
        <translation>СБРОС I/O</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="88"/>
        <source>Clear in/out markers on all clips</source>
        <translation>Очистить метки входа/выхода на всех клипах</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="92"/>
        <source>+ ADD</source>
        <translation>+ ДОБАВИТЬ</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="94"/>
        <source>Import clips — choose a folder or video file(s)</source>
        <translation>Импортировать клипы: выберите папку или видеофайл(ы)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="122"/>
        <source>EXPORTS (0)</source>
        <translation>ЭКСПОРТ (0)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="169"/>
        <source>Import Folder...</source>
        <translation>Импорт папки…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="170"/>
        <source>Import Video(s)...</source>
        <translation>Импорт видео…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="171"/>
        <source>Import Image Sequence...</source>
        <translation>Импорт секвенции…</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="180"/>
        <source>No Markers</source>
        <translation>Нет меток</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="181"/>
        <source>No clips have in/out markers set.</source>
        <translation>Ни на одном клипе не заданы метки входа/выхода.</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="188"/>
        <source>Reset In/Out Markers</source>
        <translation>Сброс меток входа/выхода</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="189"/>
        <source>This will clear in/out markers on %d clip(s).

All clips will revert to full-clip processing.
Continue?</source>
        <translation>Будут очищены метки входа/выхода на клипах (%d).

Все клипы вернутся к обработке в полном объёме.
Продолжить?</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="199"/>
        <source>Confirm Reset</source>
        <translation>Подтверждение сброса</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="200"/>
        <source>Are you sure? This cannot be undone.

Clearing in/out markers on %d clip(s).</source>
        <translation>Уверены? Это действие нельзя отменить.

Будут очищены метки входа/выхода на клипах (%d).</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="212"/>
        <source>Select Clips Directory</source>
        <translation>Выбор папки с клипами</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="220"/>
        <source>Select Video Files</source>
        <translation>Выбор видеофайлов</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="228"/>
        <source>Select Image Sequence Folder</source>
        <translation>Выбор папки с секвенцией</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="359"/>
        <source>INPUT (%d)</source>
        <translation>ВВОД (%d)</translation>
    </message>
    <message>
        <location filename="../widgets/io_tray_panel.py" line="360"/>
        <source>EXPORTS (%d)</source>
        <translation>ЭКСПОРТ (%d)</translation>
    </message>
</context>
<context>
    <name>KeyBindButton</name>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="56"/>
        <source>(none)</source>
        <translation>(нет)</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="62"/>
        <source>Press a key...</source>
        <translation>Нажмите клавишу…</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="94"/>
        <source>Shortcut Conflict</source>
        <translation>Конфликт сочетаний клавиш</translation>
    </message>
    <message>
        <location filename="../widgets/hotkeys_dialog.py" line="95"/>
        <source>&quot;%s&quot; is already assigned to:
%s

Reassign anyway? The conflicting binding will be cleared.</source>
        <translation>«%s» уже назначено:
%s

Всё равно переназначить? Конфликтующее сочетание будет сброшено.</translation>
    </message>
</context>
<context>
    <name>MainWindow</name>
    <message>
        <location filename="../main_window.py" line="263"/>
        <source>%s — Mac Performance Warning</source>
        <translation>%s: предупреждение о производительности на Mac</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="265"/>
        <source>GPU-intensive features (SAM2, GVM, VideoMaMa, MatAnyone2) are very slow on Mac (Apple Silicon MPS).

This may take hours for longer clips and could freeze your system.

Recommendation: Import pre-made alpha mattes from After Effects, DaVinci Resolve, or Nuke instead.

Continue anyway? (This warning won&apos;t appear again this session.)</source>
        <translation>Функции с высокой нагрузкой на GPU (SAM2, GVM, VideoMaMa, MatAnyone2) работают очень медленно на Mac (Apple Silicon MPS).

Для длинных клипов обработка может занять часы и привести к зависанию системы.

Рекомендация: вместо этого импортируйте готовые альфа-маски из After Effects, DaVinci Resolve или Nuke.

Всё равно продолжить? (Это предупреждение больше не появится в текущем сеансе.)</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="283"/>
        <source>EZ-CorridorKey</source>
        <translation>EZ-CorridorKey</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="439"/>
        <source>Detected GPU used for inference</source>
        <translation>Обнаружен GPU для инференса</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="442"/>
        <source>VRAM</source>
        <translation>VRAM</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="453"/>
        <source>GPU video memory usage — updates during inference</source>
        <translation>Использование видеопамяти GPU: обновляется в процессе инференса</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="459"/>
        <source>Current VRAM used / total available</source>
        <translation>Использовано / доступно видеопамяти</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="651"/>
        <source>No GPU</source>
        <translation>GPU не обнаружен</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="658"/>
        <source>Memory</source>
        <translation>Память</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="659"/>
        <source>Unified memory usage — CPU and GPU share the same pool</source>
        <translation>Использование объединённой памяти: CPU и GPU используют общий пул</translation>
    </message>
    <message>
        <location filename="../main_window.py" line="660"/>
        <source>Current unified memory used / total available</source>
        <translation>Использовано / доступно объединённой памяти</translation>
    </message>
</context>
<context>
    <name>ParameterPanel</name>
    <message>
        <location filename="../widgets/parameter_panel.py" line="132"/>
        <source>ALPHA GENERATION</source>
        <translation>ГЕНЕРАЦИЯ АЛЬФЫ</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="137"/>
        <source>Manual</source>
        <translation>Вручную</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="142"/>
        <source>CHROMA KEY</source>
        <translation>ХРОМАКЕЙ</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="147"/>
        <source>Generate alpha hints using a traditional chroma keyer.
Best for clean green/blue screen shots.
No GPU or AI model required — instant processing.

Click to expand parameters, then click GENERATE.
Hotkey: `</source>
        <translation>Генерация альфа-подсказок с помощью традиционного хромакея.
Лучший выбор для чистых зелёного/синего экрана.
GPU и AI-модель не нужны: обработка мгновенная.

Нажмите, чтобы развернуть параметры, затем нажмите ГЕНЕРАЦИЯ.
Горячая клавиша: `</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="166"/>
        <source> Pick Screen Color</source>
        <translation> Взять цвет экрана</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="169"/>
        <source>Click on the viewer to sample the screen color.
Works on either the input or output viewport.
Hotkey: E</source>
        <translation>Нажмите на окно просмотра, чтобы взять пробу цвета экрана.
Работает в любом окне просмотра: входном или выходном.
Горячая клавиша: E</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="181"/>
        <source>Sampled screen color</source>
        <translation>Взятый цвет экрана</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="186"/>
        <source>Key Strength: 1.0</source>
        <translation>Сила кеинга: 1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="191"/>
        <source>How aggressively to key the screen color. Higher = more separation.</source>
        <translation>Насколько агрессивно убирать цвет экрана. Выше — больше разделение.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="193"/>
        <source>Key Strength: %s</source>
        <translation>Сила кеинга: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="199"/>
        <source>Clip Black: 0.0</source>
        <translation>Уровень чёрного: 0.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="204"/>
        <source>Push near-transparent values to fully transparent.
Cleans up noise in background areas.</source>
        <translation>Переводит близкие к прозрачным значения в полностью прозрачные.
Очищает шум в областях фона.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="206"/>
        <source>Clip Black: %s</source>
        <translation>Уровень чёрного: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="212"/>
        <source>Clip White: 1.0</source>
        <translation>Уровень белого: 1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="217"/>
        <source>Push near-opaque values to fully opaque.
Solidifies the foreground core.</source>
        <translation>Переводит близкие к непрозрачным значения в полностью непрозрачные.
Уплотняет ядро переднего плана.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="219"/>
        <source>Clip White: %s</source>
        <translation>Уровень белого: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="231"/>
        <source>Shrink/Grow</source>
        <translation>Сжатие/расширение</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="236"/>
        <source>Erode (negative) or dilate (positive) the matte edge.
0 = no change.</source>
        <translation>Сжать (отрицательные значения) или расширить (положительные) край маски.
0 = без изменений.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="240"/>
        <source>Edge Blur</source>
        <translation>Размытие краёв</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="245"/>
        <source>Gaussian blur radius for softening matte edges.
0 = no blur.</source>
        <translation>Радиус гауссова размытия для смягчения краёв маски.
0 = без размытия.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="252"/>
        <source>GENERATE</source>
        <translation>ГЕНЕРАЦИЯ</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="253"/>
        <source>Generate alpha hint frames for the entire clip using these chroma key settings.</source>
        <translation>Сгенерировать кадры альфа-подсказки для всего клипа с текущими параметрами хромакея.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="271"/>
        <source>Automatic</source>
        <translation>Автоматически</translation>
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
        <translation>Автоматическая генерация альфа-подсказки с помощью Apple Vision (Neural Engine).
Автоматически определяет объекты переднего плана.
Только macOS 14+. Работает на Apple Neural Engine (быстро, GPU не нужен).</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="300"/>
        <source>GVM AUTO</source>
        <translation>GVM АВТО</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="304"/>
        <source>Auto-generate alpha hint for the entire clip.
Uses GVM to predict foreground/background separation.
Available when clip is in RAW state (frames extracted).</source>
        <translation>Автоматическая генерация альфа-подсказки для всего клипа.
Использует GVM для предсказания разделения переднего/заднего плана.
Доступно, когда клип находится в состоянии RAW (кадры извлечены).</translation>
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
        <translation>Автоматическая генерация альфа-подсказки с помощью BiRefNet.
Полностью автоматически: закрашивание и разметка не нужны.
При первом использовании скачивает выбранный вариант модели.

Matting: лучший выбор для волос и деталей прозрачности (рекомендуется).
Portrait: оптимизирован для крупных планов людей.
General: сбалансированное разделение переднего/заднего плана.
HR-варианты: для исходников 2K/4K (требуют больше видеопамяти).</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="334"/>
        <source>BiRefNet model variant — changes take effect on next run.</source>
        <translation>Вариант модели BiRefNet: изменения вступят в силу при следующем запуске.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="355"/>
        <source>Requires brushstrokes</source>
        <translation>Требуются мазки кисти</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="360"/>
        <source>Paint subject with 1, background with 2</source>
        <translation>Закрасьте объект клавишей 1, фон — клавишей 2</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="366"/>
        <source>TRACK MASK</source>
        <translation>ТРЕКИНГ МАСКИ</translation>
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
        <translation>Использует SAM2 для преобразования нарисованных подсказок в плотный трек маски.
Необходимо перед запуском MatAnyone2 или VideoMaMa.

КАК ИСПОЛЬЗОВАТЬ:
1. Нажмите 1, чтобы выбрать ЗЕЛЁНУЮ кисть (передний план — объект для сохранения)
2. Нажмите 2, чтобы выбрать КРАСНУЮ кисть (фон — область для удаления)
3. Нанесите мазки на исходник в левом окне просмотра
4. Нажмите ТРЕКИНГ МАСКИ для предпросмотра SAM2 на закрашенном кадре
5. Если предпросмотр верный, подтвердите для распространения на все кадры

СОВЕТЫ:
Shift + перетаскивание вверх/вниз: изменить размер кисти
Alt + перетаскивание: нарисовать прямую линию между двумя точками
Ctrl+Z: отменить последний штрих</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="600"/>
        <source>Edge refinement strength (0.0-3.0).
Scales the CNN refiner&apos;s edge corrections.
1.0 = default, 0.0 = backbone only (no refinement),
higher = sharper edges but may introduce artifacts.</source>
        <translation>Сила уточнения краёв (0.0–3.0).
Масштабирует коррекцию краёв CNN-рефайнером.
1.0 = по умолчанию, 0.0 = только базовая сеть (без уточнения),
выше = более чёткие края, но возможны артефакты.</translation>
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
        <translation>Генерация альфа-подсказок с помощью MatAnyone2.
Требуются мазки кисти на ПЕРВОМ КАДРЕ (кадр 1).

1. Перейдите к кадру 1 (самый первый кадр)
2. Закрасьте передний план (клавиша 1) и фон (клавиша 2)
3. Нажмите «Трекинг маски» для генерации плотных масок с SAM2
4. Нажмите MATANYONE2 для генерации темпорально согласованного AlphaHint</translation>
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
        <translation>Генерация альфа-подсказок из плотного трека маски VideoMaMa.

1. Нанесите редкие подсказки переднего плана/фона
2. Нажмите «Трекинг маски» для генерации плотных масок с SAM2
3. Нажмите VIDEOMAMA для генерации AlphaHint</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="433"/>
        <source>Import your own mask for VideoMaMa.

Bypasses the Track Mask step. Select a folder or
video of grayscale masks and they will be used as
VideoMaMa&apos;s guidance input directly.</source>
        <translation>Импорт собственной маски для VideoMaMa.

Пропускает шаг «Трекинг маски». Выберите папку или
видео с полутоновыми масками — они будут использованы
непосредственно как управляющий ввод для VideoMaMa.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="462"/>
        <source>IMPORT ALPHA</source>
        <translation>ИМПОРТ АЛЬФЫ</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="466"/>
        <source>Import alpha hints from an image folder or video file.
Supports: PNG/JPG/TIF/EXR sequences, or MOV/MP4/ProRes video.
White = foreground, black = background.
Files are copied into the clip&apos;s AlphaHint/ folder
and the clip advances to READY state for inference.</source>
        <translation>Импорт альфа-подсказок из папки изображений или видеофайла.
Поддерживается: PNG/JPG/TIF/EXR секвенции или видео MOV/MP4/ProRes.
Белый = передний план, чёрный = фон.
Файлы копируются в папку AlphaHint/ клипа,
и клип переходит в состояние READY для инференса.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="479"/>
        <source>INFERENCE</source>
        <translation>ИНФЕРЕНС</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="492"/>
        <source>BG Color</source>
        <translation>Цвет фона</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="495"/>
        <source>Background screen color for this clip.

Auto: detected from the middle frame of the clip.
Green: force green screen processing.
Blue: force blue screen processing.

Controls which checkpoint, despill math, and spill
detection are used. Also changes the UI accent color.</source>
        <translation>Цвет фонового экрана для данного клипа.

Авто: определяется по среднему кадру клипа.
Зелёный: принудительно зелёный экран.
Синий: принудительно синий экран.

Определяет, какой чекпоинт, формула деспилла и
детектор спилла используются. Также меняет акцентный цвет интерфейса.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Auto</source>
        <translation>Авто</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Green</source>
        <translation>Зелёный</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="505"/>
        <source>Blue</source>
        <translation>Синий</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="511"/>
        <source>Color Space</source>
        <translation>Цветовое пространство</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="515"/>
        <source>sRGB</source>
        <translation>sRGB</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="515"/>
        <source>Linear</source>
        <translation>Linear</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="525"/>
        <source>Removes small floating noise and speckles from the
alpha by discarding isolated regions smaller than the
size threshold.</source>
        <translation>Удаляет мелкий плавающий шум и крап из альфы,
отбрасывая изолированные области меньше
заданного порогового размера.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="547"/>
        <source>Garbage Matte</source>
        <translation>Мусорная маска</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="551"/>
        <source>Expands the alpha hint by N pixels, then zeros out
anything in the predicted matte that falls outside
that expanded region. Removes edge-of-frame artifacts
and background gunk that inference leaves behind.</source>
        <translation>Расширяет альфа-подсказку на N пикселей, затем обнуляет
всё в предсказанной маске, что выходит за пределы
расширенной области. Удаляет артефакты у краёв кадра
и мусор фона, оставленный инференсом.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="565"/>
        <source>Pixel expansion around the alpha hint.
Higher = more breathing room around subject edges.
Lower = tighter crop to the hint boundary.</source>
        <translation>Расширение в пикселях вокруг альфа-подсказки.
Больше = больше пространства вокруг краёв объекта.
Меньше = более жёсткое обрезание по границе подсказки.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="577"/>
        <source>Despill: 0.5</source>
        <translation>Деспилл: 0.5</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="584"/>
        <source>Screen spill removal strength (0.0-1.0).
Removes background color bleed from hair, skin, and edges.
1.0 = full despill, 0.0 = no despill (keep original colors).</source>
        <translation>Сила удаления спилла от экрана (0.0–1.0).
Убирает рефлекс цвета фона с волос, кожи и краёв.
1.0 = полный деспилл, 0.0 = без деспилла (сохранить исходные цвета).</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="521"/>
        <source>Despeckle</source>
        <translation>Удаление шума</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="538"/>
        <source>Minimum area (in pixels) for a region to survive.
Isolated alpha blobs smaller than this are removed.
Lower = keep more detail, higher = cleaner matte.</source>
        <translation>Минимальная площадь (в пикселях) для сохранения области.
Изолированные пятна альфы меньше этого значения удаляются.
Меньше = больше деталей, больше = чище маска.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="593"/>
        <source>Refiner: 1.0</source>
        <translation>Рефайнер: 1.0</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="610"/>
        <source>Live Preview</source>
        <translation>Живой предпросмотр</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="618"/>
        <source>OUTPUT</source>
        <translation>ВЫВОД</translation>
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
        <translation>FG (передний план): объект с деспиллом на чёрном фоне.
Спилл от экрана удалён с волос и краёв.
Прямая альфа (без предумножения).</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="637"/>
        <location filename="../widgets/parameter_panel.py" line="656"/>
        <source>EXR = 32-bit float (post-production).
PNG = 8-bit (general use).</source>
        <translation>EXR = 32-битный float (постпродакшн).
PNG = 8-бит (общее использование).</translation>
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
        <translation>Matte (альфа-маска): полутоновая карта прозрачности.
Белый = полностью непрозрачный, чёрный = полностью прозрачный.
Используйте в программе композитинга для ручного управления кеингом.</translation>
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
        <translation>Comp (композит): финальный результат кеинга на шахматном фоне.
Наилучшее представление качества кея.
Цвета точно соответствуют исходному исходнику.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="675"/>
        <source>PNG = 8-bit with transparency.
EXR = 32-bit float (post-production).</source>
        <translation>PNG = 8-бит с прозрачностью.
EXR = 32-битный float (постпродакшн).</translation>
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
        <translation>Processed (обработанный RGBA): RGBA, готовый к продакшну (прямая альфа, линейный).
Предназначен для импорта в Resolve, Premiere и программы композитинга.
Включает применённый деспилл и очистку мусорной маской.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="694"/>
        <source>EXR = 32-bit float (recommended for Processed).
PNG = 8-bit (lossy for straight linear RGBA).</source>
        <translation>EXR = 32-битный float (рекомендуется для Processed).
PNG = 8-бит (с потерями для прямой линейной RGBA).</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="701"/>
        <source>PERFORMANCE</source>
        <translation>ПРОИЗВОДИТЕЛЬНОСТЬ</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="706"/>
        <source>Parallel frames</source>
        <translation>Параллельные кадры</translation>
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
        <translation>Обрабатывает несколько кадров одновременно с помощью параллельных движков.

Каждый дополнительный движок загружает полную копию модели.
CUDA: ~6–8 ГБ видеопамяти на движок.

По умолчанию: 1 (наиболее надёжно). Сначала попробуйте 2, затем увеличивайте при стабильной работе.

ЭКСПЕРИМЕНТАЛЬНО: значения выше 8 предназначены для CUDA-систем с большим объёмом памяти
(например, RTX 6000).
При нехватке памяти приложение автоматически уменьшит количество движков.

Только CUDA. На Apple Silicon не поддерживается.</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="797"/>
        <source>Despill: %s</source>
        <translation>Деспилл: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="802"/>
        <source>Refiner: %s</source>
        <translation>Рефайнер: %s</translation>
    </message>
    <message>
        <location filename="../widgets/parameter_panel.py" line="939"/>
        <source>Painted: %d / %d frames</source>
        <translation>Закрашено: %d / %d кадров</translation>
    </message>
</context>
<context>
    <name>PreferencesDialog</name>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="152"/>
        <source>Preferences</source>
        <translation>Настройки</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="185"/>
        <source>User Interface</source>
        <translation>Интерфейс</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="188"/>
        <source>Show tooltips on controls</source>
        <translation>Показывать всплывающие подсказки</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="194"/>
        <source>UI sounds</source>
        <translation>Звуки интерфейса</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="200"/>
        <source>Language</source>
        <translation>Язык</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="203"/>
        <source>English</source>
        <translation>English</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="209"/>
        <source>Select display language. Restart required to apply.</source>
        <translation>Выберите язык интерфейса. Для применения требуется перезапуск.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="216"/>
        <source>Project</source>
        <translation>Проект</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="219"/>
        <source>Copy source videos into project folder</source>
        <translation>Копировать исходные видео в папку проекта</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="222"/>
        <source>When enabled, imported videos are copied into the project folder.
When disabled, the project references the original file in place.

Note: Deleting a project never touches the original source file.</source>
        <translation>Если включено, импортированные видео копируются в папку проекта.
Если выключено, проект ссылается на исходный файл по месту расположения.

Примечание: удаление проекта никогда не затрагивает исходный файл.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="232"/>
        <source>Copy imported image sequences into project folder</source>
        <translation>Копировать импортированные секвенции в папку проекта</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="235"/>
        <source>When enabled, imported image sequence files are copied into the project.
When disabled (default), the project references the original files in place.

Referencing saves disk space for large EXR/TIF sequences.
Original files are never modified regardless of this setting.</source>
        <translation>Если включено, файлы импортированных секвенций копируются в проект.
Если выключено (по умолчанию), проект ссылается на исходные файлы по месту расположения.

Ссылка экономит место на диске для больших EXR/TIF секвенций.
Исходные файлы никогда не изменяются независимо от этой настройки.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="249"/>
        <source>Output</source>
        <translation>Вывод</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="252"/>
        <source>EXR compression</source>
        <translation>Сжатие EXR</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="263"/>
        <source>Compression used when writing EXR output files.

DWAB: Lossy wavelet, smallest files. Default.
PIZ: Lossless wavelet, preferred by compositors.
ZIP: Lossless deflate, good for clean renders.
None: No compression, fastest write, largest files.</source>
        <translation>Сжатие при записи выходных файлов EXR.

DWAB: вейвлет с потерями, наименьший размер файлов. По умолчанию.
PIZ: вейвлет без потерь, предпочтителен для композеров.
ZIP: deflate без потерь, хорошо для чистых рендеров.
None: без сжатия, запись быстрее всего, наибольший размер файлов.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="273"/>
        <source>Default output directory</source>
        <translation>Папка вывода по умолчанию</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="279"/>
        <source>Default (inside project)</source>
        <translation>По умолчанию (внутри проекта)</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="285"/>
        <source>Global default directory for inference output.

When set, outputs go to:
  &lt;this folder&gt;/&lt;ProjectName&gt;/&lt;ClipName&gt;/FG, Matte, etc.

Leave empty to use the default (Output/ inside each clip).
Per-clip overrides (right-click → Set Output Directory) take priority.</source>
        <translation>Глобальная папка вывода для результатов инференса.

Если задана, выходные файлы сохраняются в:
  &lt;эта папка&gt;/&lt;ИмяПроекта&gt;/&lt;ИмяКлипа&gt;/FG, Matte и т. д.

Оставьте пустым, чтобы использовать папку по умолчанию (Output/ внутри каждого клипа).
Переопределения по клипам (ПКМ → Задать папку вывода) имеют приоритет.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="294"/>
        <location filename="../widgets/preferences_dialog.py" line="478"/>
        <source>Browse...</source>
        <translation>Обзор…</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="299"/>
        <source>Clear</source>
        <translation>Очистить</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="309"/>
        <source>Inference</source>
        <translation>Инференс</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="312"/>
        <source>Model resolution</source>
        <translation>Разрешение модели</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="316"/>
        <source>2048 — Full Quality</source>
        <translation>2048: полное качество</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="317"/>
        <source>1024 — Faster, Less Detail</source>
        <translation>1024: быстрее, меньше деталей</translation>
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
        <translation>Разрешение, при котором модель обрабатывает кадр внутри, перед увеличением до размера вашего кадра.
Применяется ко всем бэкендам (CUDA, MPS, MLX, CPU).

2048: полное качество — передаёт тонкие пряди волос и детали краёв.
Соответствует исходному качеству CorridorKey. Рекомендуется для CUDA с 8+ ГБ видеопамяти.
ВНИМАНИЕ: очень медленно на Apple Silicon (требуется 20+ ГБ памяти).

1024: более быстрый инференс с меньшим потреблением памяти.
Мелкие детали волос могут быть потеряны. Рекомендуется для Apple Silicon / GPU с малым объёмом видеопамяти.

Изменение этого параметра требует перезагрузки движка (происходит автоматически).</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="338"/>
        <source>Processing backend</source>
        <translation>Бэкенд обработки</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="342"/>
        <source>Auto — MLX if available, otherwise MPS</source>
        <translation>Авто: MLX при наличии, иначе MPS</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="343"/>
        <source>MLX — Apple Metal acceleration (recommended)</source>
        <translation>MLX: ускорение Apple Metal (рекомендуется)</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="344"/>
        <source>MPS — PyTorch Metal Performance Shaders</source>
        <translation>MPS: PyTorch Metal Performance Shaders</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="350"/>
        <source>Choose the inference backend for Apple Silicon.

MLX: Native Apple Metal — fastest on M1/M2/M3/M4.
MPS: PyTorch Metal Performance Shaders — compatible fallback.
Auto: Uses MLX if installed, otherwise falls back to MPS.

Changing this requires an engine reload (happens automatically).</source>
        <translation>Выберите бэкенд инференса для Apple Silicon.

MLX: нативный Apple Metal — самый быстрый на M1/M2/M3/M4.
MPS: PyTorch Metal Performance Shaders — совместимый резервный вариант.
Авто: использует MLX при наличии, иначе переключается на MPS.

Изменение этого параметра требует перезагрузки движка (происходит автоматически).</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="362"/>
        <source>Playback</source>
        <translation>Воспроизведение</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="365"/>
        <source>Loop playback within in/out range</source>
        <translation>Зацикленное воспроизведение в диапазоне вход/выход</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="368"/>
        <source>When enabled, playback loops back to the in-point
after reaching the out-point (or start/end if no range).</source>
        <translation>Если включено, воспроизведение зацикливается на точке входа
после достижения точки выхода (или начала/конца при отсутствии диапазона).</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="380"/>
        <source>Tracking</source>
        <translation>Трекинг</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="383"/>
        <source>SAM2 model</source>
        <translation>Модель SAM2</translation>
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
        <translation>Fast: меньше видеопамяти, ниже качество.
Base+: лучший баланс по умолчанию для этого приложения.
Highest Quality: самый медленный, самый тяжёлый трекер.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="403"/>
        <source>Models download automatically on first use. Download progress appears in the status bar.</source>
        <translation>Модели загружаются автоматически при первом использовании. Прогресс загрузки отображается в строке состояния.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="411"/>
        <source>Manage models</source>
        <translation>Управление моделями</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="423"/>
        <source>Open Cache Folder</source>
        <translation>Открыть папку кэша</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="432"/>
        <source>Video Tools</source>
        <translation>Видеоинструменты</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="435"/>
        <source>FFmpeg status</source>
        <translation>Статус FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="446"/>
        <source>Windows: Repair downloads a bundled full FFmpeg build into tools/ffmpeg without changing your system install.
macOS: Repair installs FFmpeg via Homebrew.
Linux: Repair copies the install command to your clipboard.</source>
        <translation>Windows: восстановление скачивает встроенную сборку FFmpeg в tools/ffmpeg без изменения системной установки.
macOS: восстановление устанавливает FFmpeg через Homebrew.
Linux: восстановление копирует команду установки в буфер обмена.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="464"/>
        <location filename="../widgets/preferences_dialog.py" line="714"/>
        <location filename="../widgets/preferences_dialog.py" line="737"/>
        <source>Repair FFmpeg</source>
        <translation>Восстановить FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="467"/>
        <source>Windows: download and install a full bundled FFmpeg build into tools/ffmpeg, validate ffmpeg + ffprobe 7+, and switch CorridorKey to that local copy immediately.

macOS: install FFmpeg via Homebrew and validate ffmpeg + ffprobe 7+.

Linux: do not change system packages. CorridorKey shows the exact install commands and copies them to your clipboard instead.</source>
        <translation>Windows: скачать и установить встроенную сборку FFmpeg в tools/ffmpeg, проверить ffmpeg + ffprobe 7+, немедленно переключить CorridorKey на эту локальную копию.

macOS: установить FFmpeg через Homebrew и проверить ffmpeg + ffprobe 7+.

Linux: системные пакеты не изменяются. CorridorKey покажет точные команды установки и скопирует их в буфер обмена.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="481"/>
        <source>Point CorridorKey at your own FFmpeg installation.
Select the folder containing ffmpeg.exe and ffprobe.exe.</source>
        <translation>Укажите CorridorKey путь к вашей установке FFmpeg.
Выберите папку, содержащую ffmpeg.exe и ffprobe.exe.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="488"/>
        <source>Open FFmpeg Folder</source>
        <translation>Открыть папку FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="491"/>
        <source>Open CorridorKey&apos;s bundled FFmpeg folder.
If Repair FFmpeg has been run on Windows, this is where the local full build is stored.</source>
        <translation>Открыть папку со встроенным FFmpeg от CorridorKey.
Если на Windows выполнялось восстановление FFmpeg, здесь хранится локальная полная сборка.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="520"/>
        <source>Cancel</source>
        <translation>Отмена</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="524"/>
        <source>OK</source>
        <translation>ОК</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="595"/>
        <source>Select Default Output Directory</source>
        <translation>Выбор папки вывода по умолчанию</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="610"/>
        <source>Select FFmpeg Folder (containing ffmpeg.exe and ffprobe.exe)</source>
        <translation>Выбор папки FFmpeg (содержащей ffmpeg.exe и ffprobe.exe)</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="633"/>
        <source>FFmpeg Not Found</source>
        <translation>FFmpeg не найден</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="634"/>
        <source>Could not find ffmpeg%s in:

%s

Select the folder that contains ffmpeg.exe and ffprobe.exe (usually the &apos;bin&apos; folder inside the FFmpeg download).</source>
        <translation>Не удалось найти ffmpeg%s в:

%s

Выберите папку, содержащую ffmpeg.exe и ffprobe.exe (обычно папка «bin» внутри архива FFmpeg).</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="643"/>
        <source>FFprobe Missing</source>
        <translation>FFprobe отсутствует</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="644"/>
        <source>Found ffmpeg%s but ffprobe%s is missing from:

%s

CorridorKey requires both. Download a full FFmpeg build.</source>
        <translation>Найден ffmpeg%s, но ffprobe%s отсутствует в:

%s

CorridorKey требует оба. Скачайте полную сборку FFmpeg.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="657"/>
        <source>FFmpeg Found</source>
        <translation>FFmpeg найден</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="661"/>
        <source>FFmpeg Issue</source>
        <translation>Проблема с FFmpeg</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="703"/>
        <source>FFmpeg OK</source>
        <translation>FFmpeg в порядке</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="704"/>
        <source>%s

No repair is needed.</source>
        <translation>%s

Восстановление не требуется.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="716"/>
        <source>

The install command has been copied to your clipboard.
Paste it into a terminal to install.</source>
        <translation>
Команда установки скопирована в буфер обмена.
Вставьте её в терминал для установки.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="724"/>
        <source>CorridorKey will download and install a full bundled FFmpeg build into:

%s

This does not modify your system-wide FFmpeg.

Continue?</source>
        <translation>CorridorKey скачает и установит встроенную сборку FFmpeg в:

%s

Системная установка FFmpeg не изменится.

Продолжить?</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="730"/>
        <source>CorridorKey will install FFmpeg via Homebrew:

    brew install ffmpeg

Continue?</source>
        <translation>CorridorKey установит FFmpeg через Homebrew:

    brew install ffmpeg

Продолжить?</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="744"/>
        <source>Preparing repair...</source>
        <translation>Подготовка к восстановлению…</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="747"/>
        <source>Repairing FFmpeg...</source>
        <translation>Восстановление FFmpeg…</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="781"/>
        <source>FFmpeg Repaired</source>
        <translation>FFmpeg восстановлен</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="782"/>
        <source>%s

CorridorKey will use FFmpeg immediately.</source>
        <translation>%s

CorridorKey немедленно начнёт использовать FFmpeg.</translation>
    </message>
    <message>
        <location filename="../widgets/preferences_dialog.py" line="789"/>
        <source>FFmpeg Repair Failed</source>
        <translation>Не удалось восстановить FFmpeg</translation>
    </message>
</context>
<context>
    <name>PreviewViewport</name>
    <message>
        <location filename="../widgets/preview_viewport.py" line="235"/>
        <source>Extracting frames...
%s</source>
        <translation>Извлечение кадров…
%s</translation>
    </message>
    <message>
        <location filename="../widgets/preview_viewport.py" line="261"/>
        <source>Selected: %s
State: %s</source>
        <translation>Выбран: %s
Состояние: %s</translation>
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
        <translation>Переключить A/B-сравнение со шторкой (горячая клавиша: A)

Накладывает вход (A) и текущий вывод (B) в одном окне просмотра
с диагональной разделительной линией.

Перетащите центральный маркер для смещения линии.
Перетащите выше или ниже маркера для поворота угла.
Колесо мыши для смещения линии (Shift+колесо для точной настройки).
Щелчок средней кнопкой по линии для сброса к значению по умолчанию.</translation>
    </message>
    <message>
        <location filename="../widgets/preview_viewport.py" line="560"/>
        <source>No frame available for stem %d</source>
        <translation>Кадр для вывода %d недоступен</translation>
    </message>
</context>
<context>
    <name>QueuePanel</name>
    <message>
        <location filename="../widgets/queue_panel.py" line="101"/>
        <source>Toggle queue panel (Q)</source>
        <translation>Показать/скрыть панель очереди (Q)</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="138"/>
        <source>QUEUE</source>
        <translation>ОЧЕРЕДЬ</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="153"/>
        <source>Clear</source>
        <translation>Очистить</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="162"/>
        <source>Clear completed and cancelled jobs</source>
        <translation>Очистить завершённые и отменённые задания</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="336"/>
        <source>Dismiss</source>
        <translation>Скрыть</translation>
    </message>
    <message>
        <location filename="../widgets/queue_panel.py" line="408"/>
        <source>Processing...</source>
        <translation>Обработка…</translation>
    </message>
</context>
<context>
    <name>RecentProjectCard</name>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="57"/>
        <source>Open in Finder</source>
        <translation>Открыть в Finder</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="57"/>
        <source>Open in Explorer</source>
        <translation>Открыть в Проводнике</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="66"/>
        <source>Remove project</source>
        <translation>Убрать проект</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="110"/>
        <source>Rename Project</source>
        <translation>Переименовать проект</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="116"/>
        <source>Delete Project</source>
        <translation>Удалить проект</translation>
    </message>
</context>
<context>
    <name>RecentProjectsPanel</name>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="139"/>
        <source>RECENT PROJECTS</source>
        <translation>НЕДАВНИЕ ПРОЕКТЫ</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="161"/>
        <source>No recent projects</source>
        <translation>Нет недавних проектов</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="211"/>
        <source>Rename Project</source>
        <translation>Переименовать проект</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="211"/>
        <source>Project name:</source>
        <translation>Имя проекта:</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="238"/>
        <source>Remove Project</source>
        <translation>Убрать проект</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="239"/>
        <source>Remove &quot;%s&quot; from recent projects?</source>
        <translation>Убрать «%s» из недавних проектов?</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="241"/>
        <source>Remove from List: hides it from recents (files stay on disk).
Delete from Disk: permanently deletes the project folder.</source>
        <translation>Убрать из списка: скрывает из недавних (файлы остаются на диске).
Удалить с диска: безвозвратно удаляет папку проекта.</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="245"/>
        <source>Remove from List</source>
        <translation>Убрать из списка</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="246"/>
        <source>Delete from Disk</source>
        <translation>Удалить с диска</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="260"/>
        <source>Confirm Delete</source>
        <translation>Подтверждение удаления</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="261"/>
        <source>Permanently delete this project folder?

%s</source>
        <translation>Безвозвратно удалить папку этого проекта?

%s</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="283"/>
        <source>Delete Failed</source>
        <translation>Удаление не выполнено</translation>
    </message>
    <message>
        <location filename="../widgets/recent_projects_panel.py" line="284"/>
        <source>Could not delete project:
%s</source>
        <translation>Не удалось удалить проект:
%s</translation>
    </message>
</context>
<context>
    <name>ReportIssueDialog</name>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="118"/>
        <source>Report Issue</source>
        <translation>Сообщить о проблеме</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="130"/>
        <source>Issue title:</source>
        <translation>Заголовок проблемы:</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="132"/>
        <source>Brief summary of the problem</source>
        <translation>Краткое описание проблемы</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="140"/>
        <source>What happened?</source>
        <translation>Что произошло?</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="144"/>
        <source>Describe what you were doing and what went wrong.
Steps to reproduce are very helpful.</source>
        <translation>Опишите, что вы делали и что пошло не так.
Шаги для воспроизведения будут очень полезны.</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="156"/>
        <source>System info (auto-collected, included in report)</source>
        <translation>Сведения о системе (собираются автоматически, включаются в отчёт)</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="171"/>
        <source>This will open GitHub in your browser. A free GitHub account is required to submit issues. Your report is also copied to the clipboard in case you need to paste it after logging in.</source>
        <translation>Откроется GitHub в вашем браузере. Для отправки проблемы необходима бесплатная учётная запись GitHub. Отчёт также копируется в буфер обмена на случай, если вам нужно вставить его после входа в систему.</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="184"/>
        <source>Cancel</source>
        <translation>Отмена</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="188"/>
        <source>Open GitHub</source>
        <translation>Открыть GitHub</translation>
    </message>
    <message>
        <location filename="../widgets/report_issue_dialog.py" line="293"/>
        <source>Bug Report</source>
        <translation>Отчёт об ошибке</translation>
    </message>
</context>
<context>
    <name>SetupWizard</name>
    <message>
        <location filename="../widgets/setup_wizard.py" line="652"/>
        <source>EZ-CorridorKey Setup</source>
        <translation>Установка EZ-CorridorKey</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="674"/>
        <source>Select which models to download. The core CorridorKey model is required.
Optional models can be downloaded later from Edit → Download Manager.</source>
        <translation>Выберите модели для загрузки. Основная модель CorridorKey обязательна.
Дополнительные модели можно скачать позже через Правка → Менеджер загрузок.</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="704"/>
        <source>Browse...</source>
        <translation>Обзор…</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="713"/>
        <source>Default Location</source>
        <translation>Расположение по умолчанию</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="686"/>
        <source>Data directory (models, projects, frame cache):</source>
        <translation>Папка данных (модели, проекты, кэш кадров):</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="716"/>
        <source>Reset the data directory to the platform default (in case you changed it and want to return).</source>
        <translation>Сбросить папку данных к значению по умолчанию для платформы (если вы изменили её и хотите вернуть).</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="757"/>
        <source>Create Desktop shortcut</source>
        <translation>Создать ярлык на рабочем столе</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="775"/>
        <source>Cancel &amp;&amp; Exit</source>
        <translation>Отмена и выход</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="786"/>
        <source>Download &amp;&amp; Install</source>
        <translation>Скачать и установить</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="832"/>
        <source>Choose Install Location</source>
        <translation>Выбор расположения для установки</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="846"/>
        <source>Cancelling...</source>
        <translation>Отмена…</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="872"/>
        <source>Preparing downloads (0/%d)...</source>
        <translation>Подготовка загрузок (0/%d)…</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="887"/>
        <source>Downloading %d/%d: %s...</source>
        <translation>Скачивание %d/%d: %s…</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="913"/>
        <source>All %d downloads complete!</source>
        <translation>Загрузок завершено: %d!</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="917"/>
        <source>Some downloads failed. You can retry from Edit → Download Manager.</source>
        <translation>Некоторые загрузки завершились с ошибкой. Повторите попытку через Правка → Менеджер загрузок.</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="925"/>
        <source>Continue</source>
        <translation>Продолжить</translation>
    </message>
</context>
<context>
    <name>SplitViewWidget</name>
    <message>
        <location filename="../widgets/split_view.py" line="512"/>
        <source>Extracting frames...</source>
        <translation>Извлечение кадров…</translation>
    </message>
    <message>
        <location filename="../widgets/split_view.py" line="539"/>
        <source>%d%%  (%d/%d frames)</source>
        <translation>%d%%  (%d/%d кадров)</translation>
    </message>
</context>
<context>
    <name>StartupDiagnosticDialog</name>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="159"/>
        <source>Startup Diagnostics</source>
        <translation>Диагностика при запуске</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="169"/>
        <source>EZ-CorridorKey detected issues with your environment that may prevent some features from working correctly.</source>
        <translation>EZ-CorridorKey обнаружил проблемы в вашей среде, которые могут препятствовать корректной работе некоторых функций.</translation>
    </message>
    <message>
        <location filename="../widgets/diagnostic_dialog.py" line="197"/>
        <source>Continue Anyway</source>
        <translation>Всё равно продолжить</translation>
    </message>
</context>
<context>
    <name>StatusBar</name>
    <message>
        <location filename="../widgets/status_bar.py" line="88"/>
        <source>Inference progress for the current job</source>
        <translation>Прогресс инференса для текущего задания</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="114"/>
        <location filename="../widgets/status_bar.py" line="251"/>
        <source>RUN INFERENCE</source>
        <translation>ЗАПУСК ИНФЕРЕНСА</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="121"/>
        <source>Run AI keying on the selected clip (Ctrl+R).
Requires a READY or COMPLETE clip with alpha hints.
Respects in/out range if set (I/O hotkeys).</source>
        <translation>Запустить AI-кеинг для выбранного клипа (Ctrl+R).
Требуется клип в состоянии READY или COMPLETE с альфа-подсказками.
Учитывает диапазон вход/выход, если задан (горячие клавиши I/O).</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="139"/>
        <source>RESUME</source>
        <translation>ВОЗОБНОВИТЬ</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="145"/>
        <source>Resume inference — skip already-processed frames,
fill in remaining gaps across the full clip.</source>
        <translation>Возобновить инференс: пропустить уже обработанные кадры,
заполнить оставшиеся пропуски по всему клипу.</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="154"/>
        <location filename="../widgets/status_bar.py" line="203"/>
        <source>STOP</source>
        <translation>СТОП</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="158"/>
        <location filename="../widgets/status_bar.py" line="207"/>
        <source>Stop the current job (Escape).
Already-processed frames are kept on disk.</source>
        <translation>Остановить текущее задание (Esc).
Уже обработанные кадры сохраняются на диске.</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="194"/>
        <source>FORCE STOP</source>
        <translation>ПРИНУДИТЕЛЬНЫЙ СТОП</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="198"/>
        <source>The current GPU step is blocked.
Force Stop will relaunch the app to break the stuck job.</source>
        <translation>Текущий шаг GPU заблокирован.
Принудительный стоп перезапустит приложение для выхода из зависшего задания.</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="235"/>
        <source>RUN EXTRACTION</source>
        <translation>ИЗВЛЕЧЬ КАДРЫ</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="239"/>
        <source>RUN PIPELINE</source>
        <translation>ЗАПУСТИТЬ ПАЙПЛАЙН</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="243"/>
        <source>RUN %d CLIPS</source>
        <translation>ЗАПУСТИТЬ КЛИПЫ: %d</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="247"/>
        <source>RUN SELECTED</source>
        <translation>ЗАПУСТИТЬ ВЫБРАННЫЕ</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="345"/>
        <source>1 warning</source>
        <translation>Предупреждений: 1</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="347"/>
        <source>%d warnings</source>
        <translation>Предупреждений: %d</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="353"/>
        <source>Latest:
%s

Click for all warnings</source>
        <translation>Последнее:
%s

Нажмите для просмотра всех предупреждений</translation>
    </message>
    <message>
        <location filename="../widgets/status_bar.py" line="406"/>
        <source>Warnings (%d)</source>
        <translation>Предупреждения (%d)</translation>
    </message>
</context>
<context>
    <name>ViewModeBar</name>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="85"/>
        <source>Original input footage (unprocessed)

Hotkey: F1</source>
        <translation>Исходный футаж (без обработки)

Горячая клавиша: F1</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="87"/>
        <source>Tracked mask — SAM2 segmentation output.
White = foreground, black = background.
This is the binary mask before MatAnyone2/VideoMaMa refinement.

Hotkey: F2</source>
        <translation>Отслеженная маска: результат сегментации SAM2.
Белый = передний план, чёрный = фон.
Это бинарная маска до уточнения MatAnyone2/VideoMaMa.

Горячая клавиша: F2</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="93"/>
        <source>Alpha hint — generated by GVM, VideoMaMa, or MatAnyone2.
White = foreground, black = background.
This is the pre-inference guide used by CorridorKey.

Hotkey: F3</source>
        <translation>Альфа-подсказка: сгенерирована GVM, VideoMaMa или MatAnyone2.
Белый = передний план, чёрный = фон.
Это предварительный ориентир для CorridorKey перед инференсом.

Горячая клавиша: F3</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="99"/>
        <source>Foreground — subject with screen spill removed.
Colors may look shifted; this is the despilled intermediate.

Hotkey: F4</source>
        <translation>FG (передний план): объект с удалённым спиллом от экрана.
Цвета могут выглядеть смещёнными — это промежуточный результат деспилла.

Горячая клавиша: F4</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="104"/>
        <source>Alpha matte — white = opaque, black = transparent.
Shows the AI&apos;s confidence in foreground vs background.

Hotkey: F5</source>
        <translation>Альфа-маска: белый = непрозрачный, чёрный = прозрачный.
Показывает уверенность AI в разделении переднего плана и фона.

Горячая клавиша: F5</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="109"/>
        <source>Composite — final keyed result over checkerboard.
Best preview of key quality with faithful colors.

Hotkey: F6</source>
        <translation>Comp (композит): финальный результат кеинга на шахматном фоне.
Лучший предпросмотр качества кея с точными цветами.

Горячая клавиша: F6</translation>
    </message>
    <message>
        <location filename="../widgets/view_mode_bar.py" line="114"/>
        <source>Processed — production RGBA (straight, linear).
For Resolve, Premiere, and compositing tools.
Preview composites the stored image over black.
Final compositing should happen in your compositor of choice.

Hotkey: F7</source>
        <translation>Processed (обработанный RGBA): продакшн RGBA (прямая альфа, линейный).
Для Resolve, Premiere и программ композитинга.
Предпросмотр накладывает сохранённое изображение на чёрный фон.
Финальный композитинг следует выполнять в вашей программе.

Горячая клавиша: F7</translation>
    </message>
</context>
<context>
    <name>VolumeControl</name>
    <message>
        <location filename="../widgets/volume_control.py" line="32"/>
        <source>Click to mute / unmute</source>
        <translation>Нажмите, чтобы выключить или включить звук</translation>
    </message>
    <message>
        <location filename="../widgets/volume_control.py" line="46"/>
        <source>Volume</source>
        <translation>Громкость</translation>
    </message>
</context>
<context>
    <name>WelcomeScreen</name>
    <message>
        <location filename="../widgets/welcome_screen.py" line="175"/>
        <source>Select Media Files</source>
        <translation>Выбор медиафайлов</translation>
    </message>
</context>
<context>
    <name>_DropZone</name>
    <message>
        <location filename="../widgets/welcome_screen.py" line="85"/>
        <source>Drop Videos, Image Sequences, or Click to Import</source>
        <translation>Перетащите видео или секвенции сюда либо нажмите, чтобы импортировать</translation>
    </message>
    <message>
        <location filename="../widgets/welcome_screen.py" line="93"/>
        <source>Browse...</source>
        <translation>Обзор…</translation>
    </message>
</context>
<context>
    <name>_ModelRow</name>
    <message>
        <location filename="../widgets/setup_wizard.py" line="603"/>
        <source>  — Installed</source>
        <translation>  — Установлено</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="625"/>
        <source>Downloading...</source>
        <translation>Скачивание…</translation>
    </message>
    <message>
        <location filename="../widgets/setup_wizard.py" line="632"/>
        <source>%d / %d MB</source>
        <translation>%d / %d МБ</translation>
    </message>
</context>
</TS>

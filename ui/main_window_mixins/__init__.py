from PySide6.QtCore import QCoreApplication


def _tr(source: str) -> str:
    """Translate a string with MainWindow context.

    Mixin classes can't use self.tr() because lupdate extracts the mixin class
    name as context, but at runtime the context resolves to MainWindow. This
    helper uses the explicit context so extraction and lookup match.
    """
    return QCoreApplication.translate("MainWindow", source)


from .menu_mixin import MenuMixin
from .shortcuts_mixin import ShortcutsMixin
from .clip_mixin import ClipMixin
from .import_mixin import ImportMixin
from .inference_mixin import InferenceMixin
from .alpha_import_mixin import AlphaImportMixin
from .model_run_mixin import ModelRunMixin
from .cancel_mixin import CancelMixin
from .worker_mixin import WorkerMixin
from .annotation_mixin import AnnotationMixin
from .export_mixin import ExportMixin
from .session_mixin import SessionMixin
from .settings_mixin import SettingsMixin

__all__ = [
    "MenuMixin", "ShortcutsMixin", "ClipMixin", "ImportMixin",
    "InferenceMixin", "AlphaImportMixin", "ModelRunMixin", "CancelMixin",
    "WorkerMixin", "AnnotationMixin",
    "ExportMixin", "SessionMixin", "SettingsMixin",
]

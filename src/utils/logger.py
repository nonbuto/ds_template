import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """ロガーを取得・設定する関数

    Args:
        name: ロガーの名前（通常は __name__）
        log_file: ログファイルのパス。Noneの場合は標準出力のみ
        level: ログレベル

    Returns:
        設定済みのロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 既存のハンドラをクリア（重複出力防止）
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 標準出力用ハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイル出力用ハンドラ（指定された場合）
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 伝播を防ぐ
    logger.propagate = False

    return logger

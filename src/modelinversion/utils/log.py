import sys
import os
from typing import Any, List, Tuple, Union

# class Tee(object):
#     """A workaround method to print in console and write to log file
#     """
#     def __init__(self, name, mode):
#         self.file = open(name, mode)
#         self.stdout = sys.stdout
#         sys.stdout = self
#     def __del__(self):
#         sys.stdout = self.stdout
#         self.file.close()
#     def write(self, data):
#         if not '...' in data:
#             self.file.write(data)
#         self.stdout.write(data)
#     def flush(self):
#         self.file.flush()

class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_dir: str = None, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            if file_dir is not None:
                os.makedirs(file_dir, exist_ok=True)
                file_name = os.path.join(file_dir, file_name)
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        # self.stderr = sys.stderr

        sys.stdout = self
        # sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        # if sys.stderr is self:
        #     sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
            

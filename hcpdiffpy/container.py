from pathlib import Path
from typing import Union

class SimgCmd:
    def __init__(self, config: dict, simg: Union[str, None]) -> None:
        if simg is None:
            self.command = None
        else:
            self.command = (f"singularity run -B {config['work_dir']}:{config['work_dir']},"
                        f"{config['output_dir']}:{config['output_dir']},"
                        f"{config['subject_dir']}:{config['subject_dir']}")
            self._simg = simg

    def cmd(self, command: str, options: Union[str, None] = None) -> str:
        if self.command is None:
            run_cmd = command
        else:
            if options is None:
                run_cmd = f"{self.command} {self._simg} {command}"
            else:
                run_cmd = f"{self.command} {options} {self._simg} {command}"

        return run_cmd

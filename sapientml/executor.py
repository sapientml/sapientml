# Copyright 2023 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from shutil import copyfile
from typing import Optional

from .params import CancellationToken, Pipeline, RunningResult
from .util.logging import setup_logger


def run(file_path: str, timeout: int, cancel: Optional[CancellationToken]) -> RunningResult:
    start_time = time.time()

    if platform.system() == "Windows":
        executable = None
        encoding = "cp932"
    else:
        executable = "/bin/bash"
        encoding = "utf-8"

    process = subprocess.Popen(
        f"{sys.executable} {file_path}",
        shell=True,
        executable=executable,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    endtime = time.time() + timeout if timeout > 0 else None
    interrupted_reason = None

    while process.poll() is None:
        if endtime is not None and time.time() > endtime:
            interrupted_reason = "Timeout"
            process.kill()
            break
        if cancel is not None and cancel.isTriggered:
            interrupted_reason = "Cancelled by user"
            process.kill()
            break
        time.sleep(1)

    output, error = process.communicate(timeout=timeout if timeout > 0 else None)

    if interrupted_reason is not None:
        output = ""
        error = interrupted_reason
    else:
        output = output.decode(encoding)
        error = error.decode(encoding)

    result = RunningResult(
        output=output,
        error=error,
        returncode=process.returncode,
        time=int(round(time.time() - start_time)),
    )
    return result


class PipelineExecutor:
    def __init__(
        self,
        loglevel=logging.INFO,
    ):
        self._logger = setup_logger()
        self._logger.setLevel(loglevel)

    def execute(
        self,
        pipeline_list: list[Pipeline],
        initial_timeout: int,
        output_dir: Path,
        cancel: Optional[CancellationToken],
    ) -> tuple[Optional[tuple[Pipeline, RunningResult]], Optional[list[tuple[Pipeline, RunningResult]]]]:
        candidate_scripts: list[tuple[Pipeline, RunningResult]] = []

        if candidate_scripts is None:
            self._logger.warning("No candidate is generated.")
            return None, None

        # copy libs
        lib_path = output_dir / "lib"
        lib_path.mkdir(exist_ok=True)
        copyfile(Path(__file__).parent / "../static/lib" / "sample_dataset.py", lib_path / "sample_dataset.py")

        for index, pipeline in enumerate(pipeline_list, start=1):
            script_name = f"{index}_script.py"
            script_path = (output_dir / script_name).absolute().as_posix()
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(pipeline.code_for_validation)

            self._logger.info(f"Running script ({index}/{len(pipeline_list)}) ...")
            running_result = run(script_path, initial_timeout, cancel)
            candidate_scripts.append((pipeline, running_result))
            reason = ""
            error_message = running_result.error.strip().split("\n")
            if running_result.returncode != 0:
                if running_result.returncode == 124:
                    # Status code 124 means timeout on linux timeout command
                    reason = "Timeout"
                elif error_message and error_message[-1]:
                    # Show the stack trace so users can get full error info
                    reason = "\n".join(error_message)
                else:
                    # Unknown error
                    reason = f"Status code: {running_result.returncode}"
                self._logger.warning(f"Failed to run a pipeline '{script_name}': {reason}")

        return candidate_scripts

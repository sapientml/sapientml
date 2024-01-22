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

import asyncio
import glob
import os
import platform
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path
from shutil import copyfile
from typing import Optional

import nest_asyncio

from .params import CancellationToken, Code, RunningResult
from .util.logging import setup_logger

logger = setup_logger()

nest_asyncio.apply()


def run(
    file_path: str, timeout: int, cancel: Optional[CancellationToken] = None, cwd: Optional[str] = None
) -> RunningResult:
    """Executing run() function based on operating system.

    Parameters
    ----------
    filepath : str
        Path of the file executed.
    timeout : int
        Timeout for the execution.
    cancel : CancellationToken, optional
        Object for cancellation.
    cwd : str, optional
        Working directory.

    Returns
    -------
    result : RunningResult

    """

    if platform.system() == "Windows":
        encoding = "cp932"
        replace_newline = "\r"
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        encoding = "utf-8"
        replace_newline = ""
        loop = asyncio.get_event_loop()

    async def _run() -> RunningResult:
        start_time = time.time()

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            file_path,
            cwd=cwd or os.path.dirname(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        endtime = time.time() + timeout if timeout > 0 else None

        async def _read_stream(stream, encoding, replace_newline):
            lines = []
            while True:
                line = await stream.readline()
                if line:
                    lines.append(line.decode(encoding).replace(replace_newline, ""))
                if stream.at_eof():
                    break
                await asyncio.sleep(0.1)
            return lines

        async def _wait_timeout_or_cancel(proc, endtime, cancel):
            interrupted_reason = None
            returncode = None

            while True:
                if proc.stdout.at_eof() and proc.stderr.at_eof():
                    break

                if endtime is not None and time.time() > endtime:
                    returncode = -9
                    interrupted_reason = "Timeout"
                    print("Terminating due to timeout")
                    proc.kill()
                    break
                if cancel is not None and cancel.is_triggered:
                    returncode = -9
                    interrupted_reason = "Cancelled by user"
                    print("Terminating due to cancellation")
                    proc.kill()
                    break
                await asyncio.sleep(1)
            return returncode, interrupted_reason

        results = await asyncio.gather(
            _read_stream(process.stdout, encoding, replace_newline),
            _read_stream(process.stderr, encoding, replace_newline),
            _wait_timeout_or_cancel(process, endtime, cancel),
        )

        await process.wait()

        stdout_lines = results[0]
        stderr_lines = results[1]
        returncode, interrupted_reason = results[2]

        if not returncode:
            returncode = process.returncode

        if interrupted_reason is not None:
            output = ""
            error = interrupted_reason
        else:
            output = "".join(stdout_lines)
            error = "".join(stderr_lines)

        result = RunningResult(
            output=output,
            error=error,
            returncode=returncode,
            time=int(round(time.time() - start_time)),
        )
        return result

    return loop.run_until_complete(_run())


class PipelineExecutor:
    """PipelineExecutor class for executing the generated pipelines."""

    def __init__(
        self,
    ):
        pass

    def execute(
        self,
        pipeline_list: list[Code],
        initial_timeout: int,
        output_dir: Path,
        cancel: Optional[CancellationToken],
    ) -> list[tuple[Code, RunningResult]]:
        """Executes the generated pipelines.

        Parameters
        ----------
        pipeline_list: list[Code]
            List of generated pipeline code.
        initial_timeout: int
            Timeout for the execution.
        output_dir: Path
            Output directory to store the results.
        cancel : CancellationToken, optional
            Object for cancellation.
        Returns
        -------
        candidate_scripts: list[tuple[Code, RunningResult]]
            It stores both the results and the code in list of tuples format.

        """
        candidate_scripts: list[tuple[Code, RunningResult]] = []

        # copy libs
        lib_path = output_dir / "lib"
        lib_path.mkdir(exist_ok=True)

        eps = entry_points(group="sapientml.export_modules")
        for ep in eps:
            for file in glob.glob(f"{ep.load().__path__[0]}/*.py"):
                copyfile(file, lib_path / Path(file).name)

        for index, pipeline in enumerate(pipeline_list, start=1):
            script_name = f"{index}_script.py"
            script_path = (output_dir / script_name).absolute().as_posix()
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(pipeline.validation)

            logger.info(f"Running script ({index}/{len(pipeline_list)})...")
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
                logger.warning(f"Failed to run a pipeline '{script_name}': {reason}")

        return candidate_scripts

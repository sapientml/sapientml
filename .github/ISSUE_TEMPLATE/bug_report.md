---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Show your code calling `fit()`.

<details>
<summary> script </summary>

```python
# Paste your code here. The following is an example.
from sapientml import SapientML
sml = SapientML('your arguments')
sml.fit('your arguments')
```
</details>

2. Attach the datasets or dataframes input to `fit()` if possible.
3. Show the generated code such as `1_default.py` when it was generated. You may find it at `./outputs` folder or the folder specified by param `output_dir` of `fit()`

<details>
<summary> generated code </summary>

```python
# Paste the generated code here.
```
</details>

4. Show the messages of SapientML and/or generated code.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 20.04]
 - Docker Version (if applicable): [Docker version 20.10.17, build 100c701]
 - Python Version: [e.g. 3.10.11]
 - SapientML Version: [e.g. 0.4.9]


**Additional context**
Add any other context about the problem here.

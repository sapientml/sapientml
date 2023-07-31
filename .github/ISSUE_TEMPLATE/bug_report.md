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
1. Show your code calling `generate_code()`.

<details>
<summary> script </summary>

```python
# Paste your code here. The following is an example.
from sapientml import SapientMLGenerator
sml = SapientMLGenerator()
sml.generate_code('your arguments')
```
</details>

2. Attach the datasets or dataframes input to `generate_code()` if possible.
3. Show the generated code such as `1_default.py` when it was generated.

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
 - Python Version: [e.g. 3.9.12]
 - SapientML Version: [e.g. 2.3.4]


**Additional context**
Add any other context about the problem here.

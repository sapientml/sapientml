# Contributing to SapientML

Thank you for your interest in SapientML!
We appreciate your contributions and suggestions. Most contributions require you to sign-off on your commits via the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). When you submit a pull request, [a DCO-bot](https://github.com/apps/dco) will automatically determine whether you need to provide sign-off for your commit. Please follow the instructions provided by DCO-bot, as pull requests cannot be merged until the author(s) have provided sign-off to fulfill the DCO requirement. You may find more information on the DCO requirements [below](#developer-certificate-of-origin-signing-your-work).

## Conventional commits

Our repositories follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

The commit message should be structured as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

The commit contains the following structural elements:

1. **fix:** A commit type for a bug fix in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** A commit type for a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **build:** A commit type for changes that affect the build system or external dependencies
4. **docs:** A commit type for documentation only changes
5. **perf:** A commit type for code changes that improves performance
6. **refactor:** A commit type for code changes that neither fixes a bug nor adds a feature
7. **style:** A commit type for code changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
8. **test:** A commit type for adding missing tests or correcting existing tests
9. **BREAKING CHANGE:** A commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.

You can also provide a scope to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.

## Developer Certificate of Origin: Signing your work

#### Every commit needs to be signed

The Developer Certificate of Origin (DCO) is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing to the project. Here is the full text of the [DCO](https://developercertificate.org/), reformatted for readability:

```text
By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

Contributors sign-off that they adhere to these requirements by adding a `Signed-off-by` line to commit messages.

```text
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

Git even has a `-s` command line option to append this automatically to your commit message:

```sh
git commit -s -m 'This is my commit message'
```

Each Pull Request is checked  whether or not commits in a Pull Request do contain a valid `Signed-off-by` line.

#### I did not sign my commit, now what?!

No worries - You can easily replay your changes, sign them and force push them!

```sh
git checkout <branch-name>
git commit --amend --no-edit --sign-off
git push --force-with-lease <remote-name> <branch-name>
```

## Code of Conduct

This project has adopted the [Contributor Covenant Code of Conduct](https://github.com/sapientml/sapientml/blob/main/CODE_OF_CONDUCT.md)
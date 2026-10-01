"""
SECRET GUARD: keeps API keys and .env files out of git.

    python scripts/check_secrets.py            scan every tracked file
    python scripts/check_secrets.py --staged   scan what is about to be committed
    python scripts/check_secrets.py --history  scan every commit on every branch

Three layers use this one file:
    1. .githooks/pre-commit runs --staged, so a key never gets into a commit.
       Enable once per clone:  git config core.hooksPath .githooks
    2. tests/test_no_secrets.py runs the tracked-file scan in CI on every push.
    3. GitHub push protection (repo setting) is the last line of defence.

It reports the file, line number and the KIND of secret. It never prints the secret.
Standard library only, so it runs before anything is installed.
"""

import fnmatch
import re
import subprocess
import sys

# (kind, regex). Specific provider formats first, then a generic NAME=value rule.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI-style API key", re.compile(r"sk-[A-Za-z0-9_-]{24,}")),
    ("Groq API key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("NVIDIA API key", re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")),
    ("LangSmith API key", re.compile(r"lsv2_(?:pt|sk)_[A-Za-z0-9_]{20,}")),
    ("GitHub token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Claude OAuth token", re.compile(r"sk-ant-oat[A-Za-z0-9_-]{10,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")),
    (
        "credential in a URL",
        re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@\"'<>]+:[^/\s:@\"'<>]{6,}@[A-Za-z0-9.-]+"),
    ),
    (
        "secret assigned to a setting",
        re.compile(
            r"\b[A-Z][A-Z0-9_]*(?:API_KEY|AUTH_TOKEN|ACCESS_TOKEN|OAUTH_TOKEN|SECRET|PASSWORD)\b"
            r"[ \t]*[:=][ \t]*[\"']?(?![\"'\s]|os\.|\$|<|\{)[^\s\"'#]{8,}"
        ),
    ),
]

# Files that must never be committed, whatever they contain.
FORBIDDEN_FILES = [".env", ".env.*", "*.pem", "*.key", "*.p12", "*.db", "*.sqlite",
                   "credentials.json", ".credentials.json"]
ALLOWED_FILES = [".env.example"]

# Test fixtures that are deliberately fake. Keep this list short and obvious.
# The second entry is a placeholder from an old code comment that is still in history.
FAKE_MARKERS = ("alice:s3cret@proxy.example.com", "user:" + "password@host")


def find_secrets(text: str) -> list[tuple[int, str]]:
    """[(line number, kind)] for every secret-looking string. Never returns the secret."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in FAKE_MARKERS):
            continue
        for kind, pattern in PATTERNS:
            if pattern.search(line):
                hits.append((number, kind))
                break
    return hits


def is_forbidden_file(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if name in ALLOWED_FILES:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_FILES)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    return result.stdout


def scan_files(paths: list[str], read) -> list[str]:
    problems = []
    for path in paths:
        if is_forbidden_file(path):
            problems.append(f"{path}: this kind of file must never be committed")
            continue
        try:
            text = read(path)
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: nothing to scan
        problems += [f"{path}:{line}: {kind}" for line, kind in find_secrets(text)]
    return problems


def scan_tracked() -> list[str]:
    paths = [p for p in _git("ls-files").splitlines() if p]

    def read(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    return scan_files(paths, read)


def scan_staged() -> list[str]:
    names = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    return scan_files([n for n in names if n], lambda path: _git("show", f":{path}"))


def scan_history() -> list[str]:
    problems = []
    added = _git("log", "--all", "--name-only", "--diff-filter=A", "--pretty=format:")
    for path in sorted({p for p in added.splitlines() if p}):
        if is_forbidden_file(path):
            problems.append(f"history: {path} was committed at some point")
    for line, kind in find_secrets(_git("log", "--all", "-p")):
        problems.append(f"history (log line {line}): {kind}")
    return problems


def main(argv: list[str]) -> int:
    if "--staged" in argv:
        problems, what = scan_staged(), "staged changes"
    elif "--history" in argv:
        problems, what = scan_history(), "full git history"
    else:
        problems, what = scan_tracked(), "tracked files"

    if not problems:
        print(f"secret guard: {what} are clean")
        return 0
    print(f"secret guard: BLOCKED. Possible secrets in {what}:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print("\nKeys belong in .env (ignored by git), never in a tracked file.\n"
          "If this is a false alarm, fix the pattern in scripts/check_secrets.py.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# agent/prompt/skills.py  |  BLOCK 3 SYSTEM PROMPT
"""
SKILLS (part of BLOCK 3: SYSTEM PROMPT)

Progressive disclosure, the Agent Skills convention (SKILL.md):
  - Each skill is agent/prompt/skills/<name>/SKILL.md with a frontmatter `name` and `description`.
  - Only the one-line index goes into the system prompt, so 50 skills cost ~50 lines.
  - The model calls load_skill(name) to pull the full procedure when a task needs it.

Use a skill for a reusable PROCEDURE (how to review, how to write a report). Use a tool
for a CAPABILITY (call an API). Use system.md for rules that apply to every turn.
"""

from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def discover_skills(directory: Path = SKILLS_DIR) -> dict[str, Skill]:
    skills = {}
    for file in sorted(directory.glob("*/SKILL.md")):
        meta, _ = _split_frontmatter(file.read_text(encoding="utf-8"))
        name = meta.get("name", file.parent.name)
        skills[name] = Skill(name, meta.get("description", ""), file)
    return skills


def skill_body(name: str, directory: Path = SKILLS_DIR) -> str | None:
    skill = discover_skills(directory).get(name)
    if skill is None:
        return None
    return _split_frontmatter(skill.path.read_text(encoding="utf-8"))[1]


def skills_index(directory: Path = SKILLS_DIR) -> str:
    skills = discover_skills(directory)
    if not skills:
        return ""
    lines = [f"- {s.name}: {s.description}" for s in skills.values()]
    return (
        "# Skills\n\nBefore a task that one of these covers, call load_skill(name) "
        "and follow it.\n\n" + "\n".join(lines)
    )

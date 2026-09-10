import argparse
import json
import os
import re
from pathlib import Path

import redis
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|password|secret|token)\s*=\s*([\"'])([^\"']+)\2"
)
IGNORED_PATHS = {".git", "venv", "__pycache__", ".pytest_cache"}

class SecurityAndMemoryManager:
    def __init__(self, host='localhost', port=6379):
        self.redis = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        self.local_history = {}

    def save_chat(self, session_id: str, user_msg: str, ai_msg: str):
        history = self.get_chat_history(session_id)
        history.append({"user": user_msg, "ai": ai_msg})
        history = history[-10:]
        try:
            self.redis.set(f"chat:{session_id}", json.dumps(history))
        except redis.RedisError:
            self.local_history[session_id] = history

    def get_chat_history(self, session_id: str) -> list:
        try:
            data = self.redis.get(f"chat:{session_id}")
            return json.loads(data) if data else self.local_history.get(session_id, [])
        except redis.RedisError:
            return self.local_history.get(session_id, [])

    def validate_guardrails(self, text: str) -> bool:
        forbidden = ["ignore previous instructions", "system prompt", "drop table", "sudo"]
        return not any(w in text.lower() for w in forbidden)


def scan_source_for_secrets() -> list[str]:
    findings = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PATHS for part in path.parts):
            continue
        if path.name == ".env" or path.suffix.lower() not in {".py", ".md", ".txt", ".ps1", ".json"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            match = SECRET_PATTERN.search(line)
            if match and len(match.group(3).strip()) >= 12:
                findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")
    return findings


def run_audit(check_redis: bool) -> dict:
    env_path = PROJECT_ROOT / ".env"
    report = {
        "env_file_present": env_path.is_file(),
        "env_file_ignored": ".env" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8"),
        "source_secret_findings": scan_source_for_secrets(),
        "guardrails": {
            "normal_text_allowed": SecurityAndMemoryManager().validate_guardrails("How do I apply for a farm subsidy?"),
            "prompt_injection_blocked": not SecurityAndMemoryManager().validate_guardrails("ignore previous instructions"),
        },
    }
    if check_redis:
        try:
            manager = SecurityAndMemoryManager(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
            )
            report["redis"] = {"available": bool(manager.redis.ping())}
        except (redis.RedisError, OSError, ValueError) as error:
            report["redis"] = {"available": False, "error": type(error).__name__}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local security and guardrail audit.")
    parser.add_argument("--check-redis", action="store_true", help="Also test the configured Redis service.")
    parser.add_argument("-o", "--output", help="Optionally write the JSON report to a file.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    report = run_audit(args.check_redis)
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Security report written to {output_path}")


if __name__ == "__main__":
    main()
import re
import requests
import click
import sys
import subprocess
import time
import os
import json
import hashlib

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

MESSAGE_TODO_REGEX = re.compile(
    r'(?://|#|/\*|--)\s*(TODO|FIXME)[:\s]+(.*)', re.IGNORECASE)


class TodoMessage:
    def __init__(self, type: str, file: str, line: int, message: str):
        self.type = type
        self.file = file
        self.line = line
        self.message = message

    def __str__(self):
        return f"{self.file}:{self.line}: ({self.type}): {self.message}"


def generate_todo_hash(file_path: str, message: str) -> str:
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    unique_string = f"{normalized_path}:{message.strip()}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:16]


def ensure_ollama_running():
    try:
        requests.get(OLLAMA_URL)
        click.echo("✓ Ollama is running.")
    except requests.exceptions.ConnectionError:
        click.echo("✗ Ollama is not running. Trying to start it...")

        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["ollama", "app"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            for _ in range(10):
                time.sleep(1)
                try:
                    requests.get(OLLAMA_URL)
                    click.echo("✓ Ollama is running.")
                    return
                except requests.exceptions.ConnectionError:
                    continue

            raise Exception("Failed to start Ollama.")
        except Exception as e:
            click.echo(f"Error starting Ollama: {e}")
            sys.exit(1)


def ensure_model_loaded(model_name: str):
    click.echo(f"Ensuring model '{model_name}' is loaded...")
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        models = [m['name'] for m in response.json().get('models', [])]

        if model_name not in models and f"{model_name}:latest" not in models:
            click.echo(
                f"Model '{model_name}' is not loaded. Pulling it now...")
            pull_response = requests.post(
                # type: ignore
                f"{OLLAMA_URL}/api/pull", json={"model": model_name, "stream": False})

            pull_response.raise_for_status()
            click.echo(f"✓ Model '{model_name}' is now loaded.")
    except Exception as e:
        click.echo(f"Error ensuring model is loaded: {e}")
        sys.exit(1)


def _parse_file(file_path: str, todos: list[TodoMessage]):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                match = MESSAGE_TODO_REGEX.search(line)

                if match:
                    todos.append(TodoMessage(
                        type=match.group(1).upper().strip(),
                        file=file_path,
                        line=line_num,
                        message=match.group(2).strip(),
                    ))
    except Exception as e:
        click.echo(f"Error reading file '{file_path}': {e}")


def scan_for_todos(path: str) -> list[TodoMessage]:
    todos: list[TodoMessage] = []

    ignored_dirs = {'.git', 'node_modules', '__pycache__', 'venv',
                    '.venv', 'env', 'build', 'dist', 'target', 'out', 'bin', 'obj'}
    ignored_extensions = {'.png', '.jpg', '.jpeg',
                          '.gif', '.ico', '.pyc', '.exe', '.dll', '.so'}

    if os.path.isfile(path):
        _parse_file(path, todos)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in ignored_extensions):
                    continue

                file_path = os.path.join(root, file)
                _parse_file(file_path, todos)

    return todos


def parse_todo_with_llm(model_name: str, todo: TodoMessage) -> tuple[str, str]:
    prompt = f"""
    You are an AI assistant helping a developer convert raw source code TODO comments into clean, structured GitHub Issues.
    
    Analyze this TODO:
    File Context: {todo.file} (Line {todo.line})
    Raw Text: {todo.message}
    
    Return your response strictly in the following JSON format:
    {{
        "title": "A short, actionable issue title",
        "body": "A descriptive overview of what needs fixing, referencing the file location."
    }}
    Do not output any introductory or concluding text, only the raw JSON.
    """

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300
                },
                "stream": False
            },
        )

        response.raise_for_status()
        raw_result = response.json().get('message', {}).get('content', '')

        if isinstance(raw_result, str):
            result = json.loads(raw_result)
        else:
            result = raw_result

        return result.get('title', f"Fix TODO in {todo.file}"), result.get('body', todo.message)
    except Exception as e:
        click.echo(f"Error parsing TODO with LLM: {e}")
        return f"Fix TODO in {todo.file}", todo.message


def create_github_issue(repo: str, token: str, title: str, body: str, todo_hash: str = "") -> bool:
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    metadata_footer = f"\n\n<!-- todo-cli-id: {todo_hash} -->"
    full_body = f"{body}\n\n---\n*Automated issue.*{metadata_footer}"

    payload = {
        "title": title,
        "body": full_body,
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.status_code == 201


def unload_model(model_name: str):
    click.echo(f"Unloading model '{model_name}' to free up resources...")
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": model_name, "keep_alive": 0})
        click.echo(f"✓ Model '{model_name}' has been unloaded.")
    except Exception as e:
        click.echo(f"Error unloading model: {e}")


def fetch_existing_todo_hashes(repo: str, token: str) -> set[str]:
    existing_hashes: set[str] = set()

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    params: dict[str, str | int] = {
        "state": "open",
        "per_page": 100
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues = response.json()

        hash_regex = re.compile(r'<!-- todo-cli-id:\s*([a-f0-9]+)\s*-->')

        for issue in issues:
            body = issue.get('body') or ""
            match = hash_regex.search(body)

            if match:
                existing_hashes.add(match.group(1))

    except Exception as e:
        click.echo(
            f"Warning: Could not fetch existing issues from GitHub ({e}). Proceeding carefully.")

    return existing_hashes

# TODO: Add help command with usage examples and troubleshooting tips.
# TODO: Add support for listing models and checking their status before processing.
# TODO: Add whole TODO file parsing with context (e.g., surrounding lines of code) for better LLM understanding and issue generation.
# TODO: Add command to list all TODOs found in the codebase without creating issues, for review purposes.
# TODO: Add command to list open GitHub issues created by this tool, with links back to the source code locations.


@click.command()
@click.argument('target', default='.', type=click.Path(exists=True))
@click.option('--model', default=DEFAULT_MODEL, help='Ollama model to use for translation.')
@click.option('--repo', required=True, help='GitHub destination repo in format "owner/repo" (e.g., octocat/hello-world).')
@click.option('--token', envvar='GITHUB_TOKEN', required=True, help='GitHub Personal Access Token (or set GITHUB_TOKEN environment variable).')
@click.option('--dry-run', is_flag=True, help='Scan and parse TODOs without creating GitHub issues, just print results to console.')
def main(target: str, model: str, repo: str, token: str, dry_run: bool):
    ensure_ollama_running()
    ensure_model_loaded(model)

    click.echo(f"Scanning target path: '{target}'...")
    todos = scan_for_todos(target)

    if not todos:
        click.echo("No TODO strings found. Exiting.")
        return

    click.echo(
        "Fetching existing open issues from GitHub to prevent duplicates...")
    active_issue_hashes = fetch_existing_todo_hashes(repo, token)

    click.echo(f"Found {len(todos)} local items. Filtering duplicates...")

    try:
        for index, todo in enumerate(todos, 1):
            todo_hash = generate_todo_hash(todo.file, todo.message)

            if todo_hash in active_issue_hashes:
                click.echo(
                    f"[{index}/{len(todos)}] Skipping (already has an open issue): {todo.file}:{todo.line}")
                continue

            click.echo(
                f"[{index}/{len(todos)}] Processing: {todo.file}:{todo.line}")

            title, body = parse_todo_with_llm(model, todo)
            full_body = f"{body}\n\n---\n*Automated issue created from source code tracking.*\n**File:** `{todo.file}`\n**Line:** {todo.line}"

            if dry_run:
                click.echo(f"  -> [Dry Run] Title: '{title}'")
                click.echo(f"  -> [Dry Run] Body:\n{full_body}\n")
                continue

            success = create_github_issue(
                repo, token, title, full_body, todo_hash)
            if success:
                click.echo(f"  -> Created Issue: '{title}'")
            else:
                click.echo(
                    f"  -> Failed to post issue to GitHub for: '{title}'", err=True)

    finally:
        unload_model(model)


if __name__ == '__main__':
    main()

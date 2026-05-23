import re
import click
import sys
import subprocess
import os
import json
import hashlib
import asyncio
import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

MESSAGE_TODO_REGEX = re.compile(
    r'(?://|#|/\*|--)\s*(TODO|FIXME)[:\s]+(.*)', re.IGNORECASE)

OLLAMA_SEMAPHORE = asyncio.Semaphore(2)


class TodoMessage:
    def __init__(self, type: str, file: str, line: int, message: str):
        self.type = type
        self.file = file
        self.line = line
        self.message = message

    def __str__(self):
        return f"{self.file}:{self.line}: ({self.type}): {self.message}"


def get_file_context(file_path: str, target_line: int, window: int = 80) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start_index = max(0, target_line - window - 1)
        end_index = min(len(lines), target_line + window)

        context_lines: list[str] = []
        for i in range(start_index, end_index):
            line_num = i + 1

            prefix = ">> " if line_num == target_line else "   "
            context_lines.append(f"{prefix}{line_num:4d}: {lines[i].rstrip()}")

        return "\n".join(context_lines)
    except Exception as e:
        click.echo(
            f"Warning: Could not read file context for '{file_path}' (line {target_line}): {e}")
        return f"(Could not retrieve file context: {e})"


def generate_todo_hash(file_path: str, message: str) -> str:
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    unique_string = f"{normalized_path}:{message.strip()}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:16]


async def ensure_ollama_running(client: httpx.AsyncClient):
    try:
        await client.get(OLLAMA_URL)
        click.echo("✓ Ollama is running.")
    except httpx.ConnectError:
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
                await asyncio.sleep(1)
                try:
                    await client.get(OLLAMA_URL)
                    click.echo("✓ Ollama is running.")
                    return
                except httpx.ConnectError:
                    continue

            raise Exception("Failed to start Ollama.")
        except Exception as e:
            click.echo(f"Error starting Ollama: {e}")
            sys.exit(1)


async def ensure_model_loaded(client: httpx.AsyncClient, model_name: str):
    click.echo(f"Ensuring model '{model_name}' is loaded...")
    try:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        models = [m['name'] for m in response.json().get('models', [])]

        if model_name not in models and f"{model_name}:latest" not in models:
            click.echo(
                f"Model '{model_name}' is not loaded. Pulling it now...")
            pull_response = await client.post(
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


async def parse_todo_with_llm(client: httpx.AsyncClient, model_name: str, todo: TodoMessage) -> tuple[str, str]:
    code_context = get_file_context(todo.file, todo.line)

    prompt = f"""
    You are an expert Technical Product Manager and Lead Engineer. 
    Your task is to convert a developer's code TODO comment into a rich, structured GitHub Issue.

    Analyze the following TODO and its surrounding code context to understand WHY it was created.

    File: {todo.file}
    Target TODO: "{todo.message}"

    Code Context:
    ```
    {code_context}
    ```

    Return your response strictly in the following JSON format. Do not output any Markdown wrapping the JSON, just the raw JSON object:
    {{
        "title": "A short, actionable issue title",
        "context": "Current context of why the issue was created based on the code",
        "objective": "Objective of the issue",
        "user_story": "As a [role], I want to [action] so that [benefit]",
        "acceptance_criteria": [
            "Specific measurable condition 1",
            "Specific measurable condition 2"
        ]
    }}
    """

    async with OLLAMA_SEMAPHORE:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "options": {
                        "temperature": 0.35,
                        "num_predict": 1000
                    },
                    "stream": False
                },
                timeout=90,
            )

            response.raise_for_status()
            raw_result = response.json().get('message', {}).get('content', '')

            if isinstance(raw_result, str):
                result = json.loads(raw_result)
            else:
                result = raw_result

            title = result.get('title', f'Fix TODO in {todo.file}').strip()
            ac_list = result.get('acceptance_criteria', [
                                 "Resolve the TODO successfully."])
            ac_markdown = "\n".join(f"- [ ] {ac.strip()}" for ac in ac_list)

            body_markdown = f"""### Context
            {result.get('context', 'No additional context provided.').strip()}

            ### Objective
            {result.get('objective', 'No specific objective provided.').strip()}

            ### User Story
            {result.get('user_story', 'No user story provided.').strip()}

            ### Acceptance Criteria
            {ac_markdown}
            """

            return title, body_markdown
        except Exception as e:
            click.echo(f"Error parsing TODO with LLM: {e}")
            return f"Fix TODO in {todo.file}", todo.message


async def create_github_issue(client: httpx.AsyncClient, repo: str, token: str, title: str, body: str, todo_hash: str = "") -> bool:
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

    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.status_code == 201


async def unload_model(client: httpx.AsyncClient, model_name: str):
    click.echo(f"Unloading model '{model_name}' to free up resources...")
    try:
        await client.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model_name, "keep_alive": 0})
        click.echo(f"✓ Model '{model_name}' has been unloaded.")
    except Exception as e:
        click.echo(f"Error unloading model: {e}")


async def fetch_existing_todo_hashes(client: httpx.AsyncClient, repo: str, token: str) -> set[str]:
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
        response = await client.get(url, headers=headers, params=params)
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


async def process_single_todo(*, client: httpx.AsyncClient, model: str, repo: str, token: str, todo: TodoMessage, index: int, total: int, dry_run: bool):
    todo_hash = generate_todo_hash(todo.file, todo.message)

    click.echo(f"[{index}/{total}] Processing: {todo.file}:{todo.line}")
    title, body = await parse_todo_with_llm(client, model, todo)

    full_body = (
        f"{body}\n\n---\n"
        f"*Automated issue created from source code tracking.*\n"
        f"**File:** `{todo.file}`\n**Line:** {todo.line}\n"
        f"<!-- todo-cli-id: {todo_hash} -->"
    )

    if dry_run:
        click.echo(f"  -> [Dry Run] Title: '{title}'")
        click.echo(f"  -> [Dry Run] Body:\n{full_body}\n")
        return

    success = await create_github_issue(client, repo, token, title, full_body, todo_hash)
    if success:
        click.echo(f"  -> Created Issue: '{title}'")
    else:
        click.echo(
            f"  -> Failed to post issue to GitHub for: '{title}'", err=True)


async def main_flow(target: str, model: str, repo: str, token: str, dry_run: bool):
    async with httpx.AsyncClient() as client:
        await ensure_ollama_running(client)
        await ensure_model_loaded(client, model)

        click.echo(f"Scanning target path: '{target}'...")
        todos = scan_for_todos(target)

        if not todos:
            click.echo("No TODO strings found. Exiting.")
            return

        click.echo("Syncing current live GitHub records to prune duplicates...")
        active_remote_hashes = await fetch_existing_todo_hashes(client, repo, token)

        backlog_tasks: list[tuple[TodoMessage, int]] = []
        for index, todo in enumerate(todos, 1):
            todo_hash = generate_todo_hash(todo.file, todo.message)

            if todo_hash in active_remote_hashes:
                click.echo(
                    f"[{index}/{len(todos)}] Skipping (already has an open issue): {todo.file}:{todo.line}")
                continue

            backlog_tasks.append((todo, index))

        if not backlog_tasks:
            click.echo(
                "✓ All local tasks are already mapped to active issues on GitHub.")
            await unload_model(client, model)

            return

        click.echo(
            f"Found {len(backlog_tasks)} items to process after filtering duplicates...")
        click.echo(
            f"Launching processing engine context for {len(backlog_tasks)} tasks...")

        execution_pool = [
            process_single_todo(
                client=client,
                model=model,
                repo=repo,
                token=token,
                todo=todo,
                index=index,
                total=len(backlog_tasks),
                dry_run=dry_run
            ) for todo, index in backlog_tasks
        ]

        try:
            await asyncio.gather(*execution_pool)
        finally:
            await unload_model(client, model)


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
    asyncio.run(main_flow(target, model, repo, token, dry_run))


if __name__ == '__main__':
    main()

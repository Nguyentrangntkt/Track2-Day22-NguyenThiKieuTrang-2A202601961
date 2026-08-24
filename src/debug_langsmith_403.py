"""Minimal LangSmith-only probe for diagnosing a 403 from /sessions.

This script never imports the lab pipeline or any OpenAI client, and it never
prints the API key. It reads only the lab's original LANGCHAIN_* contract.
"""
import inspect
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, __version__


ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


def status_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    return str(getattr(response, "status_code", "unknown"))


def main() -> int:
    import os

    api_key = os.getenv("LANGCHAIN_API_KEY")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    if not api_key:
        print("LANGCHAIN_API_KEY is not configured.")
        return 2

    supports_workspace_id = "workspace_id" in inspect.signature(Client).parameters
    print(f"langsmith={__version__}")
    print(f"Client(workspace_id=...): {supports_workspace_id}")

    client = Client(api_url=endpoint, api_key=api_key, auto_batch_tracing=False)
    try:
        # /settings is an authenticated, workspace-resolving request. It does
        # not create traces and does not contact any LLM provider.
        settings = client._get_settings()
    except Exception as error:
        print(f"AUTH_OR_WORKSPACE: FAIL ({status_code(error)})")
        print(f"error_type={type(error).__name__}")
        print(f"error_message={error}")
        print("The default workspace could not be confirmed at this endpoint.")
        return 1

    print("AUTH_OR_WORKSPACE: PASS")
    print(f"workspace_id={settings.id}")
    print(f"workspace_name={settings.display_name}")

    try:
        projects = list(client.list_projects(limit=3))
    except Exception as error:
        print(f"SESSIONS_READ: FAIL ({status_code(error)})")
        print(f"error_type={type(error).__name__}")
        print(f"error_message={error}")
        print("The resolved workspace denies GET /sessions for this key.")
        return 1

    print(f"SESSIONS_READ: PASS ({len(projects)} project(s) returned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

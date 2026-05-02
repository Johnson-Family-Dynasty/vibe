from config import (
    CLASSROOM_NAME,
    CLASSROOM_TIMEZONE,
    LEARNING_CENTER_FULL_NAME,
    LEARNING_CENTER_OWNER,
    LEARNING_CENTER_REPO,
)


TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            f"Get the current date and time. Defaults to the classroom timezone ({CLASSROOM_TIMEZONE}). "
            "Use this when the user asks what time it is, what day/date it is, or needs the current timestamp."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        f"Optional IANA timezone like America/Chicago or UTC. "
                        f"Defaults to {CLASSROOM_TIMEZONE}."
                    ),
                }
            },
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information. Use for current events, facts, "
            "documentation, or anything that may have changed since training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch and read the text content of a specific URL. "
            "Use when the user pastes a link they want you to read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Get current weather and a short forecast for any city or location. "
            "Always use this tool when a user asks about weather, temperature, forecast, or conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, city+state, or city+country (e.g. 'Libertyville, IL')"}
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_classroom_schedule",
        "description": (
            f"Read the {CLASSROOM_NAME} student schedule CSV files from {LEARNING_CENTER_FULL_NAME}. "
            "Use this first for questions about schoolwork, schedules, weekly timeline, or what Caleb, "
            "Elijah, Glory, or creativegt should do on a day."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "students": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Student names or aliases. Supports Caleb, creativegt, Elijah, and Glory. Omit for all students.",
                },
                "weekday": {
                    "type": "string",
                    "description": "Weekday name such as Monday, Tuesday, today, tomorrow, or next Monday.",
                },
                "target_date": {
                    "type": "string",
                    "description": "Specific date if known, ideally YYYY-MM-DD. Also accepts dates like May 4, 2026.",
                },
                "study_area": {
                    "type": "string",
                    "description": "Optional subject filter such as Math, STEM, Language Arts, or Social Studies.",
                },
            },
        },
    },
    {
        "name": "github",
        "description": (
            f"Access GitHub. Defaults to {LEARNING_CENTER_FULL_NAME} when owner/repo are omitted. "
            "Can summarize a repo, list files, read files/directories, search code, list commits, "
            "and look up issues or pull requests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "repo_summary",
                        "list_files",
                        "get_file",
                        "search_code",
                        "search_repos",
                        "get_issue",
                        "list_issues",
                        "get_pull_request",
                        "list_pull_requests",
                        "list_commits",
                    ],
                    "description": "The operation to perform",
                },
                "owner": {"type": "string", "description": f"Repo owner. Defaults to {LEARNING_CENTER_OWNER}."},
                "repo": {"type": "string", "description": f"Repo name. Defaults to {LEARNING_CENTER_REPO}."},
                "path": {"type": "string", "description": "File or directory path in the repo. Use empty string for the repo root."},
                "query": {"type": "string", "description": "Search string. For search_code, this is scoped to owner/repo unless query already includes repo:."},
                "issue_number": {"type": "integer", "description": "Issue number (for get_issue)"},
                "pull_number": {"type": "integer", "description": "Pull request number (for get_pull_request)"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Issue or pull request state. Defaults to open."},
                "branch": {"type": "string", "description": "Branch, tag, or commit SHA for list_files. Defaults to the repo default branch."},
            },
            "required": ["action"],
        },
    },
]

# Web Live Skill

Use this skill for current information, live websites, pasted URLs, weather, news, prices, docs, or anything likely to change.

Tool rules:

- Use the current classroom time context or `get_current_time` for date/time questions. Do not web search for the current time.
- Use `get_weather` for weather, temperature, forecast, or conditions.
- Use `fetch_webpage` when the user pastes a URL they want read or summarized.
- Use `web_search` for live/current information or when the user asks you to look something up.
- Cite relevant links from tool results briefly in the final answer.

Do not pretend live information came from memory. If a tool fails, say what failed and offer the next narrow lookup.

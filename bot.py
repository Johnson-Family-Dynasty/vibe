import asyncio
import concurrent.futures
import os
import re
import time
import json
from collections import defaultdict
from dataclasses import dataclass

import anthropic
import discord

from config import (
    CLASSROOM_NAME,
    COOLDOWN_SECONDS,
    ENABLE_TWO_STAGE_RESPONDER,
    LEARNING_CENTER_FULL_NAME,
    ALLOW_SARCASM,
    MAX_HISTORY,
    MAX_RESPONSE_LEN,
    MAX_SUMMARY_LENGTH,
    MAX_TOOL_ROUNDS,
    TONE_FRUSTRATED_THRESHOLD,
    TONE_FRUSTRATION_PHRASES,
    TONE_FRUSTRATION_SENSITIVITY,
    TONE_NEGATIVE_MARKERS,
    TONE_NEUTRAL_THRESHOLD,
    describe_known_classroom_member,
    infer_user_mode,
)
from prompting import build_system_prompt
from tools.github_tools import tool_github
from tools.schedule_tools import tool_get_classroom_schedule
from tools.schemas import TOOLS
from tools.time_tools import tool_get_current_time
from tools.web_tools import tool_fetch_webpage, tool_get_weather, tool_web_search

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
conversation_histories: dict[int, list] = defaultdict(list)
conversation_summaries: dict[int, str] = defaultdict(str)
summary_refresh_counts: dict[int, int] = defaultdict(int)
user_last_message: dict[int, float] = defaultdict(float)
TOOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=6)


@dataclass
class RequestTelemetry:
    tool_calls: int = 0
    tool_rounds: int = 0
    started_at: float = 0.0

    def latency_bucket(self) -> str:
        elapsed = time.time() - self.started_at
        if elapsed < 2:
            return "<2s"
        if elapsed < 5:
            return "2-5s"
        if elapsed < 10:
            return "5-10s"
        return "10s+"


def should_enable_tools(user_text: str) -> tuple[bool, str]:
    lowered = (user_text or "").lower()
    fresh_fact_keywords = (
        "weather", "forecast", "temperature", "news", "today", "live", "current",
        "latest", "breaking", "stock", "price", "score", "web", "search", "url", "website",
    )
    if any(keyword in lowered for keyword in fresh_fact_keywords):
        return True, "fresh_facts"
    return False, "conversational"


def resolve_tool_round_limit(user_text: str, tool_intent: str) -> int:
    lowered = (user_text or "").lower()
    if "github" in lowered or "repo" in lowered or "readme" in lowered:
        return MAX_TOOL_ROUNDS_REPO
    if "schedule" in lowered or "calendar" in lowered:
        return MAX_TOOL_ROUNDS_SCHEDULE
    if tool_intent == "fresh_facts":
        return MAX_TOOL_ROUNDS_FRESH_FACTS
    return MAX_TOOL_ROUNDS

# ──────────────────────────────────────────────
# DISCORD + ANTHROPIC CLIENTS
# ──────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True
intents.voice_states = True

client = discord.Client(intents=intents)
ai = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

STAGE_A_PROMPT = """Create a compact Stage A control object for the user's message.
Return JSON only with fields:
- intent: short string
- tone: one of [supportive, neutral, playful, firm, celebratory]
- need_tools: boolean
- response_plan: short bullet-style string (max 2 steps)
Do not include any answer to the user and do not mention Stage A/Stage B."""


def parse_stage_a(raw: str) -> dict:
    fallback = {
        "intent": "general_help",
        "tone": "supportive",
        "need_tools": False,
        "response_plan": "1) Answer directly\n2) Offer one helpful next step",
    }
    try:
        data = json.loads(raw)
        tone = data.get("tone", "supportive")
        if tone not in {"supportive", "neutral", "playful", "firm", "celebratory"}:
            tone = "supportive"
        return {
            "intent": str(data.get("intent", fallback["intent"]))[:80],
            "tone": tone,
            "need_tools": bool(data.get("need_tools", False)),
            "response_plan": str(data.get("response_plan", fallback["response_plan"]))[:220],
        }
    except Exception:
        return fallback


def build_stage_b_system_prompt(base_prompt: str, stage_a: dict) -> str:
    sarcasm_allowed = ALLOW_SARCASM and stage_a.get("tone") == "playful"
    return (
        base_prompt
        + "\n\nInternal response controls (do not reveal):\n"
        + f"- Intent: {stage_a.get('intent', 'general_help')}\n"
        + f"- Tone: {stage_a.get('tone', 'supportive')}\n"
        + f"- Need tools: {stage_a.get('need_tools', False)}\n"
        + "- Plan:\n"
        + f"{stage_a.get('response_plan', '')}\n"
        + f"- Sarcasm gate: {'enabled' if sarcasm_allowed else 'disabled'}\n"
        + "When sarcasm gate is disabled, avoid sarcasm entirely. When enabled, keep it very mild and friendly. "
          "Never mention internal stages or controls in the final response."
    )


async def dispatch_tool(name: str, inputs: dict) -> str:
    loop = asyncio.get_running_loop()
    try:
        if name == "web_search":
            return await loop.run_in_executor(TOOL_EXECUTOR, tool_web_search, inputs["query"])
        if name == "fetch_webpage":
            return await loop.run_in_executor(TOOL_EXECUTOR, tool_fetch_webpage, inputs["url"])
        if name == "get_weather":
            return await loop.run_in_executor(TOOL_EXECUTOR, tool_get_weather, inputs["location"])
        if name == "get_current_time":
            return await loop.run_in_executor(TOOL_EXECUTOR, lambda: tool_get_current_time(inputs.get("timezone")))
        if name == "get_classroom_schedule":
            return await loop.run_in_executor(TOOL_EXECUTOR, lambda: tool_get_classroom_schedule(**inputs))
        if name == "github":
            return await loop.run_in_executor(TOOL_EXECUTOR, lambda: tool_github(**inputs))
        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def preprocess_mentions(content: str, guild: discord.Guild) -> str:
    """Replace <@user_id> with @DisplayName so Claude sees names instead of IDs."""
    def replace(match):
        user_id = int(match.group(1))
        member = guild.get_member(user_id)
        return f"@{member.display_name} (id: {user_id})" if member else match.group(0)

    return re.sub(r"<@!?(\d+)>", replace, content)


async def send_long(channel: discord.abc.Messageable, reply_to: discord.Message, text: str) -> None:
    """Send one message reply. If too long, truncate to stay single-response."""
    if len(text) > MAX_RESPONSE_LEN:
        text = text[: MAX_RESPONSE_LEN - 22].rstrip() + "\n\n[Reply truncated.]"
    await reply_to.reply(text)


def help_text() -> str:
    return (
        "Hey! Here's what I can do:\n"
        "📚 Tutor students with step-by-step help in middle school subjects\n"
        "🧑‍🏫 Help teachers with rubric templates, lesson outlines, announcements, prompts, and assessment banks\n"
        f"🏠 Understand the `{CLASSROOM_NAME}` family/classroom profile for Caleb, Elijah, Glory, Josh, Lonisa, and Vibe\n"
        "✅ Help with family productivity, planning, routines, ideas, and quick research\n"
        "🗓️ Read student schedule CSVs — ask what Caleb, creativegt, Elijah, or Glory should do today/Monday/etc.\n"
        "🔎 Search the web for up-to-date info — just ask or paste a URL\n"
        "🌦️ Get live weather — ask for a city or forecast\n"
        f"🐙 Read GitHub repos — defaults to `{LEARNING_CENTER_FULL_NAME}`\n"
        "🏫 Stay aligned with the OpenTutor school model and daily workflow\n"
        "💬 @mention me with any question to chat\n"
        "`!reset` — clear our conversation history\n"
        "`!help` — show this message"
    )


def detect_tone_state(message_text: str) -> str:
    text = (message_text or "").strip().lower()
    if not text:
        return "calm"

    score = 0.0

    if any(phrase in text for phrase in TONE_FRUSTRATION_PHRASES):
        score += 2.0

    negative_hits = sum(1 for marker in TONE_NEGATIVE_MARKERS if marker in text)
    score += min(2.0, negative_hits * 0.6)

    repeated_punctuation_hits = len(re.findall(r"([!?])\1{1,}", text))
    score += min(1.2, repeated_punctuation_hits * 0.6)

    uppercase_words = re.findall(r"\b[A-Z]{3,}\b", message_text or "")
    if uppercase_words:
        score += 0.4

    score *= max(0.1, TONE_FRUSTRATION_SENSITIVITY)

    if score >= TONE_FRUSTRATED_THRESHOLD:
        return "frustrated"
    if score >= TONE_NEUTRAL_THRESHOLD:
        return "neutral"
    return "calm"


@client.event
async def on_ready():
    print(f"Cool cat {client.user} has logged in and is ready to vibe!")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    if not client.user.mentioned_in(message):
        return

    clean_content = re.sub(rf"<@!?{client.user.id}>", "", message.content).strip()
    if not clean_content:
        clean_content = "Hey Vibe, what's up?"

    lower = clean_content.lower()
    if lower == "!help":
        await message.reply(help_text())
        return

    if lower == "!reset":
        conversation_histories[message.channel.id] = []
        conversation_summaries[message.channel.id] = ""
        summary_refresh_counts[message.channel.id] = 0
        await message.reply("Fresh start! Conversation history cleared. 🐱")
        return

    now = time.time()
    if now - user_last_message[message.author.id] < COOLDOWN_SECONDS:
        await message.reply(f"Slow down a bit — one message every {COOLDOWN_SECONDS}s, please.")
        return
    user_last_message[message.author.id] = now

    if message.guild:
        clean_content = preprocess_mentions(clean_content, message.guild)

    roles = getattr(message.author, "roles", [])
    member_roles = [role.name for role in roles if role.name != "@everyone"]
    known_member = describe_known_classroom_member(message.author.display_name)
    likely_mode = infer_user_mode(member_roles, known_member)
    tone_state = detect_tone_state(clean_content)
    channel_name = getattr(message.channel, "name", "direct-message")
    user_context = (
        f"[Speaking with: {message.author.display_name} "
        f"(id: {message.author.id}), "
        f"Roles: {', '.join(member_roles) or 'none'}, "
        f"Known JFD member: {known_member}, "
        f"Likely mode: {likely_mode}, "
        f"tone_state: {tone_state}, "
        f"Channel: #{channel_name}]\n\n"
    )
    full_content = user_context + clean_content

    history = conversation_histories[message.channel.id]
    history.append({"role": "user", "content": full_content})

    system_prompt = build_system_prompt(full_content)
    await maybe_refresh_summary(message.channel.id, history, system_prompt)
    history = conversation_histories[message.channel.id]

    async with message.channel.typing():
        try:
            messages = list(history)
            use_tools, tool_intent = should_enable_tools(clean_content)
            max_tool_rounds = resolve_tool_round_limit(clean_content, tool_intent)
            telemetry = RequestTelemetry(started_at=time.time())

            while True:
                request_kwargs = {
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 2048,
                    "system": system_prompt,
                    "messages": messages,
                }
                if use_tools:
                    request_kwargs["tools"] = TOOLS
                response = ai.messages.create(**request_kwargs)

                if response.stop_reason == "tool_use":
                    tool_blocks = [block for block in response.content if block.type == "tool_use"]
                    telemetry.tool_rounds += 1
                    telemetry.tool_calls += len(tool_blocks)
                    if telemetry.tool_rounds > max_tool_rounds:
                        text = "I hit my tool-use limit for this question. Try narrowing the request or asking for one repo/file/site at a time."
                        conversation_histories[message.channel.id].append({"role": "assistant", "content": text})
                        await send_long(message.channel, message, text)
                        print(
                            f"[telemetry] tool_calls={telemetry.tool_calls} "
                            f"tool_rounds={telemetry.tool_rounds} "
                            f"latency_bucket={telemetry.latency_bucket()} "
                            f"intent={tool_intent} use_tools={use_tools}"
                        )
                        break

                    messages.append({"role": "assistant", "content": response.content})
                    results = await asyncio.gather(
                        *(
                            asyncio.wait_for(
                                dispatch_tool(block.name, block.input),
                                timeout=TOOL_CALL_TIMEOUT_SECONDS,
                            )
                            for block in tool_blocks
                        ),
                        return_exceptions=True,
                    )
                    normalized_results = []
                    tool_timed_out = False
                    for result in results:
                        if isinstance(result, asyncio.TimeoutError):
                            tool_timed_out = True
                            normalized_results.append("Tool timed out.")
                        elif isinstance(result, Exception):
                            normalized_results.append(f"Tool error: {result}")
                        else:
                            normalized_results.append(result)
                    tool_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                        for block, result in zip(tool_blocks, normalized_results)
                    ]
                    if tool_timed_out:
                        text = "Tools are running slow. I can answer now with what I know or keep searching—your call."
                        conversation_histories[message.channel.id].append({"role": "assistant", "content": text})
                        await send_long(message.channel, message, text)
                        print(
                            f"[telemetry] tool_calls={telemetry.tool_calls} "
                            f"tool_rounds={telemetry.tool_rounds} "
                            f"latency_bucket={telemetry.latency_bucket()} "
                            f"intent={tool_intent} use_tools={use_tools}"
                        )
                        break
                    messages.append({"role": "user", "content": tool_results})
                    continue

                text = next((block.text for block in response.content if hasattr(block, "text")), None)
                if text:
                    conversation_histories[message.channel.id].append({"role": "assistant", "content": text})
                    await send_long(message.channel, message, text)
                print(
                    f"[telemetry] tool_calls={telemetry.tool_calls} "
                    f"tool_rounds={telemetry.tool_rounds} "
                    f"latency_bucket={telemetry.latency_bucket()} "
                    f"intent={tool_intent} use_tools={use_tools}"
                )
                break

        except Exception as exc:
            await message.reply(f"Meow... hit a little snag: {exc}")


if __name__ == "__main__":
    discord_token = os.getenv("DISCORD_TOKEN")
    if discord_token:
        client.run(discord_token)
    else:
        print("Hold up! Couldn't find DISCORD_TOKEN in the .env file.")

import asyncio
import concurrent.futures
import os
import re
import time
import json
from collections import defaultdict

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

PINNED_MEMORY_HEADER = (
    "Pinned memory rules:\n"
    "- Treat user preferences and ongoing tasks as persistent.\n"
    "- Do not treat tool errors, timeouts, or failed calls as facts.\n"
)


def _message_text_for_summary(message: dict) -> str:
    role = message.get("role", "unknown")
    content = message.get("content", "")
    if isinstance(content, str):
        return f"{role.upper()}: {content}"
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                block = str(item.get("content", ""))
                if block.lower().startswith("tool error"):
                    continue
                parts.append(f"TOOL_RESULT: {block}")
        return f"{role.upper()}: " + "\n".join(parts)
    return f"{role.upper()}: {content}"


async def maybe_refresh_summary(channel_id: int, history: list, system_prompt: str) -> None:
    if len(history) < SUMMARY_TRIGGER_THRESHOLD:
        return

    summary_refresh_counts[channel_id] += 1
    should_force = not conversation_summaries[channel_id]
    if not should_force and summary_refresh_counts[channel_id] % SUMMARY_REFRESH_CADENCE != 0:
        return

    keep_recent = max(4, MAX_HISTORY)
    if len(history) <= keep_recent:
        return

    old_window = history[: -keep_recent]
    existing_summary = conversation_summaries[channel_id]
    serialized = "\n\n".join(_message_text_for_summary(m) for m in old_window)
    summarizer_prompt = (
        "Update the rolling channel memory.\n"
        "Return concise markdown with exactly these sections:\n"
        "## Facts\n## Preferences\n## Ongoing Tasks\n## Unresolved Questions\n"
        "Rules:\n"
        "- Never claim tool errors as facts.\n"
        "- Preserve user preferences and ongoing tasks if still relevant.\n"
        "- Prefer stable, durable information.\n"
        f"- Keep total response under {MAX_SUMMARY_LENGTH} characters."
    )

    response = await asyncio.get_running_loop().run_in_executor(
        TOOL_EXECUTOR,
        lambda: ai.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Existing summary:\n{existing_summary or '[none]'}\n\n"
                        f"{PINNED_MEMORY_HEADER}\n"
                        f"Older conversation window:\n{serialized}\n\n"
                        f"{summarizer_prompt}"
                    ),
                }
            ],
        ),
    )
    summary_text = next((block.text for block in response.content if hasattr(block, "text")), "").strip()
    if summary_text:
        conversation_summaries[channel_id] = summary_text[:MAX_SUMMARY_LENGTH]
        conversation_histories[channel_id] = history[-keep_recent:]

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
            channel_summary = conversation_summaries[message.channel.id]
            if channel_summary:
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "Rolling channel memory (long-term context):\n"
                            f"{channel_summary}\n\n"
                            f"{PINNED_MEMORY_HEADER}"
                        ),
                    },
                    *messages,
                ]
            tool_rounds = 0
            active_system_prompt = system_prompt

            if ENABLE_TWO_STAGE_RESPONDER:
                stage_a_response = ai.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=220,
                    system=STAGE_A_PROMPT,
                    messages=[{"role": "user", "content": clean_content}],
                )
                stage_a_text = next((block.text for block in stage_a_response.content if hasattr(block, "text")), "{}")
                stage_a = parse_stage_a(stage_a_text)
                active_system_prompt = build_stage_b_system_prompt(system_prompt, stage_a)

            while True:
                response = ai.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=active_system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    tool_blocks = [block for block in response.content if block.type == "tool_use"]
                    tool_rounds += 1
                    if tool_rounds > MAX_TOOL_ROUNDS:
                        text = "I hit my tool-use limit for this question. Try narrowing the request or asking for one repo/file/site at a time."
                        conversation_histories[message.channel.id].append({"role": "assistant", "content": text})
                        await send_long(message.channel, message, text)
                        break

                    messages.append({"role": "assistant", "content": response.content})
                    results = await asyncio.gather(
                        *(dispatch_tool(block.name, block.input) for block in tool_blocks)
                    )
                    tool_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                        for block, result in zip(tool_blocks, results)
                    ]
                    messages.append({"role": "user", "content": tool_results})
                    continue

                text = next((block.text for block in response.content if hasattr(block, "text")), None)
                if text:
                    conversation_histories[message.channel.id].append({"role": "assistant", "content": text})
                    await send_long(message.channel, message, text)
                break

        except Exception as exc:
            await message.reply(f"Meow... hit a little snag: {exc}")


if __name__ == "__main__":
    discord_token = os.getenv("DISCORD_TOKEN")
    if discord_token:
        client.run(discord_token)
    else:
        print("Hold up! Couldn't find DISCORD_TOKEN in the .env file.")

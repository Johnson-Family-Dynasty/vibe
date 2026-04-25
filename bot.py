import discord
import anthropic
import os
import re
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_HISTORY        = 10          # messages kept per channel
COOLDOWN_SECONDS   = 5           # per-user rate limit
MAX_RESPONSE_LEN   = 1900        # Discord limit is 2000; leave headroom

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
conversation_histories: dict[int, list] = defaultdict(list)
user_last_message:      dict[int, float] = defaultdict(float)

# ──────────────────────────────────────────────
# DISCORD + ANTHROPIC CLIENTS
# ──────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members         = True
intents.guilds          = True
intents.voice_states    = True

client = discord.Client(intents=intents)
ai     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vibe, the Discord classroom tutor and teacher assistant for an OpenTutor learning community.

OpenTutor school model and daily operations:
- Learning is organized in GitHub repositories, where curriculum, schedules, assignments, and student outputs are stored.
- Discord is the virtual classroom where daily check-ins, subject-channel work, announcements, and support happen.
- Students ask for help in the same channels where they complete assignments.
- Teachers and parents guide the process and use Vibe to draft classroom-ready materials.
- The full school model reference is available at [OpenTutor Site](https://discord-vibe-bot.vercel.app/site/).

You are running inside a Discord server. Users interact with you by @mentioning you.
You are not a server administrator and must not offer, suggest, or perform moderation/server-management actions.

Your job has two modes:

1. Student tutor
- Adapt your language to the student's age and grade — ask if unsure.
- Prefer coaching over dumping answers. Help students think, work step by step, and build understanding.
- Break hard tasks into small chunks, use simple examples, and check understanding when useful.
- If a student seems stuck, start with the next step, not a long lecture.
- Encourage effort, but stay practical and concise.

2. Teacher assistant
- Help teachers with lesson framing, announcements, assignment directions, study guides, quiz questions, rubrics, summaries, and rewording.
- Optimize for classroom usefulness: clear structure, easy copy-paste, minimal fluff.
- If a teacher asks for student-facing material, make it age-appropriate and easy to post in Discord.

Tone and style:
- Friendly, calm, lightly cool, and confident.
- Keep responses concise. A few sentences or a short list is usually enough unless depth is needed.
- Do not be overly slangy, corny, or performative.
- Use emojis sparingly.
- When you mention a website, tool, or resource by name, always include a clickable Markdown hyperlink like [Example](https://example.com).

You will receive context that includes who is speaking, their roles, their likely mode, and the channel name. Use that context to decide whether to act like a tutor or a teacher aide."""

# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────
TOOLS = []

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def preprocess_mentions(content: str, guild: discord.Guild) -> str:
    """Replace <@user_id> with @DisplayName (id: user_id) so Claude knows who is referenced."""
    def replace(match):
        user_id = int(match.group(1))
        member  = guild.get_member(user_id)
        return f"@{member.display_name} (id: {user_id})" if member else match.group(0)
    return re.sub(r'<@!?(\d+)>', replace, content)

def infer_user_mode(member_roles: list[str]) -> str:
    lowered_roles = [role.lower() for role in member_roles]

    if any(keyword in role for role in lowered_roles for keyword in ("teacher", "parent", "staff", "mod", "moderator")):
        return "teacher/staff"

    if any(keyword in role for role in lowered_roles for keyword in ("student", "learner", "kid")):
        return "student"

    return "unknown"


async def send_long(channel: discord.abc.Messageable, reply_to: discord.Message, text: str) -> None:
    """Send a message, splitting into chunks if it exceeds Discord's limit."""
    if len(text) <= MAX_RESPONSE_LEN:
        await reply_to.reply(text)
        return
    chunks = [text[i:i + MAX_RESPONSE_LEN] for i in range(0, len(text), MAX_RESPONSE_LEN)]
    for i, chunk in enumerate(chunks):
        if i == 0:
            await reply_to.reply(chunk)
        else:
            await channel.send(chunk)


# ──────────────────────────────────────────────
# DISCORD EVENTS
# ──────────────────────────────────────────────

@client.event
async def on_ready():
    print(f"Cool cat {client.user} has logged in and is ready to vibe!")


@client.event
async def on_message(message: discord.Message):
    # Ignore own messages
    if message.author == client.user:
        return

    # Only respond when @mentioned
    if not client.user.mentioned_in(message):
        return

    # Strip the @mention from the message
    clean_content = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not clean_content:
        clean_content = "Hey Vibe, what's up?"

    # ── Built-in commands ──
    lower = clean_content.lower()

    if lower == "!help":
        await message.reply(
            "Hey! Here's what I can do:\n"
            "📚 Tutor students with step-by-step help in middle school subjects\n"
            "🧑‍🏫 Help teachers with announcements, lesson directions, quizzes, rubrics, and summaries\n"
            "🏫 Stay aligned with the OpenTutor school model and daily workflow\n"
            "💬 @mention me with any question to chat\n"
            "`!reset` — clear our conversation history\n"
            "`!help` — show this message"
        )
        return

    if lower == "!reset":
        conversation_histories[message.channel.id] = []
        await message.reply("Fresh start! Conversation history cleared. 🐱")
        return

    # ── Rate limiting ──
    now = time.time()
    if now - user_last_message[message.author.id] < COOLDOWN_SECONDS:
        await message.reply(f"Slow down a bit — one message every {COOLDOWN_SECONDS}s, please.")
        return
    user_last_message[message.author.id] = now

    # ── Preprocess mentions so Claude sees names, not IDs ──
    if message.guild:
        clean_content = preprocess_mentions(clean_content, message.guild)

    # ── Inject user context ──
    roles = getattr(message.author, "roles", [])
    member_roles = [r.name for r in roles if r.name != "@everyone"]
    likely_mode = infer_user_mode(member_roles)
    channel_name = getattr(message.channel, "name", "direct-message")
    user_context = (
        f"[Speaking with: {message.author.display_name} "
        f"(id: {message.author.id}), "
        f"Roles: {', '.join(member_roles) or 'none'}, "
        f"Likely mode: {likely_mode}, "
        f"Channel: #{channel_name}]\n\n"
    )
    full_content = user_context + clean_content

    # ── Build history ──
    history = conversation_histories[message.channel.id]
    history.append({"role": "user", "content": full_content})
    if len(history) > MAX_HISTORY * 2:          # *2 because user+assistant pairs
        history = history[-(MAX_HISTORY * 2):]
        conversation_histories[message.channel.id] = history

    async with message.channel.typing():
        try:
            messages = list(history)

            while True:
                response = ai.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )

                if response.stop_reason == "tool_use":
                    # Tools are disabled for this tutor profile, so continue as plain chat.
                    text = "I can only help with student tutoring and teacher-assistant support."
                    conversation_histories[message.channel.id].append(
                        {"role": "assistant", "content": text}
                    )
                    await send_long(message.channel, message, text)
                    break

                else:
                    text = next((b.text for b in response.content if hasattr(b, "text")), None)
                    if text:
                        # Save assistant reply to history
                        conversation_histories[message.channel.id].append(
                            {"role": "assistant", "content": text}
                        )
                        await send_long(message.channel, message, text)
                    break

        except Exception as e:
            await message.reply(f"Meow... hit a little snag: {e}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    discord_token = os.getenv("DISCORD_TOKEN")
    if discord_token:
        client.run(discord_token)
    else:
        print("Hold up! Couldn't find DISCORD_TOKEN in the .env file.")

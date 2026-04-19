import discord
import anthropic
import os
import re
import json
import time
from datetime import timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MAX_HISTORY        = 10          # messages kept per channel
COOLDOWN_SECONDS   = 5           # per-user rate limit
MAX_RESPONSE_LEN   = 1900        # Discord limit is 2000; leave headroom
LOG_CHANNEL_NAME   = "bot-logs"  # channel to post admin-action logs
ADMIN_ROLE_NAME    = "Admin"     # role required for server-changing tools
ADMIN_TOOLS = {
    "kick_member", "ban_member", "unban_member", "timeout_member",
    "purge_messages", "create_channel", "delete_channel", "rename_channel",
    "set_slowmode", "create_role", "assign_role", "remove_role"
}

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
SYSTEM_PROMPT = """You are Vibe, the Discord classroom tutor, teacher assistant, and server admin for a homeschooling community focused on middle school education.

You are running inside a Discord server. Users interact with you by @mentioning you. You can use tools to manage the server. If an authorized user asks you to take a server action, use the tool and actually do it instead of only saying you will.

Your job has three modes:

1. Student tutor
- Teach clearly at a middle school level without sounding babyish.
- Prefer coaching over dumping answers. Help students think, work step by step, and build understanding.
- Break hard tasks into small chunks, use simple examples, and check understanding when useful.
- If a student seems stuck, start with the next step, not a long lecture.
- Encourage effort, but stay practical and concise.

2. Teacher assistant
- Help teachers with lesson framing, announcements, assignment directions, study guides, quiz questions, rubrics, summaries, and rewording.
- Optimize for classroom usefulness: clear structure, easy copy-paste, minimal fluff.
- If a teacher asks for student-facing material, make it age-appropriate and easy to post in Discord.

3. Server admin
- Handle moderation and server organization responsibly.
- Only perform server-changing actions for users who have the Admin role.
- For destructive or high-risk actions, be explicit about what you did.
- If a moderation request is ambiguous or risky, ask one short clarifying question before acting.

Tone and style:
- Friendly, calm, lightly cool, and confident.
- Keep responses concise. A few sentences or a short list is usually enough unless depth is needed.
- Do not be overly slangy, corny, or performative.
- Use emojis sparingly. Avoid overusing cat emojis.
- When you mention a website, tool, or resource by name, always include a clickable Markdown hyperlink like [Example](https://example.com).

You will receive context that includes who is speaking, their roles, their likely mode, and the channel name. Use that context to decide whether to act like a tutor, a teacher aide, or an admin."""

# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────
TOOLS = [
    {
        "name": "kick_member",
        "description": "Kick a member from the server",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Discord user ID"},
                "reason":  {"type": "string", "description": "Reason for the kick"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "ban_member",
        "description": "Ban a member from the server",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id":              {"type": "string",  "description": "Discord user ID"},
                "reason":               {"type": "string",  "description": "Reason for the ban"},
                "delete_message_days":  {"type": "integer", "description": "Days of messages to delete (0-7)", "default": 0}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "unban_member",
        "description": "Unban a previously banned member",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Discord user ID to unban"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "timeout_member",
        "description": "Temporarily mute/timeout a member",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id":          {"type": "string",  "description": "Discord user ID"},
                "duration_minutes": {"type": "integer", "description": "Timeout duration in minutes"},
                "reason":           {"type": "string",  "description": "Reason for the timeout"}
            },
            "required": ["user_id", "duration_minutes"]
        }
    },
    {
        "name": "purge_messages",
        "description": "Delete recent messages from the current channel",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of messages to delete (max 100)"}
            },
            "required": ["count"]
        }
    },
    {
        "name": "create_channel",
        "description": "Create a new text or voice channel",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Channel name"},
                "type": {"type": "string", "enum": ["text", "voice"], "description": "Channel type"}
            },
            "required": ["name", "type"]
        }
    },
    {
        "name": "delete_channel",
        "description": "Delete a channel by name",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_name": {"type": "string", "description": "Name of the channel to delete"}
            },
            "required": ["channel_name"]
        }
    },
    {
        "name": "rename_channel",
        "description": "Rename an existing channel",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_name": {"type": "string", "description": "Current channel name"},
                "new_name":     {"type": "string", "description": "New channel name"}
            },
            "required": ["channel_name", "new_name"]
        }
    },
    {
        "name": "set_slowmode",
        "description": "Set slowmode delay on the current channel (0 to disable)",
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Slowmode delay in seconds"}
            },
            "required": ["seconds"]
        }
    },
    {
        "name": "create_role",
        "description": "Create a new role in the server",
        "input_schema": {
            "type": "object",
            "properties": {
                "name":  {"type": "string", "description": "Role name"},
                "color": {"type": "string", "description": "Hex color like #ff5733 (optional)"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "assign_role",
        "description": "Assign a role to a member",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id":   {"type": "string", "description": "Discord user ID"},
                "role_name": {"type": "string", "description": "Role name to assign"}
            },
            "required": ["user_id", "role_name"]
        }
    },
    {
        "name": "remove_role",
        "description": "Remove a role from a member",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id":   {"type": "string", "description": "Discord user ID"},
                "role_name": {"type": "string", "description": "Role name to remove"}
            },
            "required": ["user_id", "role_name"]
        }
    },
    {
        "name": "get_server_info",
        "description": "Get information about the current server: name, member count, channels, and roles",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "get_member_info",
        "description": "Get information about a specific member: name, roles, join date",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Discord user ID"}
            },
            "required": ["user_id"]
        }
    }
]

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


def has_admin_role(member: discord.Member) -> bool:
    return any(r.name == ADMIN_ROLE_NAME for r in member.roles)


def infer_user_mode(member_roles: list[str]) -> str:
    lowered_roles = [role.lower() for role in member_roles]

    if any("admin" in role for role in lowered_roles):
        return "admin"

    if any(keyword in role for role in lowered_roles for keyword in ("teacher", "parent", "staff", "mod", "moderator")):
        return "teacher/staff"

    if any(keyword in role for role in lowered_roles for keyword in ("student", "learner", "kid")):
        return "student"

    return "unknown"


async def log_action(guild: discord.Guild, action: str, author: discord.Member) -> None:
    log_channel = discord.utils.get(guild.text_channels, name=LOG_CHANNEL_NAME)
    if log_channel:
        await log_channel.send(
            f"🔧 **Admin Action** | `{action}` — requested by **{author.display_name}**"
        )


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
# TOOL EXECUTOR
# ──────────────────────────────────────────────

async def execute_tool(tool_name: str, tool_input: dict, message: discord.Message) -> str:
    guild = message.guild

    # ── Permission check for destructive tools ──
    if tool_name in ADMIN_TOOLS and not has_admin_role(message.author):
        return f"Permission denied — only members with the '{ADMIN_ROLE_NAME}' role can use `{tool_name}`."

    try:
        result = await _run_tool(tool_name, tool_input, message, guild)
        # Log admin actions
        if tool_name in ADMIN_TOOLS:
            await log_action(guild, f"{tool_name}({json.dumps(tool_input)})", message.author)
        return result

    except discord.Forbidden:
        return "I don't have permission to do that."
    except Exception as e:
        return f"Error executing {tool_name}: {e}"


async def _run_tool(tool_name: str, tool_input: dict, message: discord.Message, guild: discord.Guild) -> str:

    if tool_name == "kick_member":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        await member.kick(reason=tool_input.get("reason", "No reason provided"))
        return f"Kicked {member.display_name}."

    elif tool_name == "ban_member":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        await member.ban(
            reason=tool_input.get("reason", "No reason provided"),
            delete_message_days=tool_input.get("delete_message_days", 0)
        )
        return f"Banned {member.display_name}."

    elif tool_name == "unban_member":
        user = await client.fetch_user(int(tool_input["user_id"]))
        await guild.unban(user)
        return f"Unbanned {user.name}."

    elif tool_name == "timeout_member":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        await member.timeout(
            timedelta(minutes=tool_input["duration_minutes"]),
            reason=tool_input.get("reason", "No reason provided")
        )
        return f"Timed out {member.display_name} for {tool_input['duration_minutes']} minutes."

    elif tool_name == "purge_messages":
        count   = min(tool_input["count"], 100)
        deleted = await message.channel.purge(limit=count)
        return f"Deleted {len(deleted)} messages."

    elif tool_name == "create_channel":
        if tool_input["type"] == "voice":
            channel = await guild.create_voice_channel(tool_input["name"])
        else:
            channel = await guild.create_text_channel(tool_input["name"])
        return f"Created {tool_input['type']} channel #{channel.name}."

    elif tool_name == "delete_channel":
        channel = discord.utils.get(guild.channels, name=tool_input["channel_name"])
        if not channel:
            return f"Channel '{tool_input['channel_name']}' not found."
        await channel.delete()
        return f"Deleted channel #{tool_input['channel_name']}."

    elif tool_name == "rename_channel":
        channel = discord.utils.get(guild.channels, name=tool_input["channel_name"])
        if not channel:
            return f"Channel '{tool_input['channel_name']}' not found."
        old_name = channel.name
        await channel.edit(name=tool_input["new_name"])
        return f"Renamed #{old_name} to #{tool_input['new_name']}."

    elif tool_name == "set_slowmode":
        await message.channel.edit(slowmode_delay=tool_input["seconds"])
        return "Slowmode disabled." if tool_input["seconds"] == 0 else f"Slowmode set to {tool_input['seconds']} seconds."

    elif tool_name == "create_role":
        color_str = tool_input.get("color", "000000").lstrip("#")
        color     = discord.Color(int(color_str, 16)) if color_str else discord.Color.default()
        role      = await guild.create_role(name=tool_input["name"], color=color)
        return f"Created role '{role.name}'."

    elif tool_name == "assign_role":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        role = discord.utils.get(guild.roles, name=tool_input["role_name"])
        if not role:
            return f"Role '{tool_input['role_name']}' not found."
        await member.add_roles(role)
        return f"Assigned '{role.name}' to {member.display_name}."

    elif tool_name == "remove_role":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        role = discord.utils.get(guild.roles, name=tool_input["role_name"])
        if not role:
            return f"Role '{tool_input['role_name']}' not found."
        await member.remove_roles(role)
        return f"Removed '{role.name}' from {member.display_name}."

    elif tool_name == "get_server_info":
        return json.dumps({
            "name":         guild.name,
            "member_count": guild.member_count,
            "channels":     [{"name": c.name, "type": str(c.type)} for c in guild.channels],
            "roles":        [r.name for r in guild.roles]
        })

    elif tool_name == "get_member_info":
        member = guild.get_member(int(tool_input["user_id"]))
        if not member:
            return "Member not found."
        return json.dumps({
            "name":      member.display_name,
            "id":        str(member.id),
            "roles":     [r.name for r in member.roles],
            "joined_at": str(member.joined_at)
        })

    return f"Unknown tool: {tool_name}"


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
            "🛠️ Manage the server (admins only — moderation, channels, roles, slowmode, and cleanup)\n"
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
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            result = await execute_tool(block.name, block.input, message)
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result
                            })
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user",      "content": tool_results})

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

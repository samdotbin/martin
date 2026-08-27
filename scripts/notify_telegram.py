"""
scripts/notify_telegram.py — pushes each contributor's shard-status summary
to a shared Telegram chat, on a much faster cadence than the GitHub push
(every 5 min) can reasonably go without spamming the repo with commits for
unchanged files. Read-heavy, no state, stdlib-only (matching
push_to_github.py's approach — no new dependency).

Needs a bot token + a destination chat id, both created ONCE by the project
owner and dropped as plain text in the shared Drive folder (same
zero-setup pattern as github_token.txt) so every contributor's session
picks them up automatically:
  - telegram_bot_token.txt: the bot token from @BotFather (Telegram ->
    @BotFather -> /newbot, or /mybots -> API Token for an existing one).
    NEVER paste this into chat/logs/commits -- treat it like a password.
  - telegram_chat_id.txt: the numeric chat id messages get sent to. Add
    the bot to a group with everyone in it, send it one message there,
    then GET https://api.telegram.org/bot<token>/getUpdates and read the
    "chat":{"id": ...} field of the response -- group ids are negative.

Both files being absent just means the feature isn't set up yet -- callers
treat that as "skip silently," not an error, so contributors whose owner
hasn't configured this see no difference in behavior.
"""
import json
import os
import urllib.error
import urllib.request


def send_message(bot_token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        print(f"telegram send failed: {e.code} {e.read().decode(errors='replace')}")
        return False


def _read_text(shared_folder: str, filename: str) -> str:
    path = os.path.join(shared_folder, filename)
    with open(path) as f:
        return f.read().strip()


def read_telegram_config(shared_folder: str):
    """Returns (bot_token, chat_id), or (None, None) if not configured yet
    -- deliberately NOT an exception, since this is an opt-in feature that
    should stay silent (not break the training loop) until the owner sets
    it up."""
    try:
        token = _read_text(shared_folder, "telegram_bot_token.txt")
        chat_id = _read_text(shared_folder, "telegram_chat_id.txt")
        return (token, chat_id) if token and chat_id else (None, None)
    except FileNotFoundError:
        return None, None

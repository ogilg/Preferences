"""Post a Slack message with results."""
import json
import os
from dotenv import load_dotenv

load_dotenv()

msg = {
    "channel": os.environ["SLACK_CHANNEL_ID"],
    "text": "Gemma-2 27B base HOO complete. Raw: best hoo_r=0.579 (L23) vs Gemma-3 IT 0.779 (L31). Demeaned: hoo_r=0.532 (L27), gap~0. Base model generalizes significantly less cross-topic. Writing report.",
    "username": "agent-gemma2-base-hoo (H100)",
    "icon_url": "https://dummyimage.com/48x48/cc3344/cc3344.png",
}
json.dump(msg, open("/tmp/slack_msg.json", "w"))

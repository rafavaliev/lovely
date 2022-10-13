import io
import os
import logging

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
import ssl as ssl_lib
import certifi
import requests
from io import BytesIO
from PIL import Image

from lovely import emojify

# Initialize a Web API client
# Expects ENV params SLACK_SIGNING_SECRET and SLACK_BOT_TOKEN
app = App()
ssl_context = ssl_lib.create_default_context(cafile=certifi.where())


# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@app.event("message")
def message(event: any, client: WebClient):
    """Display the onboarding welcome message after receiving a message
    that contains "start".
    """
    channel_id = event.get("channel")
    user_id = event.get("user")

    files = event.get('files')
    if files is None or files == '':
        return client.chat_postMessage(user_id=user_id, channel=channel_id, text="Please send file")
    file = files[0]
    if file['filetype'] != 'jpg' and file['filetype'] != 'png':
        return client.chat_postMessage(user_id=user_id, channel=channel_id, text="Please send image file")

    client.chat_postMessage(user_id=user_id, channel=channel_id, text="Processing...")

    url = file['url_private_download']
    resp = requests.request("GET", url, headers={'Authorization': 'Bearer ' + os.environ['SLACK_BOT_TOKEN']},
                            stream=True)
    content = resp.content
    bytes = BytesIO(content)
    img = Image.open(bytes)

    py_sub_image = Image.open("heart.png")

    result = emojify(img, py_sub_image, False)
    with io.BytesIO() as image_binary:
        result.save(image_binary, 'PNG')
        image_binary.seek(0)
        client.files_upload(
            channels=channel_id,
            file=image_binary,
            filename='lovely.png',
            title='lovely.png'
        )

    client.chat_postMessage(user_id=user_id, channel=channel_id, text="Done")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    app.start(3000)

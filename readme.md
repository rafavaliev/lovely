# Lovely

Lovely is a python library that inserts emojis to images with a face.

### Requirements

1. Python 3
2. Run `pip install -r requirements.txt` (opencv, numpy, optional slack sdk)

### How to

Default usage: script will output an image with `-lovely` suffix and ❤️ emojis: `python main.py path/to/image.jpeg`

To customized emojis pass style parameter: [`lovely`, `rage`, `sad`] or `all`(will generate images for all styles).
Example: `python main.py path/to/image.jpeg rage`

Choosing custom emoji is not possible right now.

### Why insert emoji image and not just ❤️?

I tried to use `Pillow` to write emojis, but it doesn't work as expected. If you want to fix it, just do it.

### Slack part

1. [Slack app tutorial](https://github.com/slackapi/python-slack-sdk/tree/main/tutorial)

To add an emoji to a workspace one needs `admin.emoji.add` access which is part of `admin.teams:write` scope. It's
highly unlikely that anyone will install a Slack app with such scope for just emoji, so the implementation looks like
this:

1. Anyone installs the app to a workspace
2. A user who wants to add a `lovely` emojis to a workspace sends a message to the app with a photo(s).
3. The app responds with a message to the user with `lovely` emojis on the input photo(s(), which they can use later
   when adding an emoji.
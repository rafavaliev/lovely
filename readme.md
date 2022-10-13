# Lovely

`Lovely` is a python library that inserts emojis to images with a face. In my experience this improves team engagement.
An example of usage in slack:
![slack](example-in-slack.png)

This library also used face detection via OpenCV, so emojis don't cover face.

## Example of work

![not-lovely](example.png) ->
![lovely](example-lovely.png)

### Requirements

1. Python 3
2. Run `pip install -r requirements.txt` (opencv, numpy, optional slack sdk). If you don't want to install/use slack
   sdk, you can remove it from requirements.txt.

### How to

Default usage: script will output an image with `-lovely` suffix and ❤️ emojis: `python main.py path/to/image.jpeg`

[WIP] To customize emojis pass style parameter: [`lovely`, `rage`, `sad`] or `all`(will generate images for all styles).
Example: `python main.py path/to/image.jpeg rage`

### Why insert emoji image and not just ❤️?

I tried to use `Pillow` to write emojis directly, but it doesn't work as expected. If you want to fix it, just do it.

### Slack part

To add an emoji to a slack workspace one needs `admin.emoji.add` access which is part of `admin.teams:write` scope. It's
highly unlikely that anyone will install a Slack app with such scope for just emoji, so the implementation looks like
this:

1. Anyone installs the app to a workspace
2. A user who wants to add a emojis to a workspace sends a message to the app with a photo(s).
3. The app responds with a message to the user with `lovely` emojis on the input photo(s(), which they can use later
   when adding an emoji.
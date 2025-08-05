# Requirements

*tested on macOS Sequoia*

- [Blackhole Audio Router](https://existential.audio/blackhole/)
- Python 3.13.5

# What

Takes .qt .mov MacOS screen recording and transcribes it to text, splits the dialog by speaker.

# How

- Clone repo && cd into it
- Run `chmod a+x setup.sh` && run `./setup.sh`
- Install blackhole and configure composite devices, [example](https://www.youtube.com/watch?v=DDfUJiyp_Vw)
- Set your blackhole device as input in built-in recorder - (CMD+Shift+5)
- Record your meeting with built-in recorder - (CMD+Shift+5)
- `python vtt.py ~/Desktop/ScreenRecording.file --method mfcc`
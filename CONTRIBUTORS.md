# Vosk-Speech-to-Text Dictation with Command Injection

This project uses [Vosk](https://github.com/alphacep/vosk-api) for speech-to-text and integrates a GUI built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui). The program supports both dictation and voice commands like "move left" or "stop listening."

## Features
- Live speech-to-text transcription.
- Voice command execution (e.g., keyboard actions, GUI updates).

## Current Issue
Commands are detected (e.g., "move left" and "stop listening") and parsed into JSON, but they are not executed. The issue likely lies in:
1. Communication between the main script and the GUI module.
2. Command execution (e.g., keyboard actions, dictation stopping).

### Logs
Here are some example logs showing the problem:
```
Partial: move left
Detected command: move left
{"type": "command", "action": "cmd_move_left", "params": {"key": null, "transcribed_text": ["Some text"]}}
```

Despite the detected command, no action is performed.


## How to Contribute
1. Clone the repository.
2. Run `live_gigaspeech_dictation_v31.3.py` to reproduce the issue.
3. Investigate the command handling flow:
   - Main script: `execute_command` and `execute_computer_command`.
   - GUI module: `handle_command`.
4. Submit a pull request with your fix.

^^^^^^^^^^^^^^^ AI said it was there... all i know is injecting isn't working. tokenizing semi worked but got too complex. i want simple and efficient. not complex to do a simple task. (unless complex needed for efficiency. hence audio based things within script)


Side Note: trust me this isn't version 31. it is an evovled script 12 from small model, reused 11 from medium model which then went into 60 version on the big model. it actually was 1 script but it got too big and i thought by seperating it. would prevent braking. also allow vosk vs gui to have less workload per script. then with .py to exe via nuitka ... it broke. also once adding the exe compatibility py broke. so i turned gui into a module. main script imports it. as you've read I don't know how to code. I do IT so used those skills with AI to put this together. also I HATE TOKENIZATION. the commands worked when it was seperate and finding path. it was BEAUTIFUL. but once I turned it into a module.... it broke everything. there are a few features in the gui I don't know how to do but have the idea. so they are in there.


FUTURE PLANS the vunlerability aspect. the reason MIT license is to prevent being screwed over. it's my idea and we've seen small projects be a big thing and blow up. so it's a legal protection. also to keep the credit so people know it's me that created this. with github community help. hell i'll even do a readme shout out to people that fix the commands. there are a few features in the gui I don't know how to do but have the idea. so they are in there. also others that i wouldn't know how to go about it but would like to impliment them. i just don't want to scare others with abundant features. we can put an advanced button to open a seperate window if need be.

THE "Hidden Agenda". yes this will still be Open Source always under MIT... unless i need to put it under apache 2.0 i doubt tho. sounds like a pain. HOWEVER there will be an EXE on my patereon that will cost $. the reason. either you learn or you pay. moral dilema, how valueable is your time. so EXE will or at least attempt to be subscription based. every 30days. and as i said earlier 3 models. small medium and gigaspeech. i'm thinking $20 $50 $80 (i was going to go lower but everyone IRL told me not to. and why to charge more) dragon is $150 so i figured i have no certs. so cut it in half. hence $80.
BOTH THE OPEN SOURCE AND PAID WILL HAVE THE SAME FEATURES. furher insentivising "either you learn or pay". convienence or technical skills. most people aren't that tech savy also i have no job. haven't been able to get one for yrs. so patreon will be my source of income. so yea that's about it. think of it like how Elon Musk does his patents. he does it but allows anyone to use as they see fit. MIT is my version of that.

cautionary measure. i have aspergers so if i come off as an asshole... probably not my intent unless i'm pissed. so yea i word things weirdly. just ask for clarification i'll be more than happy to do so. the only things that actually tick me off is others lack of understanding me or others, and implying/assuming others know what i'm thinking or my intent was. (soo many do so and a lot use logical fallacies). otherwise i've been told "i'm chill as if i'm high all the time" idk if that's a compliment or insult. lol 

LASTLY. if need be. I do have every script i've created for this project on my computer. if need be i can create a CONTRIUBUTORS folder... so you can see every itteration an the path that i took to get here. aka i didn't txt document. only mental notes but those scripts are my "Documenation".

Title
Sign2Speech: Real‑Time Sign Language Translator (Sign→Text/Speech and Text→Sign)

Badges

Python 3.x

License: MIT

Platforms: Windows/Linux

Keywords: Assistive‑Tech, Computer‑Vision, Deep‑Learning

Overview
Sign2Speech is a low‑cost, webcam‑based assistive system that enables two‑way communication between signers and non‑signers. It recognizes hand gestures in real time with a DenseNet121 CNN and temporal smoothing, converts recognized text to speech offline (pyttsx3), and renders Text→Sign via animated GIFs with captions. The Tkinter GUI includes accessibility features (keyboard controls, large fonts, high‑contrast), a dataset capture wizard, and an admin retraining flow to add new signs or regional variants.

Key features

Real‑time Sign→Text with DenseNet121 + majority‑vote smoothing for stable predictions

Offline TTS via pyttsx3 (no internet required)

Text→Sign animations with adjustable speed and captions

Accessible Tkinter GUI: keyboard shortcuts, large fonts, high‑contrast theme

Dataset capture wizard and admin retraining flow

Evaluation suite: per‑class precision/recall/F1, confusion matrices, latency notes

Demo

Video (recommended): [Demo link]

GIFs:

Live Sign→Text recognition

TTS playback

Text→Sign animation

Screenshots:

GUI home (Sign2Text/Text2Sign)

Classification report and confusion matrix

Architecture

OpenCV camera → preprocessing → DenseNet121 inference → temporal smoothing → text output/TTS

Text→Sign: text normalization → GIF mapping → captioned playback

Admin: dataset capture → labeling/auto‑split → fine‑tune → export versioned model

Tech stack

Python, TensorFlow/Keras (DenseNet121), OpenCV, Tkinter

pyttsx3 (offline TTS), pyenchant (suggestions)

NumPy, scikit‑learn (metrics)

Optional: gTTS alternative, PIL, YAML/JSON config

Getting started

Prerequisites

Python 3.9+ (tested on 3.9/3.10)

OS: Windows 10/11 or Ubuntu 22.04+

Webcam

Install

Clone the repo:
git clone https://github.com/<username>/sign2speech
cd sign2speech

Create environment:
pip install -r requirements.txt
or
conda env create -f environment.yml
conda activate sign2speech

Assets:

Place GIFs for Text→Sign in assets/gifs/ (A–Z and common words)

Optional: place sample datasets under data/

Run

Start GUI:
python app.py

Controls:

Space: toggle recognition

Backspace: delete last character

C: clear text

S: speak text (pyttsx3)

F1/F2: slower/faster GIF playback

Configuration:

Edit config/config.yaml for ROI, thresholds, voting window, TTS rate

Usage

Sign→Text

Select Sign2Text mode

Keep hand in ROI; pause slightly for consistent predictions

Text pane accumulates letters; suggestions appear under the last token

Press S to speak the current text via offline TTS

Text→Sign

Switch to Text2Sign mode

Type or paste text; mapped signs animate as GIFs with captions

Adjust speed with playback controls

Dataset and retraining (admin)

Open Tools → Dataset Capture to record labeled samples

Auto‑split and augment via the wizard

Start fine‑tuning (Tools → Retrain); new model is saved to models/<version>

Update config to point GUI to the latest model

Evaluation

Run evaluation script:
python tools/eval.py --model models/<version>.h5 --data data/val

Outputs:

classification_report.csv

confusion_matrix.png

latency_stats.json

Recommended targets:

Weighted F1 ≥ 0.90 under well‑lit conditions

Median per‑frame latency suited to live usage on CPU

Troubleshooting

Camera not found: verify device index in config; close other apps using the webcam

Washed images/poor predictions: adjust ROI, enable adaptive threshold/blur

TTS silent: ensure pyttsx3 engine output device is available; try a different voice

GIF missing: ensure assets/gifs/<token>.gif exists; fallback mapping in config

Project structure

app.py # Tkinter GUI entry

src/

capture.py # OpenCV capture and ROI

preprocess.py # resize/normalize/threshold ops

model_loader.py # DenseNet121 load/wrap

inference.py # predict + smoothing buffer

text_utils.py # suggestions, token ops

tts.py # pyttsx3 wrapper

gif_player.py # Text→Sign animator

tools/

dataset_capture.py # wizard for labeled samples

train.py # fine‑tune scripts

eval.py # metrics + confusion matrix

assets/

gifs/ # sign animations

icons/ # UI icons

config/

config.yaml # thresholds, ROI, paths

<versioned models>

Multilingual signs and word‑level modeling

Transformers for continuous sign sequences

Mobile/TFLite deployment

Cloud API mode with Web UI

Expanded accessibility (screen reader labels, full keyboard navigation)

Contributing

Fork → feature branch → PR

Style: black/flake8

Add unit tests for new modules

Open issues with reproducible steps and logs

License

MIT (see LICENSE)

Acknowledgments

OpenCV, TensorFlow/Keras, pyttsx3, pyenchant

Contact

Maintainer: Samiullah A Hulkoti - samiullah.hulkoti@gmail.com - LinkedIn:https://www.linkedin.com/in/samiullahhulkoti


Release notes (optional)

v1.0.0: Initial public release with Sign→Text, Text→Sign, offline TTS, retraining, and evaluation suite

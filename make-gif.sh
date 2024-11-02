#!/bin/sh

rm -rf data
nim r -d:release main.nim
ffmpeg -y -i data/weights-%03d.ppm data/out.mp4
ffmpeg -y -i data/out.mp4 -vf palettegen data/palette.png
ffmpeg -y -i data/out.mp4 -i data/palette.png -filter_complex paletteuse out.gif

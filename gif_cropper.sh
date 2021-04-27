#!/usr/bin/env bash

# See: https://stackoverflow.com/q/965053/1316276.
# for f in *.gif; do gifsicle --crop 0,0-250,266 --output ${f%.*}_cropped.gif ${f}; done
for f in *.gif; do gifsicle --crop 250,0-500,266 --output ${f%.*}_cropped.gif ${f}; done
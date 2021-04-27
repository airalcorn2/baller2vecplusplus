#!/usr/bin/env bash

# See: https://deparkes.co.uk/2015/04/30/batch-crop-images-with-imagemagick/.
mogrify -crop 250x266+250+0 *.png

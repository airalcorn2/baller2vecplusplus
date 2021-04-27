#!/usr/bin/env bash

# See: https://stackoverflow.com/a/30932152/1316276.
# Separate frames of train.gif.
convert train.gif -coalesce a-%04d.gif
# Separate frames of baller2vec.gif.
convert baller2vec.gif -coalesce b-%04d.gif
# Separate frames of baller2vec++.gif.
convert baller2vec++.gif -coalesce c-%04d.gif
# Append frames side-by-side.
for f in a-*.gif; do convert $f -size 10x xc:black ${f/a/b} -size 10x xc:black ${f/a/c} +append ${f}; done
# Rejoin frames.
convert -loop 0 -delay 40 a-*.gif result.gif
rm a-*.gif b-*.gif c-*.gif
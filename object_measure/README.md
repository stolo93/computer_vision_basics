### Object measure
#### Measure object in the image provided
##### Built with `opencv`

Objects in the picture are measured with respect to the reference object,
which is the left-most object in the picture.

Run:
> python3.11 object_measure.py --width 2.24 --image res/example.jpg

where, 2.24 is the width of the 50 cent euro coin, which I used as the reference object.

There are slight inaccuracies in the measurements because the camera was not at right angle.

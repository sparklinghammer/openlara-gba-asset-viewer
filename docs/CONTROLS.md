# Controls

| Input | Action |
|---|---|
| Tap Left / Right | start spinning that way; the same tap again stops it |
| Hold Left / Right | turn by hand, after 8 ticks (about 0.27 s at 30 Hz) |
| Up / Down | pitch |
| Select + Left / Right | roll |
| Select + Up | zoom in |
| Select + Down | zoom out |
| L / R | previous / next model |
| Start | next animation |
| Select + Start | previous animation |
| A | play / pause |
| Select + A | pause, then step one frame |
| B | face the camera again (yaw 180°); zoom and vertical offset reset |
| Select alone | show / hide the control help |
| Select + L | move the model up the screen |
| Select + R | move the model down the screen |

Sources stay separate in the backend because OpenLara's models belong to
individual levels. The interface merges them into one deduplicated catalogue and
reloads the right source on its own.

Animations are listed by their native `STATE` id. No `IDLE`, `WALK` or `ATTACK`
name is invented for them, because the data does not carry one.

Models load with a yaw of 180° so that they face the camera.

Changing clip with `Start` or `Select + Start` leaves every view setting alone,
so a spin already running keeps running.

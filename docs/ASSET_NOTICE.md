# Assets

**This repository redistributes no Tomb Raider data.** It holds source code,
scripts and documentation, and nothing else.

Viewing your own models needs none of the files below. They are only required if
you want to point the viewer at Tomb Raider's own models instead, which means
supplying them from your own copy of the game:

- `TITLE.PKD`, `GYM.PKD`, `LEVEL1.PKD`, `LEVEL2.PKD` from the OpenLara checkout;
- `TITLE.SCR` and `TRACKS.AD4` from the same place;
- a local `TR1.WAD` for the full campaign;
- the `.gba` ROMs you build, which contain whatever you fed the converter.

The 191 type names the viewer displays come from the `ITEM_TYPES(E)` macro that
OpenLara already makes public. They become strings through the C++ preprocessor
at build time; none of them is copied into this repository by hand.

`models/cheems.glb` is a third-party model kept so that a fresh clone builds and
runs. No licence to it is granted here. The same caution applies to anything you
add: check the rights before publishing a ROM built from it.

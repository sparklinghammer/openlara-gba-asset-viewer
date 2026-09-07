# OpenLara GBA Asset Viewer

Put your own 3D models on a Game Boy Advance. Drop a `.glb`, `.dae`, `.obj` or
`.fbx` in a folder, run one script, and you get a `.gba` ROM that shows them:
textured, animated, and you can spin them around on real hardware.

![The viewer running the demo model](media/cheems.gif)

It's built on the GBA port of [OpenLara](https://github.com/XProger/OpenLara), XProger's
open-source Tomb Raider engine.

It never touches your OpenLara clone. The sources it needs get copied into a
work folder, the viewer goes on top, and everything compiles there. Your clone
stays exactly as you cloned it.

Disclaimer: AI was used to help put in phrases the technical parts of this readme file.

---

## Controls

| Input | Action |
|---|---|
| Tap Left / Right | start spinning that way; tap again to stop <br>![demo spinning](media/spinning.gif) |
| Hold Left / Right | turn by hand, after about a quarter second <br>![demo turn](media/turn.gif) |
| Up / Down | pitch <br>![demo pitch](media/pitch.gif) |
| Select + Left / Right | roll <br>![demo roll](media/roll.gif) |
| Select + Up / Down | zoom in / out <br>![demo zoom](media/zoom.gif) |
| L / R | previous / next model <br>![demo next](media/next.gif) |
| Start<br>Select + Start | next animation<br>previous animation <br>![demo anim-next](media/anim-next.gif) |
| A | play / pause <br>![demo pause](media/pause.gif) |
| Select + A | pause, then step one frame <br>![demo frames](media/frames.gif) |
| B | face the camera again; zoom and vertical offset reset <br>![demo reset](media/reset.gif) |
| Select alone | show / hide the control help <br>![demo help](media/help.gif) |
| Select + L / R | move the model up / down the screen <br>![demo move](media/move.gif) |

Animations show their native `STATE` id. I don't invent `IDLE` or `WALK` names
for them, because the data doesn't have any.

---

## What you need

| | | |
|---|---|---|
| **OpenLara** | a clone of <https://github.com/XProger/OpenLara> | read-only, nothing gets written to it |
| **devkitPro** | the `gba-dev` package group | devkitARM, libgba, libtonc, maxmod |
| **Python** | 3.10+, with [Pillow](https://pypi.org/project/Pillow/) | converter and build driver |
| **GNU make** | any recent one | devkitPro ships one on Windows |

Works on Windows, Linux and macOS. `scripts/build_rom.py` does everything;
`BUILD_ROM.bat` and `build_rom.sh` just wrap it so there's something to
double-click. Same ROM either way, byte for byte.

### OpenLara

```bash
git clone https://github.com/XProger/OpenLara.git
```

Any recent revision works. It just has to have `src/platform/gba` and
`src/fixed` in it. You don't need to build OpenLara, and you don't need any Tomb
Raider game data to view your own models.

### devkitPro

Grab the [installer](https://github.com/devkitPro/installer/releases) and pick
the **gba-dev** group. If you already have `dkp-pacman`:

```bash
dkp-pacman -S gba-dev
```

It sets `DEVKITPRO` and `DEVKITARM` for you. The Makefile needs `-ltonc -lmm`,
both in that group.

### Python

3.10 or newer, then:

```bash
pip install pillow
```

Pillow reads your textures and packs them into the engine's 256×256 atlas pages.
Skip it and the converter will tell you it's missing.

---

## Setting up

```bash
git clone https://github.com/sparklinghammer/openlara-gba-asset-viewer.git
cd openlara-gba-asset-viewer
cp settings.example.json settings.json
```

Edit `settings.json` and point `openlara` at your clone, the folder with
`src/platform/gba` in it:

```json
{
  "openlara": "C:\\code\\OpenLara",
  "models": "models",
  "roms": "roms",
  "work": "build"
}
```

Watch the backslashes: JSON wants them doubled. Forward slashes work too and
save you the trouble.

## Building

**Windows:** double-click `BUILD_ROM.bat`.
**Linux / macOS:** `./build_rom.sh`.

It converts everything in `models/`, compiles, verifies, and drops the ROM in
`roms/openlara-asset-viewer-custom.gba`. There's one model in a fresh clone, so
the first build works before you've configured anything except the OpenLara
path.

Want to build from a different folder without touching `models/`? Drag it onto
the `.bat`, or pass it to the `.sh`.

Or call the driver yourself:

```bash
python scripts/build_rom.py --openlara ~/code/OpenLara --models ~/my-models
python scripts/build_rom.py --clean          # nuke the work folder first
```

If devkitPro isn't where its installer put it and `DEVKITPRO` isn't set, add a
`"devkitpro"` key to `settings.json` pointing at the folder with
`devkitARM/gba_rules` in it.

---

## Adding your own models

Drop files in `models/`. Subfolders are scanned one level deep, since a COLLADA
file usually comes as a folder with its textures next to it.

Four formats, four readers, all producing the same thing downstream:

| Format | Reader |
|---|---|
| glTF 2.0 / GLB | `scripts/gltf_reader.py` |
| COLLADA `.dae` | `scripts/collada_reader.py` |
| Wavefront `.obj` + `.mtl` | `scripts/obj_reader.py` |
| FBX (ASCII) | `scripts/fbx_reader.py` |

Add a `models/pack.json` if you want to control the order, the names, or
anything per model. Without it the folder just gets scanned:

```json
{
  "defaults": { "frame_rate": 2, "scale": "auto" },
  "models": [
    { "file": "hero.glb",  "name": "HERO", "rotate": [90, 0, 0] },
    { "file": "crate.glb", "name": "CRATE" }
  ]
}
```

`rotate` is for Blender exports that stayed Z-up. Without it your model lies on
its side. The converter prints each model's size at rest so you can spot it, but
it won't rotate anything on its own: a dog really is longer than it is tall.

Every key is optional:

| Key | Default | What it does |
|---|---|---|
| `name` | the file name | title on screen, 23 characters max |
| `slot` | position in the list | where it goes in the catalogue, 0 to 190 |
| `scale` | `"auto"` | `auto` grows anything under 64 units up to one sector (1024); or give a number |
| `frame_rate` | `2` | ticks between resampled keyframes; 1 is 30 Hz, 2 is 15 Hz |
| `rotate` | `[0, 0, 0]` | degrees X, Y, Z, applied before anything else |
| `flip_winding` | `"auto"` | force the facing when something shows its inside |
| `double_sided` | `"auto"` | `auto` fixes the winding and keeps one face per triangle, doubling only what can't be oriented at all; `true` doubles everything, `false` never does |

`defaults` applies `frame_rate` and `scale` to every model at once.

### What the hardware can take

| Subject | Limit |
|---|---|
| Materials | three: flat colour, opaque textured, colour-keyed textured |
| Joints | 32, addressed by a 32-bit visibility mask |
| Vertices | 255 per mesh block; anything bigger gets split across extra joints at zero offset |
| Faces | 1920 per frame, refused past that, warned at 80% |
| Textures | 1536 records, 256×256 atlas pages |
| UV span | 127 texels per face max, or the engine clips it without saying so |
| Scale | there's no per-joint scale, so any scale in the chain gets baked into the vertices |
| Animated scale | can't be done; the stretch is dropped and reported, the rest of the motion stays |

If the engine can't express something, you get an error instead of a silent
drop. Ngons, sparse accessors and mirrored matrices all stop the build.

### Things that bit me

Worth knowing, because when these go wrong the model breaks in a way that looks
like a completely different bug.

**Face orientation comes from the file, not from a guess.** Normals first (glTF
`NORMAL`, COLLADA `<input semantic="NORMAL">`, OBJ `vn`, FBX
`LayerElementNormal`), then signed volume if the shell is closed. If a file has
neither, the build log says so and you settle it with `flip_winding`. I used to
work the outside out from the shape of each piece, and it turned car wheels and
locks of hair inside out.

**Skeletons are posed one vertex at a time.** The engine transforms a whole mesh
block with a single matrix. That's fine for a model built as separate rigid
limbs, and it tears apart any model whose triangles cross a joint. Every DS or
GBA era character is the second kind: a quarter of Link's triangles have corners
on two different bones. So the viewer grabs each bone's matrix, moves every
vertex by the matrix of *its own* bone into a RAM copy of the block, and hands
that to the engine as normal rigid geometry. Vertices shared between bones keep
all their influences, up to four, blended by weight.

**Seams don't open any more.** Each vertex is computed once from one rest
position, so two blocks sharing it land on the same pixel. They used to round
their own copy to the format's four-unit grid in their own frame, and two frames
round two different ways. One-pixel crack along every seam, invisible until you
zoom in.

---

## Checking a conversion

The converter isn't the last word. A few tools exist to argue with it:

```bash
# Replay a converted body the way the engine walks it, compare against the source
python scripts/check_custom_body.py --glyphs <TITLE.PKD> model.glb

# Draw it offline through the engine's own pipeline
python scripts/preview_custom_body.py --glyphs <TITLE.PKD> model.glb --out preview.png

# Read the same model through two formats and diff the result
python scripts/compare_sources.py hero.dae hero.fbx

# Diagnostic model: six faces, two joints, an animation
python scripts/make_test_model.py --output models/test_cube.glb

# And one with weighted skinning, which no ripped model has
python scripts/make_test_model.py --output models/test_skin.glb --skinned
```

`tools/viewer_tour_runner.c` is the one that shows what a GBA would actually
draw. It runs the ROM in **libmGBA**, frames every model to the same size, walks
around it and dumps the real Mode 4 page. Build it against a libmGBA of your
own:

```bash
gcc -O2 -o viewer_tour_runner tools/viewer_tour_runner.c \
  -DM_CORE_GBA -DENABLE_VFS -DENABLE_VFS_FD -DENABLE_DIRECTORIES \
  -DBUILD_STATIC -DNDEBUG -D_GNU_SOURCE \
  -DHAVE_STRDUP -DHAVE_STRNDUP -DHAVE_SETLOCALE -DHAVE_VASPRINTF \
  -I<mgba-build>/include -I<mgba-source>/include <mgba-build>/libmgba.a -lm
# Windows also needs -lshlwapi -lole32 -luuid -lws2_32

viewer_tour_runner ROM OUT_DIR MODEL_COUNT [COVERAGE_PERCENT] [ANIM_FRAMES]
```

`COVERAGE_PERCENT` is how much of the screen the model should fill before it
captures, so every model is judged at the same size. `ANIM_FRAMES` steps into
the animation by a fixed amount, so you can compare two ROMs on the same pose.
It also dumps raw palette indices, which is how you tell a real crack from black
paint: the background is palette index 0 and no texel ever lands there.

---

## Layout

```text
BUILD_ROM.bat            one-click build, Windows
build_rom.sh             same thing, Linux and macOS
settings.example.json    copy to settings.json, set the OpenLara path
models/                  your models; only the demo one is in git
overlay/                 the viewer, plus the donor files it hooks into
scripts/                 converter, build driver, verifiers
tools/                   the emulator runner
docs/                    controls, asset notice
```

The build copies the OpenLara sources into `build/`, drops `overlay/` on top,
applies a few compile fixes this devkitARM needs, and builds there. Each fix is
idempotent and complains if it stops applying, which is how you find out
OpenLara moved on upstream.

Four donor files are involved:

```text
overlay/src/fixed/common.cpp                    keeps game data out of the binary
overlay/src/platform/gba/Makefile               the ASSET_VIEWER target
overlay/src/platform/gba/main.cpp               routes init/update/render to the viewer
overlay/src/platform/gba/asset_viewer.{cpp,h}   the viewer itself
```

---

## Licence and assets

OpenLara is © Timur "XProger" Gagiev, [BSD
2-Clause](https://github.com/XProger/OpenLara/blob/master/LICENSE). The four
overlay files above are derived from it and keep the same terms.

**No Tomb Raider data here**, and you don't need any to view your own models. If
you want to load Tomb Raider's models instead, that needs your own copy of the
game. `docs/ASSET_NOTICE.md` lists exactly which files stay outside.

The demo model (`models/cheems.glb`) is made by "feverpepper" on Sketchfab (https://sketchfab.com/3d-models/cheems-912a6ee6504b4b7a8b0226000e01cdea), I just decimated the geometry a bit so it can fit better.

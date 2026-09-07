/*
 * Walk every model in an asset-viewer ROM under libmGBA and dump the screen.
 *
 * The offline preview replays the pipeline in Python, which is enough to check
 * geometry but not to see the ROM: it has no clipping path, no page flipping
 * and its own camera. This runs the actual binary and captures the visible
 * Mode 4 page, so what comes out is what a GBA would show.
 *
 * For each model it captures the default framing and then a zoomed-in view,
 * because a crack between two faces is a fraction of a pixel wide at the
 * default distance and only opens up once the model fills the screen.
 *
 * Build (MSYS2 mingw32, against the libmGBA already built in ROSE_GBA):
 *   gcc -O2 -o viewer_tour_runner.exe viewer_tour_runner.c \
 *       -I<mgba-build>/include -I<mgba-source>/include \
 *       <mgba-build>/libmgba.a -lws2_32 -lm
 *
 * Usage: viewer_tour_runner ROM OUT_DIR MODEL_COUNT [COVERAGE] [ANIM_FRAMES]
 */

#include <mgba/core/config.h>
#include <mgba/core/core.h>
#include <mgba-util/vfs.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 240u
#define HEIGHT 160u
#define HALFWORDS ((WIDTH * HEIGHT) / 2u)
#define PAGE0 0x06000000u
#define PAGE1 0x0600A000u
#define DISPCNT 0x04000000u
#define FRAME_SELECT 0x0010u
#define VIDEO_STRIDE 256u
#define VIDEO_PIXELS (VIDEO_STRIDE * 256u)

/* mGBA key order is the hardware KEYINPUT order. */
#define KEY_A (1u << 0)
#define KEY_SELECT (1u << 2)
#define KEY_START (1u << 3)
#define KEY_B (1u << 1)
#define KEY_LEFT (1u << 5)
#define KEY_START (1u << 3)
#define KEY_UP (1u << 6)
#define KEY_R (1u << 8)
#define KEY_L (1u << 9)

#define STARTUP_FRAMES 90u
#define SETTLE_FRAMES 24u
#define TAP_FRAMES 8u
#define GAP_FRAMES 12u
/* A yaw press only starts turning after YAW_HOLD_TICKS, then runs at three
 * degrees a frame, so this many frames is one eighth of a turn. */
#define YAW_WARMUP 8u
#define YAW_STEP_FRAMES (YAW_WARMUP + 15u)
#define ORBIT_STEPS 8u
/* The bands the viewer draws its own text into. */
#define HUD_TOP 20u
#define HUD_BOTTOM 128u
#define ZOOM_FRAME_CAP 400u

static unsigned visible_page(struct mCore* core) {
	uint16_t control = (uint16_t) core->busRead16(core, DISPCNT);
	/* The page being displayed is the opposite of the one being drawn. */
	return (control & FRAME_SELECT) ? 1u : 0u;
}

static void read_visible_page(struct mCore* core, uint8_t* out) {
	uint32_t base = visible_page(core) ? PAGE1 : PAGE0;
	for (unsigned i = 0; i < HALFWORDS; ++i) {
		uint16_t pair = (uint16_t) core->busRead16(core, base + i * 2u);
		out[i * 2u] = (uint8_t) (pair & 0xFFu);
		out[i * 2u + 1u] = (uint8_t) (pair >> 8);
	}
}

/* Mode 4 stores palette indices, so the dump resolves them against BG palette
 * RAM: the result is a plain 240x160 RGB555 image any tool can read. */
static bool write_rgb555(struct mCore* core, const uint8_t* page, const char* path) {
	FILE* file = fopen(path, "wb");
	if (!file) {
		fprintf(stderr, "cannot open %s\n", path);
		return false;
	}
	for (unsigned i = 0; i < WIDTH * HEIGHT; ++i) {
		uint16_t colour = (uint16_t) core->busRead16(core, 0x05000000u + (uint32_t) page[i] * 2u);
		fputc(colour & 0xFFu, file);
		fputc(colour >> 8, file);
	}
	fclose(file);
	return true;
}

static void hold(struct mCore* core, uint32_t keys, unsigned frames) {
	core->setKeys(core, keys);
	for (unsigned f = 0; f < frames; ++f) {
		core->runFrame(core);
	}
	core->setKeys(core, 0u);
	for (unsigned f = 0; f < GAP_FRAMES; ++f) {
		core->runFrame(core);
	}
}

/* The palette indices themselves, before any colour is applied. A crack that
 * lets the background through is index 0 and nothing else, so dumping the
 * indices is what separates a hole in the geometry from a wrongly sampled
 * texel that merely looks like one. */
static bool write_indices(const uint8_t* page, const char* path) {
	FILE* file = fopen(path, "wb");
	if (!file) {
		fprintf(stderr, "cannot open %s\n", path);
		return false;
	}
	fwrite(page, 1, WIDTH * HEIGHT, file);
	fclose(file);
	return true;
}

/* Palette RAM as-is, so an index can be named by the colour it resolves to. */
static bool write_palette(struct mCore* core, const char* path) {
	FILE* file = fopen(path, "wb");
	if (!file) {
		fprintf(stderr, "cannot open %s\n", path);
		return false;
	}
	for (unsigned i = 0; i < 256u; ++i) {
		uint16_t colour = (uint16_t) core->busRead16(core, 0x05000000u + i * 2u);
		fputc(colour & 0xFFu, file);
		fputc(colour >> 8, file);
	}
	fclose(file);
	return true;
}

/* How much of the screen the model covers, ignoring the title and the status
 * bar so their text is not counted as geometry. */
static unsigned coverage(struct mCore* core, uint8_t* page) {
	read_visible_page(core, page);
	unsigned filled = 0;
	for (unsigned y = HUD_TOP; y < HUD_BOTTOM; ++y) {
		for (unsigned x = 0; x < WIDTH; ++x) {
			if (page[y * WIDTH + x]) {
				filled++;
			}
		}
	}
	return filled * 100u / (WIDTH * (HUD_BOTTOM - HUD_TOP));
}

/* Zoom until the model covers the wanted share of the screen rather than for a
 * fixed number of frames: the step is a sixteenth of the model's own fitting
 * radius, so one count frames a car and puts the camera inside a coin. A crack
 * is a fraction of a pixel at the default distance and only opens up once the
 * model is large, so the framing has to be the same for every model, not the
 * key press. */
static unsigned zoom_to_fit(struct mCore* core, uint8_t* page, unsigned target) {
	unsigned frames = 0;
	while (frames < ZOOM_FRAME_CAP && coverage(core, page) < target) {
		core->setKeys(core, KEY_SELECT | KEY_UP);
		core->runFrame(core);
		frames++;
	}
	core->setKeys(core, 0u);
	for (unsigned f = 0; f < GAP_FRAMES; ++f) {
		core->runFrame(core);
	}
	return frames;
}

static void capture(struct mCore* core, uint8_t* page, const char* dir,
                    unsigned model, const char* label) {
	char path[1024];
	read_visible_page(core, page);
	snprintf(path, sizeof(path), "%s/model-%02u-%s.rgb555", dir, model, label);
	write_rgb555(core, page, path);
	snprintf(path, sizeof(path), "%s/model-%02u-%s.idx", dir, model, label);
	write_indices(page, path);
	snprintf(path, sizeof(path), "%s/model-%02u-%s.pal", dir, model, label);
	write_palette(core, path);
	printf("  model %u %s\n", model, label);
	fflush(stdout);
}

int main(int argc, char** argv) {
	if (argc < 4) {
		fprintf(stderr, "Usage: %s ROM OUT_DIR MODEL_COUNT [COVERAGE_PERCENT] [ANIM_FRAMES]\n", argv[0]);
		return 2;
	}
	const char* romPath = argv[1];
	const char* outDir = argv[2];
	unsigned models = (unsigned) strtoul(argv[3], NULL, 10);
	unsigned zoomTarget = argc > 4 ? (unsigned) strtoul(argv[4], NULL, 10) : 30u;
	unsigned stepFrames = argc > 5 ? (unsigned) strtoul(argv[5], NULL, 10) : 0u;

	mColor* video = calloc(VIDEO_PIXELS, sizeof(*video));
	uint8_t* page = malloc(WIDTH * HEIGHT);
	if (!video || !page) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}

	struct mCore* core = mCoreFind(romPath);
	if (!core || !core->init(core)) {
		fprintf(stderr, "cannot open %s\n", romPath);
		return 1;
	}
	core->setVideoBuffer(core, video, VIDEO_STRIDE);
	if (!mCoreLoadFile(core, romPath)) {
		fprintf(stderr, "cannot load %s\n", romPath);
		return 1;
	}
	mCoreInitConfig(core, "openlara-viewer-tour");
	mCoreConfigSetDefaultValue(&core->config, "idleOptimization", "remove");
	mCoreLoadConfig(core);
	core->reset(core);

	for (unsigned f = 0; f < STARTUP_FRAMES; ++f) {
		core->runFrame(core);
	}

	for (unsigned model = 0; model < models; ++model) {
		char label[64];
		for (unsigned f = 0; f < SETTLE_FRAMES; ++f) {
			core->runFrame(core);
		}
		capture(core, page, outDir, model, "default");

		/* Freeze the animation so every angle of one model shows the same pose:
		 * a crack that moves with the pose is a skinning fault, one that stays
		 * put is in the geometry. A shows/hides that difference. */
		hold(core, KEY_A, TAP_FRAMES);

		/* Start reselects the animation, which puts the frame back to the
		 * beginning; stepping from there is the only way two ROMs land on the
		 * same pose, and a pose is what a skinning change shows up in. */
		hold(core, KEY_START, TAP_FRAMES);
		for (unsigned f = 0; f < stepFrames; ++f) {
			hold(core, KEY_SELECT | KEY_A, TAP_FRAMES);
		}

		/* Select held with Up is the viewer's zoom. Holding it also marks the
		 * Select press as used, so releasing it does not open the menu. */
		printf("  model %u framed in %u frames\n", model, zoom_to_fit(core, page, zoomTarget));

		/* One camera angle proves nothing. A face that faces away is invisible
		 * from the front and a gaping hole from the side, so walk the model all
		 * the way round and capture every eighth of a turn. */
		for (unsigned step = 0; step < ORBIT_STEPS; ++step) {
			snprintf(label, sizeof(label), "yaw%03u", step * 45u);
			capture(core, page, outDir, model, label);
			hold(core, KEY_LEFT, YAW_STEP_FRAMES);
		}

		/* B puts the yaw, pitch, roll and zoom back, so the next model starts
		 * from its own fitted view. */
		hold(core, KEY_B, TAP_FRAMES);
		hold(core, KEY_A, TAP_FRAMES);
		if (model + 1 < models) {
			hold(core, KEY_R, TAP_FRAMES);
		}
	}

	core->unloadROM(core);
	core->deinit(core);
	free(page);
	free(video);
	return 0;
}

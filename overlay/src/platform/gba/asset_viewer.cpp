#include "common.h"
#include "asset_viewer.h"

#ifdef ASSET_VIEWER

void updateLevel(int32 frames);

#include "ASSET_VIEWER_PAK.h"

namespace AssetViewer
{
    static const int32 GLYPH_COUNT = 110;
    static const int32 ROTATION_STEP = ANGLE_1 * 3;
    static const int32 AUTO_ROTATION_STEP = ANGLE_1;
    static const int32 YAW_HOLD_TICKS = 8;
    static const int32 VERTICAL_MOVE_STEP = 2;

    static const int32 PANEL_LIGHT = 14;
    static const int32 PANEL_DARK = 10;
    static const int32 MODEL_TARGET_X = FRAME_WIDTH / 2;
    static const int32 MODEL_TARGET_Y = 81;
    static const int32 INFO_PANEL_TOP = 140;
    static const int32 INFO_STATUS_X = 190;
    static const int32 HELP_PANEL_TOP = 31;
    static const int32 MENU_ROW_CURSOR = 52;
    static const int32 MENU_ROW_NAME = 64;
    static const int32 MENU_ROW_VALUE = 160;

    // Every custom body embeds the same TR1 glyph strip, and its colours are
    // reserved first, so palette entries 0..17 are identical in every model.
    // Index 12 is the brightest of them, which makes it the one colour a
    // wireframe can rely on whatever model is loaded.
    static const int32 WIRE_COLOR = 12;

    enum SettingId
    {
        SETTING_WIREFRAME = 0,
        SETTING_COUNT
    };

    static const uint8 CHAR_WIDTH[GLYPH_COUNT] = {
        14, 11, 11, 11, 11, 11, 11, 13, 8, 11, 12, 11, 13, 13, 12, 11, 12, 12, 11, 12, 13, 13, 13, 12, 12, 11,
        9, 9, 9, 9, 9, 9, 9, 9, 5, 9, 9, 5, 12, 10, 9, 9, 9, 8, 9, 8, 9, 9, 11, 9, 9, 9,
        12, 8, 10, 10, 10, 10, 10, 9, 10, 10,
        5, 5, 5, 11, 9, 7, 8, 6, 0, 7, 7, 3, 8, 8, 13, 7, 9, 4, 12, 12,
        7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 16, 14, 14, 14, 16, 16, 16, 16, 16, 12, 14, 8, 8, 8, 8, 8, 8, 8
    };

    static const uint8 CHAR_MAP[102] = {
        0, 64, 66, 78, 77, 74, 78, 79, 69, 70, 92, 72, 63, 71, 62, 68, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 73, 73, 66, 74, 75, 65, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 80, 76, 81, 97, 98, 77, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 100, 101, 102, 67, 0, 0, 0, 0, 0, 0, 0
    };

    #define ASSET_VIEWER_MODEL_NAME(name) #name,
    static const char* const MODEL_NAMES[MAX_MODELS] = {
        ITEM_TYPES(ASSET_VIEWER_MODEL_NAME)
    };
    #undef ASSET_VIEWER_MODEL_NAME

    static const uint32 PACK_MAGIC = 0x31505641; // "AVP1"
    static const uint32 PACK_VERSION_MIN = 2;
    static const uint32 PACK_VERSION_MAX = 4;
    static const int32 PACK_HEADER_SIZE = 24;    // v2 header, still the common prefix
    static const int32 PACK_HEADER_SIZE_V3 = 32; // v2 header + flags + name table offset
    static const int32 PACK_HEADER_SIZE_V4 = 36; // v3 header + skin table offset
    static const int32 SOURCE_ENTRY_SIZE = 16;
    static const int32 MODEL_ENTRY_SIZE = 12;
    static const int32 NAME_ENTRY_SIZE = 24;
    static const int32 SKIN_ENTRY_SIZE = 16;   // bone table, then the blend records

    // PACK_FLAG_TR1 marks bodies converted from Tomb Raider level data. It enables the
    // runtime fixups that only make sense there: armed-Lara mesh composition and the
    // 50% rescale of the native glyph strip. A custom producer that borrows the TR1
    // font must pre-scale its own sprite records instead of arming this flag.
    static const uint32 PACK_FLAG_TR1 = 1 << 0;
    static const uint32 PACK_FLAG_NAMES = 1 << 1;
    // PACK_FLAG_SKIN marks a pack carrying, for some models, one byte per vertex
    // naming the joint that vertex follows. The engine's mesh blocks are untouched
    // by it; it is read here, to pose a model a vertex at a time.
    static const uint32 PACK_FLAG_SKIN = 1 << 2;

    static uint16 readU16(const uint8* data)
    {
        return uint16(data[0]) | (uint16(data[1]) << 8);
    }

    static uint32 readU32(const uint8* data)
    {
        return uint32(data[0]) |
              (uint32(data[1]) << 8) |
              (uint32(data[2]) << 16) |
              (uint32(data[3]) << 24);
    }

    static bool validPack()
    {
        if (ASSET_VIEWER_PAK_size < PACK_HEADER_SIZE)
            return false;
        if (readU32(ASSET_VIEWER_PAK + 0) != PACK_MAGIC)
            return false;
        uint32 version = readU32(ASSET_VIEWER_PAK + 4);
        if (version < PACK_VERSION_MIN || version > PACK_VERSION_MAX)
            return false;
        if (version >= 3 && ASSET_VIEWER_PAK_size < PACK_HEADER_SIZE_V3)
            return false;
        if (version >= 4 && ASSET_VIEWER_PAK_size < PACK_HEADER_SIZE_V4)
            return false;
        return readU32(ASSET_VIEWER_PAK + 8) <= ASSET_VIEWER_PAK_size;
    }

    // v2 predates the flag word. Every v2 pack was produced from TR1 level data.
    static uint32 packFlags()
    {
        if (!validPack())
            return 0;
        if (readU32(ASSET_VIEWER_PAK + 4) < 3)
            return PACK_FLAG_TR1;
        return readU32(ASSET_VIEWER_PAK + 24);
    }

    static const uint8* sourceData(int32 index)
    {
        if (!validPack())
            return NULL;
        int32 sourceCount = readU16(ASSET_VIEWER_PAK + 12);
        if (index < 0 || index >= sourceCount)
            return NULL;
        uint32 table = readU32(ASSET_VIEWER_PAK + 16);
        const uint8* entry = ASSET_VIEWER_PAK + table + index * SOURCE_ENTRY_SIZE;
        uint32 offset = readU32(entry + 8);
        uint32 size = readU32(entry + 12);
        uint32 total = readU32(ASSET_VIEWER_PAK + 8);
        if (size < sizeof(Level) || offset > total || size > total - offset)
            return NULL;
        return ASSET_VIEWER_PAK + offset;
    }

    struct TextBuffer
    {
        char data[64];
        int32 length;

        void clear()
        {
            length = 0;
            data[0] = 0;
        }

        bool push(char c)
        {
            if (length >= int32(sizeof(data)) - 1)
                return false;
            data[length++] = c;
            data[length] = 0;
            return true;
        }

        void pop()
        {
            if (length > 0)
                length--;
            data[length] = 0;
        }

        void append(const char* text)
        {
            while (*text && push(*text))
                text++;
        }

        void appendUInt(uint32 value)
        {
            char digits[10];
            int32 count = 0;
            do {
                digits[count++] = char('0' + value % 10);
                value /= 10;
            } while (value && count < int32(sizeof(digits)));

            while (count > 0)
                push(digits[--count]);
        }
    };

    struct State
    {
        uint8 modelTypes[MAX_MODELS];
        uint8 modelSources[MAX_MODELS];
        uint32 modelMasks[MAX_MODELS];
        char modelName[NAME_ENTRY_SIZE];
        int32 modelCount;
        int32 modelIndex;
        int32 sourceIndex;
        int32 animationSlot;
        int32 frameIndex;
        int32 yaw;
        int32 pitch;
        int32 roll;
        int32 zoom;
        int32 verticalOffset;
        vec3i fitCenter;
        int32 fitRadius;
        int32 fitDistance;
        int32 fitScale;
        int32 yawHoldTicks;
        int32 yawPressDirection;
        int32 autoYawDirection;
        uint32 previousKeys;
        uint32 uiTicks;
        int32 menuCursor;
        bool settings[SETTING_COUNT];
        bool playing;
        bool showHelp;
        bool selectUsed;
        bool yawTapEligible;
        bool yawManual;
    };

    EWRAM_DATA State state;

    static int32 steppedIndex(int32 index, int32 count, int32 direction)
    {
        if (count <= 0)
            return 0;
        if (direction >= 0)
            return (index + 1) % count;
        return index == 0 ? count - 1 : index - 1;
    }

    static int32 totalAnimationCount()
    {
        intptr_t begin = intptr_t(level.anims);
        intptr_t end = intptr_t(level.animStates);
        if (end <= begin)
            return 0;
        return int32((end - begin) / sizeof(Anim));
    }

    static void animationRange(int32 type, int32 &start, int32 &end)
    {
        start = 0;
        end = 0;
        if (type < 0 || type >= MAX_MODELS)
            return;

        int32 total = totalAnimationCount();
        const Model &model = level.models[type];
        start = model.animIndex;
        if (model.count <= 0 || start >= total) {
            start = 0;
            return;
        }

        end = total;
        for (int32 i = 0; i < MAX_MODELS; i++)
        {
            const Model &candidate = level.models[i];
            int32 candidateStart = candidate.animIndex;
            if (candidate.count > 0 && candidateStart > start && candidateStart < end)
                end = candidateStart;
        }
    }

    static int32 selectedType()
    {
        if (state.modelCount <= 0)
            return -1;
        return state.modelTypes[state.modelIndex];
    }

    static uint32 selectedMeshMask()
    {
        int32 type = selectedType();
        return type < 0 ? 0 : state.modelMasks[type];
    }

    static int32 animationCount()
    {
        int32 start, end;
        animationRange(selectedType(), start, end);
        return end - start;
    }

    static int32 selectedAnimation()
    {
        int32 start, end;
        animationRange(selectedType(), start, end);
        if (end <= start)
            return -1;
        int32 slot = X_CLAMP(state.animationSlot, 0, end - start - 1);
        return start + slot;
    }

    static AABBi rootNeutralBounds(const AnimFrame* frame)
    {
        AABBi bounds;
        bounds.minX = int32(frame->box.minX) - frame->pos.x;
        bounds.maxX = int32(frame->box.maxX) - frame->pos.x;
        bounds.minY = int32(frame->box.minY) - frame->pos.y;
        bounds.maxY = int32(frame->box.maxY) - frame->pos.y;
        bounds.minZ = int32(frame->box.minZ) - frame->pos.z;
        bounds.maxZ = int32(frame->box.maxZ) - frame->pos.z;
        return bounds;
    }

    static void extendBounds(AABBi &bounds, const AABBi &other)
    {
        bounds.minX = X_MIN(bounds.minX, other.minX);
        bounds.maxX = X_MAX(bounds.maxX, other.maxX);
        bounds.minY = X_MIN(bounds.minY, other.minY);
        bounds.maxY = X_MAX(bounds.maxY, other.maxY);
        bounds.minZ = X_MIN(bounds.minZ, other.minZ);
        bounds.maxZ = X_MAX(bounds.maxZ, other.maxZ);
    }

    static void getViewerFrames(int32 type, int32 animation, int32 frameIndex,
                                const AnimFrame* &frameA, const AnimFrame* &frameB,
                                int32 &frameRate, int32 &frameDelta)
    {
        const Anim &anim = level.anims[animation];
        int32 frameSize = (sizeof(AnimFrame) >> 1) + (level.models[type].count << 1);
        const AnimFrame* first = (const AnimFrame*)(level.animFrames + (anim.frameOffset >> 1));
        if (anim.frameBegin == anim.frameEnd) {
            frameA = frameB = first;
            frameRate = 1;
            frameDelta = 0;
            return;
        }

        frameRate = X_MAX(int32(anim.frameRate), 1);
        int32 relative = X_MAX(frameIndex - int32(anim.frameBegin), 0);
        int32 indexA = relative / frameRate;
        frameDelta = relative - indexA * frameRate;
        int32 indexB = indexA + 1;
        if (indexB * frameRate >= anim.frameEnd)
            indexB = indexA;
        frameA = (const AnimFrame*)((const uint16*)first + indexA * frameSize);
        frameB = (const AnimFrame*)((const uint16*)first + indexB * frameSize);
        if (!frameDelta || frameA == frameB) {
            frameDelta = 0;
            return;
        }
        int32 frameBIndex = indexB * frameRate;
        if (frameBIndex > anim.frameEnd)
            frameRate -= frameBIndex - anim.frameEnd;
    }

    static void recomputeFit()
    {
        int32 type = selectedType();
        int32 animationIndex = selectedAnimation();
        if (type < 0 || animationIndex < 0) {
            state.fitCenter = _vec3i(0, 0, 0);
            state.fitRadius = 128;
            state.fitDistance = 768;
            state.fitScale = 1 << FIXED_SHIFT;
            state.zoom = 0;
            return;
        }

        const Anim &anim = level.anims[animationIndex];
        int32 begin = anim.frameBegin;
        int32 end = X_MAX(int32(anim.frameEnd), begin);
        int32 step = X_MAX(int32(anim.frameRate), 1);

        const AnimFrame *frameA, *frameB;
        int32 frameRate, frameDelta;
        getViewerFrames(type, animationIndex, begin, frameA, frameB, frameRate, frameDelta);
        AABBi bounds = rootNeutralBounds(frameA);
        extendBounds(bounds, rootNeutralBounds(frameB));

        for (int32 frame = begin; ; )
        {
            getViewerFrames(type, animationIndex, frame, frameA, frameB, frameRate, frameDelta);
            extendBounds(bounds, rootNeutralBounds(frameA));
            extendBounds(bounds, rootNeutralBounds(frameB));

            if (frame >= end)
                break;
            int32 next = X_MIN(frame + step, end);
            if (next <= frame)
                break;
            frame = next;
        }

        state.fitCenter = _vec3i(
            (int32(bounds.minX) + int32(bounds.maxX)) / 2,
            (int32(bounds.minY) + int32(bounds.maxY)) / 2,
            (int32(bounds.minZ) + int32(bounds.maxZ)) / 2
        );

        int32 radiusX = (int32(bounds.maxX) - int32(bounds.minX) + 1) / 2;
        int32 radiusY = (int32(bounds.maxY) - int32(bounds.minY) + 1) / 2;
        int32 radiusZ = (int32(bounds.maxZ) - int32(bounds.minZ) + 1) / 2;
        uint32 radiusSquared = uint32(radiusX) * uint32(radiusX) +
                               uint32(radiusY) * uint32(radiusY) +
                               uint32(radiusZ) * uint32(radiusZ);
        state.fitRadius = X_MAX(64, int32(phd_sqrt(radiusSquared)));

        // The bounding-sphere radius covers every X/Y/Z orientation. With the
        // exact Mode 4 projection, five radii leave the nearest point at four
        // radii and project the sphere to about 51 pixels.
        int32 distanceForScale = state.fitRadius * 5;
        int32 distanceForDepth = state.fitRadius + 384;
        int32 idealDistance = X_MAX(distanceForScale, distanceForDepth);
        int32 maximumDistance = VIEW_DIST - 512;
        state.fitDistance = X_CLAMP(idealDistance, 512, maximumDistance);
        state.fitScale = idealDistance > maximumDistance
                       ? maximumDistance * (1 << FIXED_SHIFT) / idealDistance
                       : 1 << FIXED_SHIFT;
        state.zoom = 0;
    }

    static void resetClip()
    {
        int32 count = animationCount();
        if (count <= 0) {
            state.animationSlot = 0;
            state.frameIndex = 0;
            recomputeFit();
            return;
        }

        state.animationSlot %= count;
        int32 animationIndex = selectedAnimation();
        state.frameIndex = level.anims[animationIndex].frameBegin;
        recomputeFit();
    }

    static void resetOrientation()
    {
        state.yaw = ANGLE_180;
        state.pitch = 0;
        state.roll = 0;
        state.zoom = 0;
        state.verticalOffset = 0;
        state.yawHoldTicks = 0;
        state.yawPressDirection = 0;
        state.autoYawDirection = 0;
        state.yawTapEligible = false;
        state.yawManual = false;
    }

    static void buildUnifiedCatalog()
    {
        state.modelCount = 0;
        state.modelIndex = 0;
        memset(state.modelSources, 0xFF, sizeof(state.modelSources));
        memset(state.modelMasks, 0, sizeof(state.modelMasks));
        if (!validPack())
            return;

        int32 sourceCount = readU16(ASSET_VIEWER_PAK + 12);
        int32 modelCount = readU16(ASSET_VIEWER_PAK + 14);
        uint32 table = readU32(ASSET_VIEWER_PAK + 20);
        uint32 total = readU32(ASSET_VIEWER_PAK + 8);
        if (modelCount > MAX_MODELS || table > total ||
            uint32(modelCount * MODEL_ENTRY_SIZE) > total - table)
            return;

        for (int32 index = 0; index < modelCount; index++)
        {
            const uint8* entry = ASSET_VIEWER_PAK + table + index * MODEL_ENTRY_SIZE;
            int32 type = entry[0];
            int32 source = entry[1];
            int32 meshCount = entry[2];
            uint32 meshMask = readU32(entry + 8);
            uint32 allowedMask = meshCount >= 32 ? 0xFFFFFFFF : ((uint32(1) << meshCount) - 1);
            if (type >= MAX_MODELS || source >= sourceCount || meshCount <= 0 ||
                meshCount > 32 || meshMask == 0 || (meshMask & ~allowedMask) ||
                state.modelSources[type] != 0xFF) {
                state.modelCount = 0;
                return;
            }
            state.modelTypes[index] = uint8(type);
            state.modelSources[type] = uint8(source);
            state.modelMasks[type] = meshMask;
            state.modelCount++;
        }
    }

    static void scaleGlyphs()
    {
        int32 start = level.models[ITEM_GLYPHS].start;
        if (start < 0 || start + GLYPH_COUNT > level.spritesCount)
            return;

        for (int32 i = 0; i < GLYPH_COUNT; i++)
        {
            Sprite &sprite = level.sprites[start + i];
            sprite.l /= 2;
            sprite.t /= 2;
            sprite.r /= 2;
            sprite.b /= 2;
        }
    }

    // The catalog title comes from the pack when it carries a name table, so a pack
    // built from sources other than TR1 is not restricted to the donor's ITEM_TYPES
    // slots. MODEL_NAMES stays as the fallback for v2 packs and for any pack that
    // omits the table.
    static void cacheModelName()
    {
        state.modelName[0] = 0;

        int32 type = selectedType();
        if (type < 0)
            return;

        if (packFlags() & PACK_FLAG_NAMES)
        {
            uint32 total = readU32(ASSET_VIEWER_PAK + 8);
            uint32 table = readU32(ASSET_VIEWER_PAK + 28);
            int32 count = readU16(ASSET_VIEWER_PAK + 14);
            int32 index = state.modelIndex;
            if (table && index >= 0 && index < count && table <= total &&
                uint32(count * NAME_ENTRY_SIZE) <= total - table)
            {
                const uint8* slot = ASSET_VIEWER_PAK + table + index * NAME_ENTRY_SIZE;
                int32 i = 0;
                while (i < NAME_ENTRY_SIZE - 1 && slot[i])
                {
                    state.modelName[i] = char(slot[i]);
                    i++;
                }
                state.modelName[i] = 0;
                if (i > 0)
                    return;
            }
        }

        const char* fallback = MODEL_NAMES[type];
        int32 i = 0;
        while (i < NAME_ENTRY_SIZE - 1 && fallback[i])
        {
            state.modelName[i] = fallback[i];
            i++;
        }
        state.modelName[i] = 0;
    }

    static void prepareLaraVariant(int32 type)
    {
        if (type < ITEM_LARA_PISTOLS || type > ITEM_LARA_UZIS)
            return;

        const Model &base = level.models[ITEM_LARA];
        const Model &variant = level.models[type];
        const uint32 handMask = (uint32(1) << JOINT_ARM_R3) | (uint32(1) << JOINT_ARM_L3);
        if (base.count != JOINT_MAX || variant.count != JOINT_MAX ||
            base.start + JOINT_MAX > level.meshesCount ||
            variant.start + JOINT_MAX > level.meshesCount ||
            (state.modelMasks[type] & handMask) != handMask)
            return;

        const Mesh* handR = level.meshes[variant.start + JOINT_ARM_R3];
        const Mesh* handL = level.meshes[variant.start + JOINT_ARM_L3];
        for (int32 joint = 0; joint < JOINT_MAX; joint++)
            level.meshes[variant.start + joint] = level.meshes[base.start + joint];
        level.meshes[variant.start + JOINT_ARM_R3] = handR;
        level.meshes[variant.start + JOINT_ARM_L3] = handL;
        state.modelMasks[type] = (uint32(1) << JOINT_MAX) - 1;
    }

    // Defined with the rest of the posing code, further down.
    static void prepareSkin(int32 type);

    static void loadSelectedModel()
    {
        if (state.modelCount <= 0)
            return;

        int32 type = selectedType();
        cacheModelName();
        state.sourceIndex = state.modelSources[type];
        const uint8* data = sourceData(state.sourceIndex);
        if (!data)
            return;

        dmaFill((void*)MEM_VRAM, 0, VRAM_PAGE_SIZE * 2);
        readLevel(data);
        if (packFlags() & PACK_FLAG_TR1)
        {
            prepareLaraVariant(type);
            scaleGlyphs();
        }
        // After the level is in place and any TR1 fixup has run, because the
        // blocks it copies are the ones that will actually be drawn.
        prepareSkin(type);

        gBrightness = 0;
        palSet(level.palette, gSettings.video_gamma << 4, 0);
        *((volatile uint16*)MEM_PAL_BG) = 0;

        drawLevelInit();
        state.animationSlot = 0;
        state.frameIndex = 0;
        state.playing = true;
        resetOrientation();
        resetClip();
    }

    static void selectModel(int32 direction)
    {
        if (state.modelCount <= 1)
            return;
        drawLevelFree();
        state.modelIndex = steppedIndex(state.modelIndex, state.modelCount, direction);
        loadSelectedModel();
    }

    static void selectAnimation(int32 direction)
    {
        int32 count = animationCount();
        if (count <= 1)
            return;

        // Keep every view field untouched: changing a clip only selects its
        // first frame and does not invoke either fitting/reset helper.
        state.animationSlot = steppedIndex(state.animationSlot, count, direction);
        int32 animationIndex = selectedAnimation();
        state.frameIndex = level.anims[animationIndex].frameBegin;
    }

    static void adjustVertical(int32 direction, int32 frames)
    {
        state.verticalOffset += direction * VERTICAL_MOVE_STEP * X_MAX(frames, 1);
        state.verticalOffset = X_CLAMP(state.verticalOffset, -144, 144);
    }

    static void stepFrames(int32 amount)
    {
        int32 animationIndex = selectedAnimation();
        if (animationIndex < 0)
            return;

        const Anim &anim = level.anims[animationIndex];
        int32 begin = anim.frameBegin;
        int32 end = X_MAX(int32(anim.frameEnd), begin);
        int32 length = end - begin + 1;
        if (length <= 1) {
            state.frameIndex = begin;
            return;
        }

        int32 relative = state.frameIndex - begin + amount;
        relative %= length;
        if (relative < 0)
            relative += length;
        state.frameIndex = begin + relative;
    }

    static void adjustZoom(int32 direction, int32 frames)
    {
        int32 step = X_MAX(state.fitRadius / 16, 8) * X_MAX(frames, 1);
        state.zoom += direction * step;
        int32 minimum = 320 - state.fitDistance;
        int32 maximum = VIEW_DIST - 512 - state.fitDistance;
        state.zoom = X_CLAMP(state.zoom, minimum, maximum);
    }

    static int32 charRemapCompact(uint8 c)
    {
        if (c < 11)
            return c + 81;
        if (c < 16)
            return c + 91;
        if (c < 32 || c >= 134)
            return -1;
        return CHAR_MAP[c - 32];
    }

    static int32 glyphAdvance(int32 index)
    {
        return (CHAR_WIDTH[index] + 2) / 2;
    }

    static int32 textWidth(const char* text)
    {
        int32 width = 0;
        while (*text)
        {
            uint8 c = uint8(*text++);
            if (c == ' ' || c == '_') {
                width += 3;
                continue;
            }
            int32 index;
            if (c == '$') {
                if (!*text)
                    break;
                index = uint8(*text++);
            } else {
                index = charRemapCompact(c);
            }
            if (index >= 0 && index < GLYPH_COUNT)
                width += glyphAdvance(index);
        }
        return width;
    }

    static void drawTextCompact(int32 x, int32 y, const char* text, TextAlign align)
    {
        if (!text || !*text)
            return;

        int32 width = textWidth(text);
        if (align == TEXT_ALIGN_CENTER)
            x += (FRAME_WIDTH - width) / 2;
        else if (align == TEXT_ALIGN_RIGHT)
            x += FRAME_WIDTH - width;

        int32 glyphStart = level.models[ITEM_GLYPHS].start;
        while (*text)
        {
            uint8 c = uint8(*text++);
            if (c == ' ' || c == '_') {
                x += 3;
                continue;
            }
            int32 index;
            if (c == '$') {
                if (!*text)
                    break;
                index = uint8(*text++);
            } else {
                index = charRemapCompact(c);
            }
            if (index >= 0 && index < GLYPH_COUNT) {
                renderGlyph(x, y, glyphStart + index);
                x += glyphAdvance(index);
            }
        }
    }

    static void drawTextClipped(int32 x, int32 y, const char* text, int32 maxWidth)
    {
        TextBuffer line;
        line.clear();
        while (*text)
        {
            if (!line.push(*text++))
                break;
            if (textWidth(line.data) > maxWidth) {
                line.pop();
                break;
            }
        }
        drawTextCompact(x, y, line.data, TEXT_ALIGN_LEFT);
    }

    static void drawMarquee(const char* text, int32 y)
    {
        if (textWidth(text) <= FRAME_WIDTH - 4) {
            drawTextCompact(0, y, text, TEXT_ALIGN_CENTER);
            return;
        }

        int32 length = strlen(text);
        int32 cycle = length + 5;
        int32 start = (state.uiTicks / 6) % cycle;
        TextBuffer line;
        line.clear();

        for (int32 offset = 0; offset < cycle * 2; offset++)
        {
            int32 index = (start + offset) % cycle;
            char c = index < length ? text[index] : ' ';
            if (!line.push(c))
                break;
            if (textWidth(line.data) > FRAME_WIDTH - 4) {
                line.pop();
                break;
            }
        }
        drawTextCompact(2, y, line.data, TEXT_ALIGN_LEFT);
    }

    static void drawDetails()
    {
        renderFill(0, INFO_PANEL_TOP, FRAME_WIDTH, FRAME_HEIGHT - INFO_PANEL_TOP, 0, 2);
        renderBorder(0, INFO_PANEL_TOP, FRAME_WIDTH, FRAME_HEIGHT - INFO_PANEL_TOP,
                     PANEL_LIGHT, PANEL_DARK, 1);

        if (state.modelCount <= 0) {
            drawTextCompact(0, 150, "NO DRAWABLE MODELS", TEXT_ALIGN_CENTER);
            return;
        }

        TextBuffer line;
        line.clear();
        line.append("MODEL ");
        line.appendUInt(state.modelIndex + 1);
        line.push('/');
        line.appendUInt(state.modelCount);
        line.append("  ANIM ");

        int32 count = animationCount();
        if (count <= 0) {
            line.append("0/0");
        } else {
            line.appendUInt(state.animationSlot + 1);
            line.push('/');
            line.appendUInt(count);
        }

        int32 animationIndex = selectedAnimation();
        if (animationIndex >= 0)
        {
            const Anim &anim = level.anims[animationIndex];
            line.append("  STATE ");
            line.appendUInt(anim.state);
        }
        drawTextClipped(4, 149, line.data, FRAME_WIDTH - 8);

        line.clear();
        if (animationIndex >= 0)
        {
            const Anim &anim = level.anims[animationIndex];
            int32 begin = anim.frameBegin;
            int32 end = X_MAX(int32(anim.frameEnd), begin);
            line.append("FRAME ");
            line.appendUInt(X_MAX(state.frameIndex - begin, 0));
            line.push('/');
            line.appendUInt(end - begin);
        }
        drawTextClipped(4, 158, line.data, INFO_STATUS_X - 8);
        drawTextCompact(INFO_STATUS_X, 158,
                        state.playing ? "PLAYING" : "PAUSED",
                        TEXT_ALIGN_LEFT);
    }

    static vec3i interpolatedRoot(const AnimFrame* frameA, const AnimFrame* frameB,
                                  int32 frameRate, int32 frameDelta)
    {
        if (frameDelta == 0 || frameA == frameB || frameRate <= 0)
            return _vec3i(frameA->pos.x, frameA->pos.y, frameA->pos.z);

        int32 t = GET_FRAME_T(frameDelta, frameRate);
        return _vec3i(
            frameA->pos.x + ((frameB->pos.x - frameA->pos.x) * t >> 16),
            frameA->pos.y + ((frameB->pos.y - frameA->pos.y) * t >> 16),
            frameA->pos.z + ((frameB->pos.z - frameA->pos.z) * t >> 16)
        );
    }

    static int32 screenOffsetToWorld(int32 pixels, int32 distance)
    {
        // transformMesh.s projects with divTable[(z >> 4) + (z >> 6)]
        // and a final shift of 12. Its inverse divides the projection depth
        // by 16; using 8 would double every requested screen-space offset.
        // This gives a stable
        // screen-space target independent of the fitted model radius.
        int32 projectionDepth = (distance >> 4) + (distance >> 6);
        return pixels * projectionDepth / 16;
    }

    static void scaleCurrentBasis(int32 scale)
    {
        if (scale >= (1 << FIXED_SHIFT))
            return;

        Matrix &matrix = matrixGet();
        matrix.e00 = matrix.e00 * scale >> FIXED_SHIFT;
        matrix.e01 = matrix.e01 * scale >> FIXED_SHIFT;
        matrix.e02 = matrix.e02 * scale >> FIXED_SHIFT;
        matrix.e10 = matrix.e10 * scale >> FIXED_SHIFT;
        matrix.e11 = matrix.e11 * scale >> FIXED_SHIFT;
        matrix.e12 = matrix.e12 * scale >> FIXED_SHIFT;
        matrix.e20 = matrix.e20 * scale >> FIXED_SHIFT;
        matrix.e21 = matrix.e21 * scale >> FIXED_SHIFT;
        matrix.e22 = matrix.e22 * scale >> FIXED_SHIFT;
    }

    // ---- wireframe --------------------------------------------------------
    // The engine draws filled polygons and has no line mode, so the wireframe
    // is drawn here rather than by adding a face type: the viewer walks the
    // same node hierarchy, projects each mesh with the same matrix, and puts
    // the edges straight into the framebuffer. Nothing in the engine changes,
    // which is the point -- the renderer this viewer is meant to show off stays
    // exactly as it ships.

    struct WireVertex
    {
        int32 x;
        int32 y;
        int32 clipped;      // behind the near plane or past the far one
    };

    EWRAM_DATA static WireVertex wireVertices[256];

    // Mode 4 video RAM refuses byte writes: a pixel has to be merged into the
    // 16-bit word it shares with its neighbour. Same idiom as rasterizeLineV.
    static void wirePixel(int32 x, int32 y)
    {
        volatile uint8* ptr = (uint8*)fb + y * FRAME_WIDTH + x;
        if (intptr_t(ptr) & 1) {
            *(uint16*)(ptr - 1) = *(ptr - 1) | (WIRE_COLOR << 8);
        } else {
            *(uint16*)ptr = WIRE_COLOR | (*ptr << 8);
        }
    }

    static int32 wireOutcode(int32 x, int32 y)
    {
        int32 code = 0;
        if (x < 0) code |= 1; else if (x >= FRAME_WIDTH)  code |= 2;
        if (y < 0) code |= 4; else if (y >= FRAME_HEIGHT) code |= 8;
        return code;
    }

    // Clipped before it is walked, not per pixel: a vertex just off the near
    // plane projects hundreds of screens away, and stepping there one pixel at
    // a time would cost more than the whole model.
    static void wireLine(int32 x0, int32 y0, int32 x1, int32 y1)
    {
        int32 code0 = wireOutcode(x0, y0);
        int32 code1 = wireOutcode(x1, y1);

        for (int32 guard = 0; (code0 | code1) && guard < 8; guard++)
        {
            if (code0 & code1)
                return;                     // both beyond the same edge

            int32 code = code0 ? code0 : code1;
            int32 x = x0;
            int32 y = y0;
            if (code & 8) {
                x = x0 + (x1 - x0) * (FRAME_HEIGHT - 1 - y0) / (y1 - y0);
                y = FRAME_HEIGHT - 1;
            } else if (code & 4) {
                x = x0 + (x1 - x0) * (0 - y0) / (y1 - y0);
                y = 0;
            } else if (code & 2) {
                y = y0 + (y1 - y0) * (FRAME_WIDTH - 1 - x0) / (x1 - x0);
                x = FRAME_WIDTH - 1;
            } else if (code & 1) {
                y = y0 + (y1 - y0) * (0 - x0) / (x1 - x0);
                x = 0;
            }

            if (code == code0) {
                x0 = x; y0 = y; code0 = wireOutcode(x0, y0);
            } else {
                x1 = x; y1 = y; code1 = wireOutcode(x1, y1);
            }
        }

        if (code0 | code1)
            return;                         // integer truncation left it outside

        int32 dx = x1 - x0; if (dx < 0) dx = -dx;
        int32 dy = y1 - y0; if (dy < 0) dy = -dy;
        int32 sx = (x0 < x1) ? 1 : -1;
        int32 sy = (y0 < y1) ? 1 : -1;
        int32 error = dx - dy;

        while (1)
        {
            wirePixel(x0, y0);
            if (x0 == x1 && y0 == y1)
                break;
            int32 doubled = error << 1;
            if (doubled > -dy) { error -= dy; x0 += sx; }
            if (doubled <  dx) { error += dx; y0 += sy; }
        }
    }

    // The same projection transformMesh applies, so the wireframe lands exactly
    // where the solid model would.
    static void wireTransform(const MeshVertex* vertices, int32 count)
    {
        const Matrix &m = matrixGet();

        for (int32 i = 0; i < count; i++)
        {
            int32 vx = vertices[i].x << 2;
            int32 vy = vertices[i].y << 2;
            int32 vz = vertices[i].z << 2;

            int32 x = DP43(m.e00, m.e01, m.e02, m.e03, vx, vy, vz);
            int32 y = DP43(m.e10, m.e11, m.e12, m.e13, vx, vy, vz);
            int32 z = DP43(m.e20, m.e21, m.e22, m.e23, vx, vy, vz);

            int32 clipped = 0;
            if (z <= VIEW_MIN_F) { clipped = 1; z = VIEW_MIN_F; }
            if (z >= VIEW_MAX_F) { clipped = 1; z = VIEW_MAX_F; }

            x >>= FIXED_SHIFT;
            y >>= FIXED_SHIFT;
            z >>= FIXED_SHIFT;

            PERSPECTIVE(x, y, z);

            wireVertices[i].x = x + (FRAME_WIDTH  >> 1);
            wireVertices[i].y = y + (FRAME_HEIGHT >> 1);
            wireVertices[i].clipped = clipped;
        }
    }

    static void wireEdge(int32 a, int32 b)
    {
        if (wireVertices[a].clipped || wireVertices[b].clipped)
            return;                         // its projection means nothing
        wireLine(wireVertices[a].x, wireVertices[a].y,
                 wireVertices[b].x, wireVertices[b].y);
    }

    static void wireMesh(int32 meshIndex)
    {
        const Mesh* mesh = level.meshes[meshIndex];
        int32 vCount = mesh->vCount;
        if (vCount <= 0 || vCount > int32(sizeof(wireVertices) / sizeof(wireVertices[0])))
            return;

        const uint8* ptr = (uint8*)mesh + sizeof(Mesh);
        const MeshQuad* quads = (MeshQuad*)ptr;
        ptr += mesh->rCount * sizeof(MeshQuad);
        const MeshTriangle* triangles = (MeshTriangle*)ptr;
        ptr += mesh->tCount * sizeof(MeshTriangle);
        const MeshVertex* vertices = (MeshVertex*)ptr;

        wireTransform(vertices, vCount);

        // Face corners are stored as int8 steps from the previous face's last
        // corner, so the chain has to be walked exactly as faceAddMesh walks
        // it. Quads and triangles are two separate chains, each starting again
        // at the mesh's first vertex.
        int32 previous = 0;
        for (int32 i = 0; i < mesh->rCount; i++, quads++)
        {
            int32 a = previous + quads->indices[0];
            int32 b = a + quads->indices[1];
            int32 c = b + quads->indices[2];
            int32 d = c + quads->indices[3];
            previous = d;
            // Every corner checked, not just the ends: a corrupt chain can step
            // negative in the middle as easily as at either end.
            if (a < 0 || b < 0 || c < 0 || d < 0 ||
                a >= vCount || b >= vCount || c >= vCount || d >= vCount)
                continue;
            wireEdge(a, b);
            wireEdge(b, c);
            wireEdge(c, d);
            wireEdge(d, a);
        }

        previous = 0;
        for (int32 i = 0; i < mesh->tCount; i++, triangles++)
        {
            int32 a = previous + triangles->indices[0];
            int32 b = a + triangles->indices[1];
            // indices[2] is the engine's spare slot; the third step is last.
            int32 c = b + triangles->indices[3];
            previous = c;
            if (a < 0 || b < 0 || c < 0 ||
                a >= vCount || b >= vCount || c >= vCount)
                continue;
            wireEdge(a, b);
            wireEdge(b, c);
            wireEdge(c, a);
        }
    }

    // ---- per-vertex posing --------------------------------------------------
    //
    // The engine transforms a mesh block with one matrix, which is right for a
    // model built as separate rigid limbs and wrong for one whose triangles
    // straddle its joints. A DS or GBA era character is the second kind: every
    // vertex names one bone, and a triangle with corners on two bones stretches
    // between them. Freezing such a triangle onto a single bone splits the
    // vertices it shares with its neighbours and the model comes apart -- a hole
    // in a cap, with the hair behind it showing through.
    //
    // So the pose is baked here instead: each bone's matrix is captured in model
    // space, every vertex is moved by its own bone into a copy of the block held
    // in RAM, and the engine then draws that copy as ordinary rigid geometry. A
    // vertex is computed once from one rest position, so two blocks that share
    // it still agree exactly and the seam cannot open.

    static const int32 SKIN_BUFFER_BYTES = 24 * 1024;
    static const int32 SKIN_MAX_BLOCKS = 32;

    EWRAM_DATA static uint8 skinBuffer[SKIN_BUFFER_BYTES];

    struct SkinBlock
    {
        Mesh* posed;                // the RAM copy the engine draws
        const MeshVertex* rest;     // ROM vertices, each in its own bone's frame
        const uint8* bones;         // one joint index per vertex
        int32 vCount;
        int32 slot;                 // joint slot, for the visibility mask
    };

    static SkinBlock skinBlocks[SKIN_MAX_BLOCKS];
    static int32 skinBlockCount;
    static Matrix skinJoint[SKIN_MAX_BLOCKS];

    // The few vertices a source shares between bones, in increasing vertex
    // order: {u16 vertex; u8 count; u8 pad} then count x {u8 bone; u8 weight;
    // int16 x, y, z}, the position being the vertex at rest in that bone's own
    // frame. At the bind pose every bone puts it in the same place, so this
    // changes nothing until the skeleton moves.
    static const uint8* skinBlends;
    static int32 skinBlendCount;

    static const uint8* skinTable(int32 source, uint32* length)
    {
        *length = 0;
        if (!validPack() || readU32(ASSET_VIEWER_PAK + 4) < 4)
            return NULL;
        if (!(packFlags() & PACK_FLAG_SKIN))
            return NULL;
        int32 sourceCount = readU16(ASSET_VIEWER_PAK + 12);
        if (source < 0 || source >= sourceCount)
            return NULL;
        uint32 table = readU32(ASSET_VIEWER_PAK + 32);
        uint32 total = readU32(ASSET_VIEWER_PAK + 8);
        if (table == 0 || table > total ||
            uint32(sourceCount * SKIN_ENTRY_SIZE) > total - table)
            return NULL;
        const uint8* entry = ASSET_VIEWER_PAK + table + source * SKIN_ENTRY_SIZE;
        uint32 offset = readU32(entry);
        uint32 size = readU32(entry + 4);
        if (offset == 0 || size == 0 || offset > total || size > total - offset)
            return NULL;
        *length = size;
        return ASSET_VIEWER_PAK + offset;
    }

    static const uint8* blendTable(int32 source, int32* count)
    {
        *count = 0;
        uint32 unused = 0;
        if (!skinTable(source, &unused))
            return NULL;
        uint32 table = readU32(ASSET_VIEWER_PAK + 32);
        uint32 total = readU32(ASSET_VIEWER_PAK + 8);
        const uint8* entry = ASSET_VIEWER_PAK + table + source * SKIN_ENTRY_SIZE;
        uint32 offset = readU32(entry + 8);
        uint32 records = readU32(entry + 12);
        if (offset == 0 || records == 0 || offset > total)
            return NULL;
        *count = int32(records);
        return ASSET_VIEWER_PAK + offset;
    }

    static int32 meshBlockBytes(const Mesh* mesh)
    {
        return sizeof(Mesh) +
               mesh->rCount * sizeof(MeshQuad) +
               mesh->tCount * sizeof(MeshTriangle) +
               mesh->vCount * sizeof(MeshVertex);
    }

    // Copy the model's blocks into RAM once, and note where each one's rest
    // vertices and bone bytes live. Nothing here runs per frame.
    static void prepareSkin(int32 type)
    {
        skinBlockCount = 0;
        skinBlends = NULL;
        skinBlendCount = 0;
        if (type < 0 || type >= MAX_MODELS)
            return;

        int32 source = state.modelSources[type];
        if (source == 0xFF)
            return;

        uint32 length = 0;
        const uint8* bones = skinTable(source, &length);
        if (!bones)
            return;
        skinBlends = blendTable(source, &skinBlendCount);

        const Model* model = level.models + type;
        uint32 mask = state.modelMasks[type];
        int32 used = 0;
        uint32 taken = 0;

        for (int32 slot = 0; slot < model->count; slot++)
        {
            if (!((mask >> slot) & 1))
                continue;
            if (skinBlockCount >= SKIN_MAX_BLOCKS)
            {
                skinBlockCount = 0;
                return;
            }

            const Mesh* mesh = level.meshes[model->start + slot];
            int32 bytes = meshBlockBytes(mesh);
            if (used + bytes > SKIN_BUFFER_BYTES ||
                taken + uint32(mesh->vCount) > length)
            {
                // Too big to pose, or the table does not cover it. Falling back
                // to the rigid draw shows the seams but never wrong geometry.
                skinBlockCount = 0;
                return;
            }

            uint8* copy = skinBuffer + used;
            memcpy(copy, mesh, bytes);
            used += (bytes + 3) & ~3;

            SkinBlock& block = skinBlocks[skinBlockCount++];
            block.posed = (Mesh*)copy;
            block.rest = (const MeshVertex*)((const uint8*)mesh + sizeof(Mesh) +
                          mesh->rCount * sizeof(MeshQuad) +
                          mesh->tCount * sizeof(MeshTriangle));
            block.bones = bones + taken;
            block.vCount = mesh->vCount;
            block.slot = slot;
            taken += uint32(mesh->vCount);
        }
    }

    // drawNodesLerp's walk again, storing each joint's matrix instead of drawing
    // with it. The stack is left exactly as it was found: the walk's pushes and
    // pops are not balanced, so the pointer is restored rather than unwound.
    static void captureJoints(const ItemObj* item, const AnimFrame* frameA,
                              const AnimFrame* frameB, int32 frameDelta, int32 frameRate)
    {
        const Model* model = level.models + item->type;
        const ModelNode* node = level.nodes + model->nodeIndex;
        int32 meshCount = model->count;
        int32 slot = 0;

        const uint32* anglesA = (uint32*)(frameA->angles + 1);
        const uint32* anglesB = (uint32*)(frameB->angles + 1);

        Matrix* savedPtr = gMatrixPtr;
        Matrix saved = *gMatrixPtr;

        // Identity, so what comes out is each bone's place in the model rather
        // than on the screen: the engine still applies the model matrix after.
        matrixSetIdentity();

        if (frameDelta == 0)
        {
            matrixFrame(&frameA->pos, anglesA);
        }
        else
        {
            int32 t = GET_FRAME_T(frameDelta, frameRate);
            vec4s posLerp;
            posLerp.x = frameA->pos.x + ((frameB->pos.x - frameA->pos.x) * t >> 16);
            posLerp.y = frameA->pos.y + ((frameB->pos.y - frameA->pos.y) * t >> 16);
            posLerp.z = frameA->pos.z + ((frameB->pos.z - frameA->pos.z) * t >> 16);
            matrixFrameLerp(&posLerp, anglesA, anglesB, frameDelta, frameRate);
        }
        skinJoint[slot++] = matrixGet();

        while (meshCount > 1 && slot < SKIN_MAX_BLOCKS)
        {
            anglesA++;
            anglesB++;

            if (node->flags & NODE_FLAG_POP)  matrixPop();
            if (node->flags & NODE_FLAG_PUSH) matrixPush();

            if (frameDelta == 0)
            {
                matrixFrame(&node->pos, anglesA);
            }
            else
            {
                matrixFrameLerp(&node->pos, anglesA, anglesB, frameDelta, frameRate);
            }

            skinJoint[slot++] = matrixGet();
            meshCount--;
            node++;
        }

        gMatrixPtr = savedPtr;
        *gMatrixPtr = saved;
    }

    static MeshVertex* blockVertices(const SkinBlock& block)
    {
        return (MeshVertex*)((uint8*)block.posed + sizeof(Mesh) +
                block.posed->rCount * sizeof(MeshQuad) +
                block.posed->tCount * sizeof(MeshTriangle));
    }

    static void poseBlock(const SkinBlock& block)
    {
        MeshVertex* out = blockVertices(block);
        const MeshVertex* rest = block.rest;
        const uint8* bone = block.bones;

        for (int32 i = 0; i < block.vCount; i++, out++, rest++)
        {
            const Matrix& m = skinJoint[*bone++];
            // The block format keeps a quarter of a unit, which is the quantum
            // the rest position arrived in, so nothing is given away twice.
            int32 vx = rest->x << 2;
            int32 vy = rest->y << 2;
            int32 vz = rest->z << 2;
            out->x = int16(DP43(m.e00, m.e01, m.e02, m.e03, vx, vy, vz) >> (FIXED_SHIFT + 2));
            out->y = int16(DP43(m.e10, m.e11, m.e12, m.e13, vx, vy, vz) >> (FIXED_SHIFT + 2));
            out->z = int16(DP43(m.e20, m.e21, m.e22, m.e23, vx, vy, vz) >> (FIXED_SHIFT + 2));
        }
    }

    // A vertex several bones pull on: each influence moves it into place from
    // that bone's own rest frame, and the results are mixed by weight. The
    // weights sum to 256, so the mix is a shift and nothing is lost to it.
    static void applyBlend(const SkinBlock& block, int32 index,
                           const uint8* data, int32 count)
    {
        int32 px = 0, py = 0, pz = 0;
        for (int32 i = 0; i < count; i++, data += 8)
        {
            const Matrix& m = skinJoint[data[0]];
            int32 weight = data[1];
            int32 vx = int32(int16(readU16(data + 2))) << 2;
            int32 vy = int32(int16(readU16(data + 4))) << 2;
            int32 vz = int32(int16(readU16(data + 6))) << 2;
            px += weight * (DP43(m.e00, m.e01, m.e02, m.e03, vx, vy, vz) >> FIXED_SHIFT);
            py += weight * (DP43(m.e10, m.e11, m.e12, m.e13, vx, vy, vz) >> FIXED_SHIFT);
            pz += weight * (DP43(m.e20, m.e21, m.e22, m.e23, vx, vy, vz) >> FIXED_SHIFT);
        }
        MeshVertex* out = blockVertices(block) + index;
        out->x = int16(px >> (8 + 2));
        out->y = int16(py >> (8 + 2));
        out->z = int16(pz >> (8 + 2));
    }

    static void drawSkinnedNodes(const ItemObj* item, const AnimFrame* frameA,
                                 const AnimFrame* frameB, int32 frameDelta, int32 frameRate)
    {
        captureJoints(item, frameA, frameB, frameDelta, frameRate);

        const uint8* blend = skinBlends;
        int32 remaining = blend ? skinBlendCount : 0;
        int32 base = 0;

        for (int32 i = 0; i < skinBlockCount; i++)
        {
            const SkinBlock& block = skinBlocks[i];
            poseBlock(block);

            // The records are in vertex order and the blocks are walked in the
            // same order, so one pass over each is enough.
            while (remaining > 0)
            {
                int32 vertex = int32(readU16(blend));
                if (vertex >= base + block.vCount)
                    break;
                int32 count = blend[2];
                applyBlend(block, vertex - base, blend + 4, count);
                blend += 4 + count * 8;
                remaining--;
            }
            base += block.vCount;

            if ((item->visibleMask >> block.slot) & 1)
                renderMesh(block.posed);
        }
    }

    // drawNodesLerp's walk, with the mesh drawn as edges. Kept in step with it
    // by hand: the matrix stack, the pop/push flags and the visibility mask all
    // have to be applied identically or the parts come apart.
    static void wireNodes(const ItemObj* item, const AnimFrame* frameA,
                          const AnimFrame* frameB, int32 frameDelta, int32 frameRate)
    {
        const Model* model = level.models + item->type;
        const ModelNode* node = level.nodes + model->nodeIndex;
        int32 meshIndex = model->start;
        int32 meshCount = model->count;
        uint32 visibleMask = item->visibleMask;

        const uint32* anglesA = (uint32*)(frameA->angles + 1);
        const uint32* anglesB = (uint32*)(frameB->angles + 1);

        if (frameDelta == 0)
        {
            matrixFrame(&frameA->pos, anglesA);
        }
        else
        {
            int32 t = GET_FRAME_T(frameDelta, frameRate);
            vec4s posLerp;
            posLerp.x = frameA->pos.x + ((frameB->pos.x - frameA->pos.x) * t >> 16);
            posLerp.y = frameA->pos.y + ((frameB->pos.y - frameA->pos.y) * t >> 16);
            posLerp.z = frameA->pos.z + ((frameB->pos.z - frameA->pos.z) * t >> 16);
            matrixFrameLerp(&posLerp, anglesA, anglesB, frameDelta, frameRate);
        }

        if (visibleMask & 1) {
            wireMesh(meshIndex);
        }

        while (meshCount > 1)
        {
            meshIndex++;
            visibleMask >>= 1;
            anglesA++;
            anglesB++;

            if (node->flags & NODE_FLAG_POP)  matrixPop();
            if (node->flags & NODE_FLAG_PUSH) matrixPush();

            if (frameDelta == 0) {
                matrixFrame(&node->pos, anglesA);
            } else {
                matrixFrameLerp(&node->pos, anglesA, anglesB, frameDelta, frameRate);
            }

            if (visibleMask & 1) {
                wireMesh(meshIndex);
            }

            meshCount--;
            node++;
        }
    }

    static void drawCurrentModel()
    {
        int32 type = selectedType();
        int32 animationIndex = selectedAnimation();
        if (type < 0 || animationIndex < 0)
            return;

        const AnimFrame *frameA, *frameB;
        int32 frameRate, frameDelta;
        getViewerFrames(type, animationIndex, state.frameIndex, frameA, frameB, frameRate, frameDelta);
        vec3i root = interpolatedRoot(frameA, frameB, frameRate, frameDelta);
        int32 distance = X_CLAMP(state.fitDistance + state.zoom, 320, VIEW_DIST - 512);
        int32 targetX = screenOffsetToWorld(MODEL_TARGET_X - FRAME_WIDTH / 2, distance);
        int32 targetY = screenOffsetToWorld(MODEL_TARGET_Y + state.verticalOffset - FRAME_HEIGHT / 2, distance);

        ItemObj item;
        memset(&item, 0, sizeof(item));
        item.type = uint8(type);
        item.animIndex = uint16(animationIndex);
        item.frameIndex = uint16(state.frameIndex);
        item.intensity = 255;
        item.visibleMask = selectedMeshMask();

        matrixSetIdentity();
        matrixSetView(_vec3i(0, 0, 0), 0, 0);
        matrixPush();
        matrixTranslateAbs(targetX, targetY, distance);
        matrixRotateYXZ(state.pitch, state.yaw, state.roll);
        scaleCurrentBasis(state.fitScale);
        matrixTranslateRel(
            -state.fitCenter.x - root.x,
            -state.fitCenter.y - root.y,
            -state.fitCenter.z - root.z
        );

        calcLightingStatic(255 << 5);
        if (state.settings[SETTING_WIREFRAME]) {
            // Drawn straight into the framebuffer, before the ordering table
            // is flushed, so the panels and text still land on top of it.
            wireNodes(&item, frameA, frameB, frameDelta, frameRate);
        } else if (skinBlockCount > 0) {
            drawSkinnedNodes(&item, frameA, frameB, frameDelta, frameRate);
        } else {
            drawNodesLerp(&item, frameA, frameB, frameDelta, frameRate);
        }

        matrixPop();
        matrixSetIdentity();
    }

    // Name on the left, value on the right, both at fixed positions: a
    // centred row would shift sideways as ON becomes OFF.
    static void drawSettingRow(int32 y, int32 id, const char* name)
    {
        if (state.menuCursor == id) {
            drawTextCompact(MENU_ROW_CURSOR, y, ">", TEXT_ALIGN_LEFT);
        }
        drawTextCompact(MENU_ROW_NAME, y, name, TEXT_ALIGN_LEFT);
        drawTextCompact(MENU_ROW_VALUE, y, state.settings[id] ? "ON" : "OFF",
                        TEXT_ALIGN_LEFT);
    }

    static void drawUI()
    {
        drawTextCompact(0, 9, "ASSET VIEWER", TEXT_ALIGN_CENTER);
        const char* name = state.modelCount > 0 ? state.modelName : "NO MODEL";
        drawMarquee(name, 19);

        if (!state.showHelp)
        {
            drawDetails();
        }
        else
        {
            // The help is an opaque overlay. It never changes MODEL_TARGET_Y,
            // so showing it cannot move or refit the rendered asset. The model
            // and regular detail glyphs are deliberately not submitted while
            // this panel is visible; the native ordering table can otherwise
            // place very near mesh faces in front of a depth-1 UI fill.
            renderFill(0, HELP_PANEL_TOP, FRAME_WIDTH, FRAME_HEIGHT - HELP_PANEL_TOP, 0, 1);
            renderBorder(0, HELP_PANEL_TOP, FRAME_WIDTH, FRAME_HEIGHT - HELP_PANEL_TOP,
                         PANEL_LIGHT, PANEL_DARK, 1);
            drawTextCompact(0, 36, "SETTINGS", TEXT_ALIGN_CENTER);
            drawSettingRow(45, SETTING_WIREFRAME, "WIREFRAME");
            drawTextCompact(0, 54, "UP/DN PICK  A TOGGLE", TEXT_ALIGN_CENTER);

            drawTextCompact(0, 63,  "CONTROLS", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 72,  "TAP LEFT/RIGHT AUTO Y", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 81,  "HOLD LEFT/RIGHT ROTATE Y", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 90,  "DPAD UP/DN ROTATE X", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 99,  "SEL+DPAD ROTATE Z ZOOM", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 108, "SEL+L/R BUTTONS MOVE Y", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 117, "L/R PREV/NEXT MODEL", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 126, "START NEXT SEL+START PREV", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 135, "A PLAY/PAUSE SEL+A STEP", TEXT_ALIGN_CENTER);
            drawTextCompact(0, 144, "B RESET  SELECT CLOSE", TEXT_ALIGN_CENTER);
        }
    }
}

void assetViewerInit()
{
    memset(&AssetViewer::state, 0, sizeof(AssetViewer::state));

    gSettings.version = SETTINGS_VER;
    gSettings.controls_vibration = 1;
    gSettings.controls_swap = 0;
    gSettings.audio_sfx = 0;
    gSettings.audio_music = 0;
    gSettings.video_gamma = 0;
    gSettings.video_fps = 0;
    gSettings.video_vsync = 1;
    osLoadSettings();

    drawInit();
    AssetViewer::state.playing = true;
    AssetViewer::state.previousKeys = keys;
    AssetViewer::buildUnifiedCatalog();
    AssetViewer::loadSelectedModel();
}

void assetViewerUpdate(int32 frames)
{
    using namespace AssetViewer;

    frames = X_CLAMP(frames, 0, MAX_UPDATE_FRAMES);
    uint32 pressed = keys & ~state.previousKeys;
    uint32 released = state.previousKeys & ~keys;
    bool select = (keys & IK_SELECT) != 0;
    state.uiTicks += frames;

    if (pressed & IK_SELECT)
        state.selectUsed = false;
    if (select && (keys & ~IK_SELECT))
        state.selectUsed = true;
    if ((released & IK_SELECT) && !state.selectUsed)
        state.showHelp = !state.showHelp;

    // With the panel open the D-pad and A drive the settings instead of the
    // model. Nothing else is read: the model is not even drawn behind the
    // panel, so rotating or stepping it while it cannot be seen would only
    // surprise whoever closes the menu again.
    if (state.showHelp)
    {
        if (!select)
        {
            if ((pressed & IK_UP) && state.menuCursor > 0)
                state.menuCursor--;
            if ((pressed & IK_DOWN) && state.menuCursor < SETTING_COUNT - 1)
                state.menuCursor++;
            if (pressed & IK_A)
                state.settings[state.menuCursor] = !state.settings[state.menuCursor];
        }
        state.previousKeys = keys;
        return;
    }

    if (!select && (pressed & IK_L)) {
        selectModel(-1);
        state.previousKeys = keys;
        return;
    }
    if (!select && (pressed & IK_R)) {
        selectModel(1);
        state.previousKeys = keys;
        return;
    }

    int32 ticks = X_MAX(frames, 1);
    int32 rotation = ROTATION_STEP * ticks;
    if (select) {
        state.yawHoldTicks = 0;
        state.yawPressDirection = 0;
        state.yawTapEligible = false;
        state.yawManual = false;
        if (keys & IK_LEFT)  state.roll -= rotation;
        if (keys & IK_RIGHT) state.roll += rotation;
        if (keys & IK_UP)    adjustZoom(-1, frames);
        if (keys & IK_DOWN)  adjustZoom(1, frames);
        if (keys & IK_L)     adjustVertical(-1, frames);
        if (keys & IK_R)     adjustVertical(1, frames);
    } else {
        if (state.yawPressDirection == 0) {
            if ((pressed & IK_LEFT) && !(keys & IK_RIGHT))
                state.yawPressDirection = -1;
            else if ((pressed & IK_RIGHT) && !(keys & IK_LEFT))
                state.yawPressDirection = 1;

            if (state.yawPressDirection != 0) {
                state.yawHoldTicks = 0;
                state.yawTapEligible = true;
                state.yawManual = false;
            }
        }

        if (state.yawPressDirection != 0) {
            uint32 yawKey = state.yawPressDirection < 0 ? IK_LEFT : IK_RIGHT;
            if ((keys & (IK_LEFT | IK_RIGHT)) == (IK_LEFT | IK_RIGHT) ||
                (keys & ~(IK_LEFT | IK_RIGHT)))
                state.yawTapEligible = false;

            if (keys & yawKey) {
                state.yawHoldTicks += ticks;
                if (state.yawHoldTicks >= YAW_HOLD_TICKS) {
                    state.yawManual = true;
                    state.autoYawDirection = 0;
                    state.yaw += state.yawPressDirection * rotation;
                }
            }

            if (released & yawKey) {
                if (state.yawTapEligible && !state.yawManual) {
                    state.autoYawDirection =
                        state.autoYawDirection == state.yawPressDirection
                        ? 0 : state.yawPressDirection;
                }
                state.yawHoldTicks = 0;
                state.yawPressDirection = 0;
                state.yawTapEligible = false;
                state.yawManual = false;
            }
        }

        if (keys & IK_UP)    state.pitch -= rotation;
        if (keys & IK_DOWN)  state.pitch += rotation;
    }

    if (state.autoYawDirection != 0 && state.yawPressDirection == 0)
        state.yaw += state.autoYawDirection * AUTO_ROTATION_STEP * ticks;

    if (pressed & IK_B)
        resetOrientation();

    if (select) {
        if (pressed & IK_START)
            selectAnimation(-1);
        if (pressed & IK_A) {
            state.playing = false;
            stepFrames(1);
        }
    } else {
        if (pressed & IK_START)
            selectAnimation(1);
        if (pressed & IK_A)
            state.playing = !state.playing;
    }

    if (state.playing)
        stepFrames(X_MAX(frames, 1));

    updateLevel(frames);
    state.previousKeys = keys;
}

void assetViewerRender()
{
    setViewport(RectMinMax(0, 0, FRAME_WIDTH, FRAME_HEIGHT));
    clear();
    if (!AssetViewer::state.showHelp)
        AssetViewer::drawCurrentModel();
    AssetViewer::drawUI();
    flush();
}

#endif

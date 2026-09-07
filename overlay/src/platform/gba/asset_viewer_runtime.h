#ifndef H_ASSET_VIEWER_RUNTIME
#define H_ASSET_VIEWER_RUNTIME

#include "common.h"

// level.h normally receives this storage from room.h through game.h. The
// viewer has no rooms, sectors or gameplay, but readLevel resets the counter.
EWRAM_DATA int32 dynSectorsCount;

#include "level.h"

// Level keeps its original fixed-size ItemObj storage. Constructing that
// storage requires the base vtable even though viewer-only packs contain no
// items and never execute gameplay callbacks.
void ItemObj::activate() {}
void ItemObj::deactivate() {}
void ItemObj::hit(int32 damage, const vec3i &point, int32 soundId) {}
void ItemObj::collide(Lara* lara, CollisionInfo* cinfo) {}
void ItemObj::update() {}
void ItemObj::draw() {}
uint8* ItemObj::save(uint8* data) { return data; }
uint8* ItemObj::load(uint8* data) { return data; }

void drawInit()
{
    renderInit();
}

void drawFree()
{
    renderFree();
}

void drawLevelInit()
{
    renderLevelInit();
}

void drawLevelFree()
{
    renderLevelFree();
}

void calcLightingStatic(int32 intensity)
{
    gLightAmbient = intensity - 4096;
    Matrix &matrix = matrixGet();
    int32 fogZ = matrix.e23 >> (FIXED_SHIFT - MATRIX_FIXED_SHIFT);
    if (fogZ > FOG_MIN) {
        gLightAmbient += (fogZ - FOG_MIN) << FOG_SHIFT;
        gLightAmbient = X_MIN(gLightAmbient, 8191);
    }
}

static void drawViewerMesh(int32 meshIndex)
{
    renderMesh(level.meshes[meshIndex]);
}

static void drawViewerNodes(const ItemObj* item, const AnimFrame* frame)
{
    const Model* model = level.models + item->type;
    const ModelNode* node = level.nodes + model->nodeIndex;
    int32 meshIndex = model->start;
    int32 meshCount = model->count;
    uint32 visibleMask = item->visibleMask;
    const uint32* angles = (const uint32*)(frame->angles + 1);

    matrixFrame(&frame->pos, angles);
    if (visibleMask & 1)
        drawViewerMesh(meshIndex);

    while (meshCount > 1)
    {
        meshIndex++;
        visibleMask >>= 1;
        angles++;
        if (node->flags & NODE_FLAG_POP)  matrixPop();
        if (node->flags & NODE_FLAG_PUSH) matrixPush();
        matrixFrame(&node->pos, angles);
        if (visibleMask & 1)
            drawViewerMesh(meshIndex);
        meshCount--;
        node++;
    }
}

void drawNodesLerp(const ItemObj* item, const AnimFrame* frameA,
                   const AnimFrame* frameB, int32 frameDelta, int32 frameRate)
{
    if (frameDelta == 0) {
        drawViewerNodes(item, frameA);
        return;
    }

    const Model* model = level.models + item->type;
    const ModelNode* node = level.nodes + model->nodeIndex;
    int32 meshIndex = model->start;
    int32 meshCount = model->count;
    uint32 visibleMask = item->visibleMask;
    const uint32* anglesA = (const uint32*)(frameA->angles + 1);
    const uint32* anglesB = (const uint32*)(frameB->angles + 1);
    int32 t = GET_FRAME_T(frameDelta, frameRate);

    vec4s pos;
    pos.x = frameA->pos.x + ((frameB->pos.x - frameA->pos.x) * t >> 16);
    pos.y = frameA->pos.y + ((frameB->pos.y - frameA->pos.y) * t >> 16);
    pos.z = frameA->pos.z + ((frameB->pos.z - frameA->pos.z) * t >> 16);
    matrixFrameLerp(&pos, anglesA, anglesB, frameDelta, frameRate);
    if (visibleMask & 1)
        drawViewerMesh(meshIndex);

    while (meshCount > 1)
    {
        meshIndex++;
        visibleMask >>= 1;
        anglesA++;
        anglesB++;
        if (node->flags & NODE_FLAG_POP)  matrixPop();
        if (node->flags & NODE_FLAG_PUSH) matrixPush();
        matrixFrameLerp(&node->pos, anglesA, anglesB, frameDelta, frameRate);
        if (visibleMask & 1)
            drawViewerMesh(meshIndex);
        meshCount--;
        node++;
    }
}

#endif

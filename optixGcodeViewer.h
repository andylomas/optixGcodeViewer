//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// Optix stuff

#define AMBOCC_JITTER_SAMPLES 1
#define DEBUG_MODE 0

enum Culling
{
    CULLING_NONE,
    CULLING_FRONTFACE,
    CULLING_BACKFACE
};

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

enum LensType
{
    LENS_TYPE_PINHOLE,
    LENS_TYPE_FULLDOME,
    LENS_TYPE_LOOKING_GLASS_QUILT
};

struct Params
{
    uchar4*                frame_buffer;
    unsigned char*         depth_buffer;
    unsigned int           width;
    unsigned int           height;
    float3                 eye, U, V, W;
    OptixTraversableHandle handle;
    int                    antialias_samples;
    int                    ambocc_samples;
    float                  ambocc_minZ;
    float                  ambocc_gamma;
    bool                   use_geo_normal;
    bool                   anaglyph_mode;
    float                  scene_epsilon;
    int                    subframe_index;
    LensType               lens_type;
    float                  view_distance;
    float                  view_offset_angle;
    float                  view_offset;
    float                  lens_angle;
    float                  convergence_radius;
    unsigned int           quilt_columns;
    unsigned int           quilt_rows;
    float3*                d_vertices;
    uint3*                 d_triangleIndices;
    float3*                d_normals;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};


struct MissData
{
};


struct HitGroupData
{
};

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

#include <optix.h>

#include "optixGcodeViewer.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "random.h"
#include "myHelpers.h"

extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void setPayload( float3 p, float a = 1.f )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
    optixSetPayload_3( float_as_int( a ) );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int2 pixel_idx  = make_int2(idx.x, idx.y);
    const unsigned int image_index  = params.width*pixel_idx.y+pixel_idx.x;

    float3 eye = params.eye;
    float3 direction;

    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;

    float4 payload_sum = make_float4(0);
    float aa_factor = 1.f / params.antialias_samples;
    float aa_offset = 0.5f * (params.antialias_samples - 1.f);
    for (int sample_x = 0; sample_x < params.antialias_samples; sample_x++)
        for (int sample_y = 0; sample_y < params.antialias_samples; sample_y++)
        {
            const float2      d = 2.0f * make_float2(
                (static_cast< float >( idx.x ) + aa_factor * (static_cast< float >(sample_x) - aa_offset)) / static_cast< float >( dim.x ),
                (static_cast< float >( idx.y ) + aa_factor * (static_cast< float >(sample_y) - aa_offset)) / static_cast< float >( dim.y )
            ) - 1.0f;

            int num_passes = params.anaglyph_mode ? 2 : 1;

            for (int pass_idx = 0; pass_idx < num_passes; pass_idx++)
            {
                unsigned int seed = tea<16>(image_index, params.subframe_index + 17 * pass_idx);

                // Flag to indicate whether we should trace the current sample
                bool trace_sample = true;
                float view_offset = params.view_offset;
                float view_offset_angle = params.view_offset_angle;
                if (params.anaglyph_mode)
                {
                    if (pass_idx == 1)
                    {
                        view_offset = -view_offset;
                        view_offset_angle = -view_offset_angle;
                    }
                    else if (pass_idx == 2)
                    {
                        view_offset = 0.0f;
                        view_offset_angle = 0.0f;
                    }
                }

                if (params.lens_type == LENS_TYPE_PINHOLE)
                {
                    // Standard pin-hole camera
                    if (params.view_offset == 0)
                    {
                        direction = normalize( d.x * U + d.y * V + W );
                    }
                    else
                    {
                        // We've got a shift lens to do things like stereo cameras or multiple views for lenticular renders
                        float3 view_point = eye + params.view_distance * (d.x*U + d.y*V + W) / length(W);
                        eye += params.view_offset * normalize(U);
                        direction = normalize(view_point - eye);
                    }
                }
                else if (params.lens_type == LENS_TYPE_FULLDOME)
                {
                    // Reverse fish-eye suitable for fulldome projection
                    // Assumes that the projection centre for spherically inverting the world is at (0, 0, 0)
                    float view_distance = params.view_distance;


                    float2 dd = (params.width > params.height) ?
                        make_float2(d.x * (float)params.width / (float)params.height, d.y) :
                        make_float2(d.x, d.y * (float)params.height / (float)params.height);

                    float phi = (dd.x == 0 && dd.y == 0) ? 0.f : atan2f(dd.y, dd.x);
                    float theta = params.lens_angle * length(dd);

                    if (theta <= params.lens_angle)
                    {
                        float3 normU = normalize(U);
                        float3 normV = normalize(V);
                        float3 normW = normalize(W);

                        float3 projection_centre = make_float3(0,0,0);

                        eye = projection_centre + params.view_distance * (sinf(theta) * cosf(phi + view_offset_angle) * normU +
                                                                          sinf(theta) * sinf(phi + view_offset_angle) * normV +
                                                                         -cosf(theta) * normW);

                        float3 view_point = projection_centre + params.convergence_radius * (sinf(theta) * cosf(phi) * normU +
                                                                                                    sinf(theta) * sinf(phi) * normV +
                                                                                                   -cosf(theta) * normW);
                        direction = normalize(view_point - eye);
                    }
                    else
                    {
                        trace_sample = false;
                    }
                }
                else if (params.lens_type == LENS_TYPE_LOOKING_GLASS_QUILT)
                {
                    // Looking Glass quilt

                    // Calculate which tile of the quilt the current sub-pixel is in
                    int column = params.quilt_columns * (0.5f + 0.5f * d.x);
                    int row = params.quilt_rows * (0.5f + 0.5f * d.y);
                    int tile_index = column + params.quilt_columns * row;
                    int num_tiles = params.quilt_columns * params.quilt_rows;

                    // Calculate ray origin and ray_direction
                    float ddx, ddy, dummy;
                    ddx = modf((0.5f * d.x + 0.5f) * params.quilt_columns, &dummy) * 2.0f - 1.0f;
                    ddy = modf((0.5f * d.y + 0.5f) * params.quilt_rows, &dummy) * 2.0f - 1.0f;
                    float2 dd = make_float2(ddx, ddy);

                    // We've got a shift lens to do things like stereo cameras or multiple views for lentivular renders
                    float horiz_offset_factor = (2.0f * tile_index) / (num_tiles - 1.0f) - 1.0f;
                    float3 view_point = eye + params.view_distance * (dd.x*U + dd.y*V + W) / length(W);
                    eye += horiz_offset_factor * params.view_offset * normalize(U);
                    direction = normalize(view_point - eye);
                }

                float4 payload_rgba = make_float4( 0.f );
                unsigned int p0, p1, p2, p3;
                p0 = float_as_int( payload_rgba.x );
                p1 = float_as_int( payload_rgba.y );
                p2 = float_as_int( payload_rgba.z );
                p3 = float_as_int( payload_rgba.z );
                optixTrace(
                    params.handle,
                    eye,
                    direction,
                    0.00f,
                    1e16f,
                    0.0f,                // rayTime
                    OptixVisibilityMask( 1 ),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_RADIANCE,                   // SBT offset
                    RAY_TYPE_COUNT,                      // SBT stride
                    RAY_TYPE_RADIANCE,                   // missSBTIndex
                    p0, p1, p2, p3, seed);
                payload_rgba.x = int_as_float( p0 );
                payload_rgba.y = int_as_float( p1 );
                payload_rgba.z = int_as_float( p2 );
                payload_rgba.w = int_as_float( p3 );

                if (params.anaglyph_mode)
                {
                    if (pass_idx == 0)
                    {
                        payload_sum += make_float4(payload_rgba.x, 0, 0, 1);
                    }
                    else if (pass_idx == 1)
                    {
                        payload_sum += make_float4(0, payload_rgba.x, payload_rgba.x, 0);
                    }
                    else
                    {
                        payload_sum += make_float4(0, 0, payload_rgba.x, 0);
                    }
                }
                else
                {
                    payload_sum += payload_rgba;
                }
            }
        }

    float num_samples = params.antialias_samples * params.antialias_samples;
    params.frame_buffer[idx.y * params.width + idx.x] = make_color( payload_sum / num_samples );
}


extern "C" __global__ void __closesthit__ch()
{
    // Calculate hit_point
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();
    const float3 hit_point = ray_orig + ray_t * ray_dir;

    // Calculate the geometry normal from the triangles
    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data );
    data[1] -= data[0];
    data[2] -= data[0];
    float3 geo_normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x );
    if (optixIsTriangleBackFaceHit())
        geo_normal = -geo_normal;

    float3 normal = normalize(geo_normal);

    // Scale the normal before setting the payload
    setPayload( (0.5f * normal + make_float3(0.5f)) );
}


static __device__ __inline__ void ambocc_sample_hemisphere( const float u1, const float u2, float3& p, float minZ = 0.f)
{
    //p.z = sqrtf(u1 / (1.f - minZ) + minZ);
    p.z = sqrtf(u1);
    const float phi = 2.0f*M_PIf * u2;
    const float r = sqrtf(1 - p.z * p.z);;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);
}


extern "C" __global__ void __anyhit__cull_backface()
{
    if (optixIsTriangleBackFaceHit())
        optixIgnoreIntersection();
}


extern "C" __global__ void __anyhit__cull_frontface()
{
    if (optixIsTriangleFrontFaceHit())
        optixIgnoreIntersection();
}


extern "C" __global__ void __closesthit__radiance_ambocc()
{
    // Calculate hit_point
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();
    const float3 hit_point = ray_orig + ray_t * ray_dir;
    unsigned int seed = optixGetPayload_4();
    const unsigned int prim_idx = optixGetPrimitiveIndex();


    // Calculate the geometry normal from the triangles
    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
        optixGetRayTime(), data );
    data[1] -= data[0];
    data[2] -= data[0];
    float3 geo_normal = make_float3(
        data[1].y*data[2].z - data[1].z*data[2].y,
        data[1].z*data[2].x - data[1].x*data[2].z,
        data[1].x*data[2].y - data[1].y*data[2].x );
    if (optixIsTriangleBackFaceHit())
        geo_normal = -geo_normal;

    // Calculate normal for ambocc calculations
    float3 normal;
    if (params.use_geo_normal) {
        normal = normalize(geo_normal);
    }
    else {
        const float2 barycentrics    = optixGetTriangleBarycentrics();

        const float3 n0 = params.d_normals[params.d_triangleIndices[prim_idx].x];
        const float3 n1 = params.d_normals[params.d_triangleIndices[prim_idx].y];
        const float3 n2 = params.d_normals[params.d_triangleIndices[prim_idx].z];

        normal = normalize( (1.0f - barycentrics.x - barycentrics.y) * n0 + barycentrics.x * n1 + barycentrics.y * n2 );
    }

    // float3 color_mult = 0.1f * make_float3(prim_idx % 2, (prim_idx >> 1) % 2, (prim_idx >> 2) % 2);
    // if (prim_idx < 4) printf("idx: %d\n", prim_idx);
    float3 color_mult = make_float3(1);

    // Ambient occlusion
    int const ambocc_samplesX = params.ambocc_samples ? params.ambocc_samples : 1;
    int const ambocc_samplesY = params.ambocc_samples ? 4 * params.ambocc_samples : 1;

    float multX = 1.0f / ambocc_samplesX;
    float multY = 1.0f / ambocc_samplesY;

    // Get orthonormal basis using the normal
    Onb onb(normal);
    float amboccTotal = 0.f;
    int numSamples = 0;
    for (int i = 0; i < ambocc_samplesX; i++) {
        for (int j = 0; j < ambocc_samplesY; j++) {
            float3 dir;

#if AMBOCC_JITTER_SAMPLES
            float z1 = (i + rnd(seed)) * multX;
            float z2 = (j + rnd(seed)) * multY;
#else
            float z1 = (i + 0.5f) * multX;
            float z2 = (j + 0.5f) * multY;
#endif

            ambocc_sample_hemisphere(z1, z2, dir, params.ambocc_minZ);

            // Check that this sample isn't in the oposite direction to geo_normal
            //if (dot(dir, geo_normal) >= 0)
            {
                onb.inverse_transform(dir);

                float attenuation = 1.0f; // Default attenuation if ray doesn't hit anything

                optixTrace(
                    params.handle,
                    hit_point,
                    dir,
                    params.scene_epsilon,
                    1e16f,
                    0.f,
                    OptixVisibilityMask( 1 ),
                    OPTIX_RAY_FLAG_NONE,
                    RAY_TYPE_OCCLUSION,
                    RAY_TYPE_COUNT,
                    RAY_TYPE_OCCLUSION,
                    reinterpret_cast<unsigned int&>(attenuation) );

                    amboccTotal += attenuation;
                    numSamples++;
            }
        }
    }

    // pass the ambient occlusion result back as payload 0
    setPayload( color_mult * make_float3(powf(amboccTotal / numSamples, params.ambocc_gamma)) );

    // pass the depth back as payload 1
    //optixSetPayload_1( float_as_int(ray_t) );

    // pass the seed value back at payload 4
    optixSetPayload_4( seed );
}


extern "C" __global__ void __anyhit__occlusion_ambocc()
{
    optixSetPayload_0( float_as_int(0.f) );
    optixTerminateRay();
}

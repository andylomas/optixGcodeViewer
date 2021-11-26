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

//#include <assert.h>
#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixGcodeViewer.h"
#include "gcodeReader.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#undef NDEBUG

bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;
float3 camera_pos = make_float3( 4.2f, 3.0f, 4.2f );
float3 camera_interest = make_float3( 0.0f, 0.18f, 0.0f );
float camera_vfov = 20.0f;
float camera_aspect = 0;

// Should be able to remove this, but also need to remove Optix anyhit culling programs
Culling culling_mode = CULLING_NONE;

// Mouse state
int32_t mouse_button = -1;

// Mesh data
unsigned int numVertices = -1;
unsigned int numFaces = -1;
unsigned int numTriangles = -1;
bool useGeoNormal = false;

// G-code conversion parameters
float gcodeNozzleWidth = 0.0f;
float gcodeLayerHeight = 0.0f;

bool saveImageKeyPressed = false;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

struct DynamicGeometryState
{
    OptixDeviceContext context = 0;

    size_t                         temp_buffer_size = 0;
    CUdeviceptr                    d_temp_buffer = 0;
    CUdeviceptr                    d_vertices = 0;
    CUdeviceptr                    d_triangleIndices = 0;
    CUdeviceptr                    d_normals = 0;
    CUdeviceptr                    d_instances = 0;

    unsigned int                   triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;

    OptixBuildInput                ias_instance_input = {};
    OptixBuildInput                triangle_input = {};

    OptixTraversableHandle         ias_handle;
    OptixTraversableHandle         marching_cubes_gas_handle;

    CUdeviceptr                    d_ias_output_buffer = 0;
    CUdeviceptr                    d_marching_cubes_gas_output_buffer;

    size_t                         ias_output_buffer_size = 0;;
    size_t                         marching_cubes_gas_output_buffer_size = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group;
    OptixProgramGroup              default_miss_prog_group = 0;
    OptixProgramGroup              colour_normal_hit_prog_group = 0;
    OptixProgramGroup              ambocc_radiance_hit_prog_group  = 0;
    OptixProgramGroup              ambocc_occlusion_hit_prog_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params;

    float                          time = 0.0f;

    OptixShaderBindingTable        sbt_colour_normal = {};
    OptixShaderBindingTable        sbt_ambocc = {};
};


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

float degToRad(float v)
{
    return 0.017453292f * v;
}


float radToDeg(float v)
{
    return 57.29578f * v;
}


bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const int32_t INST_COUNT = 1;

struct Instance
{
    float m[12];
};

const std::array<Instance, INST_COUNT> g_instances =
{ {
    {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}}
} };


void readPlyFile( DynamicGeometryState& state, std::string filePath ) {
    std::cout << "Reading data from PLY file " << filePath << "\n";

    ofMesh myMesh;
    myMesh.readPlyFile( filePath );
    myMesh.flipYZ();
    myMesh.fitToBounds();

    // Get numVertices and numTriangles
    numVertices = myMesh.getNumVertices();
    numTriangles = myMesh.getNumTriangles();

    std::cout << "  numVertices: " << numVertices << "\n";
    std::cout << "  numTriangles: " << numTriangles << "\n";

    // Copy data to the GPU
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_vertices ), numVertices * sizeof(float3) ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_vertices, myMesh.getVertexData(), numVertices * sizeof(float3), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_triangleIndices ), numTriangles * sizeof(uint3) ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_triangleIndices, myMesh.getTriangleIndexData(), numTriangles * sizeof(uint3), cudaMemcpyHostToDevice ) );

    if (myMesh.getNumNormals() > 0) {
        assert(myMesh.getNumNormals() == numVertices);
        CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_normals ), numVertices * sizeof(float3) ) );
        CUDA_CHECK( cudaMemcpy( ( void* )state.d_normals, myMesh.getNormalData(), numVertices * sizeof(float3), cudaMemcpyHostToDevice ) );
    }

    // Set pointers in state.params
    state.params.d_vertices = (float3*)state.d_vertices;
    state.params.d_normals = (float3*)state.d_normals;
    state.params.d_triangleIndices = (uint3*)state.d_triangleIndices;
}


void readGcodeFile( DynamicGeometryState& state, std::string filePath, float nozzleWidth, float layerHeight) {
    std::cout << "Reading data from G-code file " << filePath << "\n";

    // Read gcode data file and convert to mesh
    GcodeReader myReader;
    myReader.readPathsFromGcode(filePath);
    myReader.cleanupPaths();
    myReader.convertPathsToMesh(nozzleWidth, layerHeight, 8, 3);
    myReader.getMesh().flipYZ();
    myReader.getMesh().fitToBounds();

    // Get numVertices and numTriangles
    numVertices = myReader.getMesh().getNumVertices();
    numTriangles = myReader.getMesh().getNumTriangles();

    std::cout << "  numVertices: " << numVertices << "\n";
    std::cout << "  numTriangles: " << numTriangles << "\n";

    // Copy data to the GPU
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_vertices ), numVertices * sizeof(float3) ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_vertices, myReader.getMesh().getVertexData(), numVertices * sizeof(float3), cudaMemcpyHostToDevice ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_triangleIndices ), numTriangles * sizeof(uint3) ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_triangleIndices, myReader.getMesh().getTriangleIndexData(), numTriangles * sizeof(uint3), cudaMemcpyHostToDevice ) );

    if (myReader.getMesh().getNumNormals() > 0) {
        assert(myReader.getMesh().getNumNormals() == numVertices);
        CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_normals ), numVertices * sizeof(float3) ) );
        CUDA_CHECK( cudaMemcpy( ( void* )state.d_normals, myReader.getMesh().getNormalData(), numVertices * sizeof(float3), cudaMemcpyHostToDevice ) );
    }

    // Set pointers in state.params
    state.params.d_vertices = (float3*)state.d_vertices;
    state.params.d_normals = (float3*)state.d_normals;
    state.params.d_triangleIndices = (uint3*)state.d_triangleIndices;
}


void readMeshFile( DynamicGeometryState& state, std::string filePath ) {
    // Detect file type from extension and call relevant reader
    if (hasEnding(filePath, ".ply")) {
        readPlyFile( state, filePath );
    }
    else if (hasEnding(filePath, ".gcode")) {
        readGcodeFile( state, filePath, gcodeNozzleWidth, gcodeLayerHeight );
    }
    else {
        std::cout << "ERROR: unsupported file type: " << filePath << "\n";
        exit(1);
    }
}



//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking( static_cast< int >( xpos ), static_cast< int >( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast< Params* >( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast< int >( xpos ), static_cast< int >( ypos ), params->width, params->height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast< Params* >( glfwGetWindowUserPointer( window ) );
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
        else if( key == GLFW_KEY_G )
        {
            useGeoNormal = !useGeoNormal;
            camera_changed = true;
        }
        else if( key == GLFW_KEY_S )
        {
            saveImageKeyPressed = true;
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( ( int )yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " <infile> [options]\n";
    std::cerr << "Options: --outfile | -f <filename>        File for image output\n";
    std::cerr << "         --interactive                    Use interactive interface even if outfile specified\n";
    std::cerr << "         --antialias | -aa <samples>      Antialising samples\n";
    std::cerr << "         --ambocc | -ao <samples>         Ambient occlusion samples\n";
    std::cerr << "         --gamma | -g <gamma>             Gamma for the rendered image\n";
    std::cerr << "         --geonormal | -gn                Use geometry normals (sharp polygon edges)\n";
    std::cerr << "         --cam_pos <x> <y> <z>            Camera position\n";
    std::cerr << "         --cam_interest <x> <y> <z>       Camera interest\n";
    std::cerr << "         --vfov <fov>                     Vertical field of view\n";
    std::cerr << "         --aspect <aspect>                Aspect ratio\n";
    std::cerr << "         --help | -h                      Print this usage message\n";
    std::cerr << "\n";
    std::cerr << "Keyboard controls: ESCAPE, Q              Quit program\n";
    std::cerr << "                   G                      Toggle geometry normals\n";
    std::cerr << "                   S                      Save image\n";
    exit( 0 );
}


void initLaunchParams( DynamicGeometryState& state )
{
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped
    state.params.depth_buffer = nullptr;  // Will be set when output buffer is mapped = 0u;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_params ), sizeof( Params ) ) );
}


void handleCameraUpdate( Params& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    if (camera_aspect == 0)
    {
        camera.setAspectRatio( static_cast< float >( params.width ) / static_cast< float >( params.height ) );
    }
    else
    {
        camera.setAspectRatio( camera_aspect );
    }

    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if (DEBUG_MODE) printf("Calling updateState()\n");

    handleCameraUpdate( params );
    handleResize( output_buffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, DynamicGeometryState& state )
{
    if (DEBUG_MODE) printf("Calling launchSubframe()\n");

    state.params.use_geo_normal = useGeoNormal;

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast< void* >( state.d_params ),
        &state.params, sizeof( Params ),
        cudaMemcpyHostToDevice, state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast< CUdeviceptr >( state.d_params ),
        sizeof( Params ),
        //&state.sbt_colour_normal,
        &state.sbt_ambocc,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ) );
    cudaDeviceSynchronize();
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window )
{
    if (DEBUG_MODE) printf("Calling displaySubframe()\n");

    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


void initCameraState()
{
    camera.setEye( camera_pos );
    camera.setLookat( camera_interest );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( camera_vfov );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame(
        make_float3( 1.0f, 0.0f, 0.0f ),
        make_float3( 0.0f, 0.0f, 1.0f ),
        make_float3( 0.0f, 1.0f, 0.0f )
    );
    trackball.setGimbalLock( true );
}


void createContext( DynamicGeometryState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    state.context = context;
}

void updateMeshAccel( DynamicGeometryState& state )
{
    if (DEBUG_MODE) printf("Calling updateMeshAccel()\n");

    // Update the numVertices value in triangleArray
    state.triangle_input.triangleArray.numVertices = numVertices;
    state.triangle_input.triangleArray.vertexBuffers = (numVertices != 0) ? &state.d_vertices : NULL;

    OptixAccelBuildOptions gas_accel_options = {};
    //gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    //gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

    // Update GAS
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OPTIX_CHECK( optixAccelBuild(
        state.context,
        state.stream,                       // CUDA stream
        &gas_accel_options,
        &state.triangle_input,
        1,                                  // num build inputs
        state.d_temp_buffer,
        state.temp_buffer_size,
        state.d_marching_cubes_gas_output_buffer,
        state.marching_cubes_gas_output_buffer_size,
        &state.marching_cubes_gas_handle,
        nullptr,                           // emitted property list
        0                                   // num emitted properties
    ) );

    // Update the IAS
    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;;

    OPTIX_CHECK( optixAccelBuild( state.context, state.stream, &ias_accel_options, &state.ias_instance_input, 1, state.d_temp_buffer, state.temp_buffer_size,
        state.d_ias_output_buffer, state.ias_output_buffer_size, &state.ias_handle, nullptr, 0 ) );

    CUDA_SYNC_CHECK();
}

void buildMeshAccel( DynamicGeometryState& state )
{
    // Build an AS over the triangles.
    // We use un-indexed triangles so we can explode the sphere per triangle.
    state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    state.triangle_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    state.triangle_input.triangleArray.numVertices = static_cast< uint32_t >( numVertices );
    state.triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    state.triangle_input.triangleArray.flags = &state.triangle_flags;
    state.triangle_input.triangleArray.numSbtRecords = 1;
    state.triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    state.triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    if (state.d_triangleIndices) {
        state.triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        state.triangle_input.triangleArray.indexStrideInBytes = sizeof( int3 );
        state.triangle_input.triangleArray.numIndexTriplets = static_cast< uint32_t >( numTriangles );
        state.triangle_input.triangleArray.indexBuffer = state.d_triangleIndices;
    }

    OptixAccelBuildOptions accel_options = {};
    //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute the memory usage for the GAS
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &state.triangle_input,
        1,  // num_build_inputs
        &gas_buffer_sizes
    ) );

    printf("GAS: outputSize %lu\n", gas_buffer_sizes.outputSizeInBytes / (1024 * 1024));
    printf("     tempSize %lu\n", gas_buffer_sizes.tempSizeInBytes / (1024 * 1024));
    printf("     tempUpdateSize %lu\n", gas_buffer_sizes.tempUpdateSizeInBytes / (1024 * 1024));
    printf("     total size per triangle: %f\n", (float)(gas_buffer_sizes.outputSizeInBytes + gas_buffer_sizes.tempSizeInBytes + gas_buffer_sizes.tempUpdateSizeInBytes) / (float)(numTriangles/3));

    state.marching_cubes_gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    // Allocate GPU temp buffer to create the GAS
    state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &state.d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_buffer_temp_output_gas_and_compacted_size ),
        compactedSizeOffset + 8
    ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( ( char* )d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,                                  // CUDA stream
        &accel_options,
        &state.triangle_input,
        1,                                  // num build inputs
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.marching_cubes_gas_handle,
        NULL,                      // emitted property list
        0                                   // num emitted properties
    ) );

    //     OPTIX_CHECK( optixAccelBuild(
    //     state.context,
    //     0,                                  // CUDA stream
    //     &accel_options,
    //     &state.triangle_input,
    //     1,                                  // num build inputs
    //     state.d_temp_buffer,
    //     gas_buffer_sizes.tempSizeInBytes,
    //     d_buffer_temp_output_gas_and_compacted_size,
    //     gas_buffer_sizes.outputSizeInBytes,
    //     &state.marching_cubes_gas_handle,
    //     &emitProperty,                      // emitted property list
    //     1                                   // num emitted properties
    // ) );

    state.d_marching_cubes_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

    // Build the IAS

    std::vector<OptixInstance> instances( g_instances.size() );

    for( size_t i = 0; i < g_instances.size(); ++i )
    {
        memcpy( instances[i].transform, g_instances[i].m, sizeof( float ) * 12 );
        instances[i].sbtOffset = static_cast< unsigned int >( i );
        instances[i].visibilityMask = 255;
    }

    instances[0].traversableHandle = state.marching_cubes_gas_handle;

    size_t      instances_size_in_bytes = sizeof( OptixInstance ) * instances.size();
    CUDA_CHECK( cudaMalloc( ( void** )&state.d_instances, instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy( ( void* )state.d_instances, instances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice ) );

    state.ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    state.ias_instance_input.instanceArray.instances = state.d_instances;
    state.ias_instance_input.instanceArray.numInstances = static_cast<int>( instances.size() );

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &ias_accel_options, &state.ias_instance_input, 1, &ias_buffer_sizes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    compactedSizeOffset = roundUp<size_t>( ias_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_ias_and_compacted_size ), compactedSizeOffset + 8 ) );

    CUdeviceptr d_ias_temp_buffer;
    bool        needIASTempBuffer = ias_buffer_sizes.tempSizeInBytes > state.temp_buffer_size;
    if( needIASTempBuffer )
    {
        CUDA_CHECK( cudaMalloc( (void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes ) );
    }
    else
    {
        d_ias_temp_buffer = state.d_temp_buffer;
    }

    emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( state.context, 0, &ias_accel_options, &state.ias_instance_input, 1, d_ias_temp_buffer,
                                  ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias_and_compacted_size,
                                  ias_buffer_sizes.outputSizeInBytes, &state.ias_handle, &emitProperty, 1 ) );

    if( needIASTempBuffer )
    {
        CUDA_CHECK( cudaFree( (void*)d_ias_temp_buffer ) );
    }

    // Compress the IAS

    size_t compacted_ias_size;
    CUDA_CHECK( cudaMemcpy( &compacted_ias_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

    if( compacted_ias_size < ias_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_ias_output_buffer ), compacted_ias_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.ias_handle, state.d_ias_output_buffer,
                                        compacted_ias_size, &state.ias_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_ias_and_compacted_size ) );

        state.ias_output_buffer_size = compacted_ias_size;
    }
    else
    {
        state.d_ias_output_buffer = d_buffer_temp_output_ias_and_compacted_size;

        state.ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }

    // allocate enough temporary update space for updating the deforming GAS, exploding GAS and IAS.
    size_t maxUpdateTempSize = std::max( ias_buffer_sizes.tempUpdateSizeInBytes, gas_buffer_sizes.tempUpdateSizeInBytes );
    if( state.temp_buffer_size < maxUpdateTempSize )
    {
        CUDA_CHECK( cudaFree( (void*)state.d_temp_buffer ) );
        state.temp_buffer_size = maxUpdateTempSize;
        CUDA_CHECK( cudaMalloc( (void**)&state.d_temp_buffer, state.temp_buffer_size ) );
    }

    state.params.handle = state.ias_handle;
}


void createModule( DynamicGeometryState& state )
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipeline_compile_options.numPayloadValues = 5;
    state.pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixGcodeViewer.cu", inputSize );

    std::cout << "OPTIX_SAMPLE_NAME: " << OPTIX_SAMPLE_NAME << "\n";
    std::cout << "OPTIX_SAMPLE_DIR: " << OPTIX_SAMPLE_DIR << "\n";

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.ptx_module
    ) );
}


void createProgramGroups( DynamicGeometryState& state )
{
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &state.raygen_prog_group
        ) );
    }

    // Default miss_prog_group
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, &sizeof_log,
            &state.default_miss_prog_group
        ) );
    }

    // Original hit_prog_group
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &state.colour_normal_hit_prog_group
        ) );
    }

    // ambocc_radiance hit_prog_group
    {
        OptixProgramGroupDesc ambocc_radiance_hit_prog_group_desc = {};
        ambocc_radiance_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (culling_mode == CULLING_FRONTFACE)
        {
            ambocc_radiance_hit_prog_group_desc.hitgroup.moduleAH = state.ptx_module;
            ambocc_radiance_hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__cull_frontface";
        }
        else if (culling_mode == CULLING_BACKFACE)
        {
            ambocc_radiance_hit_prog_group_desc.hitgroup.moduleAH = state.ptx_module;
            ambocc_radiance_hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__cull_backface";
        }
        else
        {
            ambocc_radiance_hit_prog_group_desc.hitgroup.moduleAH = nullptr;
            ambocc_radiance_hit_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        }
        ambocc_radiance_hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        ambocc_radiance_hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_ambocc";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &ambocc_radiance_hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &state.ambocc_radiance_hit_prog_group
        ) );
    }

    // ambocc_occlusion hit_prog_group
    {
        OptixProgramGroupDesc ambocc_occlusion_hit_prog_group_desc = {};
        ambocc_occlusion_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        ambocc_occlusion_hit_prog_group_desc.hitgroup.moduleAH = state.ptx_module;
        ambocc_occlusion_hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion_ambocc";
        ambocc_occlusion_hit_prog_group_desc.hitgroup.moduleCH = nullptr;
        ambocc_occlusion_hit_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &ambocc_occlusion_hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &state.ambocc_occlusion_hit_prog_group
        ) );
    }
}


void createPipeline( DynamicGeometryState& state )
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.default_miss_prog_group,
        state.colour_normal_hit_prog_group,
        state.ambocc_radiance_hit_prog_group,
        state.ambocc_occlusion_hit_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof( program_groups ) / sizeof( program_groups[0] ),
        log,
        &sizeof_log,
        &state.pipeline
    ) );

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygen_prog_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.default_miss_prog_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.colour_normal_hit_prog_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.ambocc_radiance_hit_prog_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.ambocc_occlusion_hit_prog_group, &stack_sizes ) );

    uint32_t max_trace_depth = 1;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ) );

    // This is 2 since the largest depth is IAS->GAS
    const uint32_t max_traversable_graph_depth = 2;

    OPTIX_CHECK( optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth
    ) );
}


void createSBT_colourNormal( DynamicGeometryState& state )
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ) );

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_miss_records ), miss_record_size ) );

    MissRecord ms_sbt[1];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.default_miss_prog_group, &ms_sbt[0] ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_miss_records ),
        ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ) );

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_hitgroup_records ),
        hitgroup_record_size * g_instances.size()
    ) );

    std::vector<HitGroupRecord> hitgroup_records( g_instances.size() );
    for( int i = 0; i < static_cast<int>( g_instances.size() ); ++i )
    {
        const int sbt_idx = i;

        OPTIX_CHECK( optixSbtRecordPackHeader( state.colour_normal_hit_prog_group, &hitgroup_records[sbt_idx] ) );
    }

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_hitgroup_records ),
        hitgroup_records.data(),
        hitgroup_record_size*hitgroup_records.size(),
        cudaMemcpyHostToDevice
    ) );

    state.sbt_colour_normal.raygenRecord = d_raygen_record;
    state.sbt_colour_normal.missRecordBase = d_miss_records;
    state.sbt_colour_normal.missRecordStrideInBytes = static_cast< uint32_t >( miss_record_size );
    state.sbt_colour_normal.missRecordCount = 1;
    state.sbt_colour_normal.hitgroupRecordBase = d_hitgroup_records;
    state.sbt_colour_normal.hitgroupRecordStrideInBytes = static_cast< uint32_t >( hitgroup_record_size );
    state.sbt_colour_normal.hitgroupRecordCount = static_cast< uint32_t >( hitgroup_records.size() );
}


void createSBT_ambocc( DynamicGeometryState& state )
{
    // Create raygen records for the SBT
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast< void** >( &d_raygen_record ), raygen_record_size ) );

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ) );

    // Create miss records for the SBT
    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof( MissRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >(&d_miss_records ),
        miss_record_size * RAY_TYPE_COUNT ) );

    MissRecord ms_sbt[RAY_TYPE_COUNT];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.default_miss_prog_group, &ms_sbt[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.default_miss_prog_group, &ms_sbt[1] ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_miss_records ),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ) );

    // Create hitgroup records for the SBT
    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast< void** >( &d_hitgroup_records ),
        hitgroup_record_size * RAY_TYPE_COUNT
    ) );

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.ambocc_radiance_hit_prog_group, &hitgroup_records[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( state.ambocc_occlusion_hit_prog_group, &hitgroup_records[1] ) );

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast< void* >( d_hitgroup_records ),
        hitgroup_records,
        hitgroup_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ) );

    // Create the SBT for ambient occlusion renders
    state.sbt_ambocc.raygenRecord = d_raygen_record;
    state.sbt_ambocc.missRecordBase = d_miss_records;
    state.sbt_ambocc.missRecordStrideInBytes = static_cast< uint32_t >( miss_record_size );
    state.sbt_ambocc.missRecordCount = RAY_TYPE_COUNT;
    state.sbt_ambocc.hitgroupRecordBase = d_hitgroup_records;
    state.sbt_ambocc.hitgroupRecordStrideInBytes = static_cast< uint32_t >( hitgroup_record_size );
    state.sbt_ambocc.hitgroupRecordCount = RAY_TYPE_COUNT;
}


void cleanupState( DynamicGeometryState& state )
{
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.default_miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.colour_normal_hit_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptx_module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_colour_normal.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_colour_normal.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_colour_normal.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_ambocc.raygenRecord ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_ambocc.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.sbt_ambocc.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_vertices ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_marching_cubes_gas_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_instances ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_ias_output_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast< void* >( state.d_params ) ) );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    DynamicGeometryState state;
    state.params.width  = 1024;
    state.params.height = 768;
    state.params.ambocc_samples = 3;
    state.params.antialias_samples = 2;
    state.params.ambocc_minZ = 0.0f;
    state.params.ambocc_gamma = 1.4f;
    state.params.scene_epsilon = 1.e-4f;
    state.params.lens_type = LENS_TYPE_PINHOLE;
    state.params.view_distance = 0;
    state.params.view_offset = 0;
    state.params.quilt_columns = 0;
    state.params.quilt_rows = 0;
    state.params.anaglyph_mode = false;

    state.time = 0.0f;
    bool force_interactive_mode = false;

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::ZERO_COPY;


    std::cout << "arc = " << argc << "\n";
    // Parse command line options
    if( argc < 2 )
                printUsageAndExit( argv[0] );

    std::string in_file = argv[1];
    std::string out_file = "";

    unsigned int cuda_device_idx = 0;

    for( int i = 2; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--outfile" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            out_file = argv[++i];
        }
        else if( arg == "--interactive" )
        {
            force_interactive_mode = true;
        }
        else if( arg == "--antialias" || arg == "-aa" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            state.params.antialias_samples = atoi( argv[++i] );
        }
        else if( arg == "--ambocc" || arg == "-ao" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            state.params.ambocc_samples = atoi( argv[++i] );
        }
        else if( arg == "--gamma" || arg == "-g" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            state.params.ambocc_gamma = atof( argv[++i] );
        }
        else if( arg == "--geonormal" || arg == "-gn" )
        {
            useGeoNormal = true;
        }
        else if( arg == "--vertexnormal" || arg == "-vn" )
        {
            useGeoNormal = false;
        }
        else if( arg == "--device" || arg == "-v" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            cuda_device_idx = atoi( argv[++i] );
        }
        else if( arg == "--device_auto" || arg == "-va" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            char* processSlotNumber = getenv("PROCESS_SLOT_NUMBER");
            if (processSlotNumber)
            {
                cuda_device_idx = atoi( processSlotNumber );
            }
            else
            {
                std::cout << "Warning: environment variable PROCESS_SLOT_NUMBER not set\n";
            }
        }
        else if( arg == "--anaglyph" )
        {
            state.params.anaglyph_mode = true;
        }
        else if( arg == "--looking_glass_quilt" || arg == "-lg" )
        {
            if( i >= argc - 2 )
                printUsageAndExit( argv[0] );

            state.params.lens_type = LENS_TYPE_LOOKING_GLASS_QUILT;
            state.params.quilt_columns = (int)atoi(argv[++i]);
            state.params.quilt_rows = (int)atoi(argv[++i]);
        }
        else if (arg == "--view_distance" || arg == "-vd" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            state.params.view_distance = (float)atof(argv[++i]);
            state.params.view_offset = state.params.view_distance * tanf(state.params.view_offset_angle);
        }
        else if ( arg == "--view_offset_angle" || arg == "-va" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            state.params.view_offset_angle = degToRad((float)atof(argv[++i]));
            state.params.view_offset = state.params.view_distance * tanf(state.params.view_offset_angle);
        }
        else if ( arg == "--cam_pos" )
        {
            if (argc - i <= 3)
                printUsageAndExit(argv[0]);

            float cx = (float)atof(argv[++i]);
            float cy = (float)atof(argv[++i]);
            float cz = (float)atof(argv[++i]);

            camera_pos = make_float3( cx, cy, cz );
        }
        else if ( arg == "--cam_interest" )
        {
            if (argc - i <= 3)
                printUsageAndExit(argv[0]);

            float cx = (float)atof(argv[++i]);
            float cy = (float)atof(argv[++i]);
            float cz = (float)atof(argv[++i]);

            camera_interest = make_float3( cx, cy, cz );
        }
        else if ( arg == "--vfov" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            camera_vfov = (float)atof(argv[++i]);
        }
        else if ( arg == "--aspect" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            camera_aspect = (float)atof(argv[++i]);
        }
        else if ( arg == "--convergence_radius" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            state.params.convergence_radius = (float)atof(argv[++i]);
        }
        else if ( arg == "--nozzle_width" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            gcodeNozzleWidth = (float)atof(argv[++i]);
        }
        else if ( arg == "--layer_height" )
        {
            if (argc - i <= 1)
                printUsageAndExit(argv[0]);

            gcodeLayerHeight = (float)atof(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    if (in_file.empty()) {
        std::cout << "You must specify an input file using --infile\n";
        exit(1);
    }

    try
    {
        CUDA_CHECK( cudaSetDevice(cuda_device_idx) );

        initCameraState();

        //
        // Set up OptiX state
        //
        createContext( state );

        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSBT_colourNormal( state );
        createSBT_ambocc( state );
        initLaunchParams( state );

        // Load geometry data
        readMeshFile( state, in_file );

        buildMeshAccel( state );

        if( out_file.empty() || force_interactive_mode )
        {
            // Interactive display mode

            GLFWwindow* window = sutil::initUI( "optixGcodeViewer", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                std::chrono::duration<double> cur_gas_update_time( 0.0 );
                std::chrono::duration<double> gas_update_time( 0.0 );


                auto tstart = std::chrono::system_clock::now();

                do
                {
                    CUDA_CHECK( cudaSetDevice(cuda_device_idx) );

                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    auto tnow = std::chrono::system_clock::now();
                    std::chrono::duration<double> time = tnow - tstart;
                    state.time = (float)time.count();

                    // auto gas_start = std::chrono::steady_clock::now();
                    // updateMeshAccel( state );
                    // cudaDeviceSynchronize();
                    // cur_gas_update_time = std::chrono::steady_clock::now() - gas_start;
                    // gas_update_time += cur_gas_update_time;

                    if (camera_changed) {
                        updateState( output_buffer, state.params );
                        auto t1 = std::chrono::steady_clock::now();
                        state_update_time += t1 - t0;
                        t0 = t1;

                        launchSubframe( output_buffer, state );
                        t1 = std::chrono::steady_clock::now();
                        render_time += t1 - t0;
                        t0 = t1;
                    }

                    if (saveImageKeyPressed) {
                        if (out_file.empty())
                            out_file = "outfile.png";

                        std::cout << "Saving image " << out_file << "\n";

                        // Re-render the frame with higher render settings
                        int old_ambocc_samples = state.params.ambocc_samples;
                        int old_antialias_samples = state.params.antialias_samples;
                        state.params.ambocc_samples = 6;
                        state.params.antialias_samples = 10;
                        launchSubframe( output_buffer, state );

                        // Write out .png file
                        sutil::ImageBuffer buffer;
                        buffer.data = output_buffer.getHostPointer();
                        buffer.width = output_buffer.width();
                        buffer.height = output_buffer.height();
                        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
                        sutil::saveImage( out_file.c_str(), buffer, false );

                        // Restore old sample settings
                        state.params.ambocc_samples = old_ambocc_samples;
                        state.params.antialias_samples = old_antialias_samples;

                        saveImageKeyPressed = false;
                    }

                    displaySubframe( output_buffer, gl_display, window );
                    auto t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    // Display timings
                    static int32_t last_update_frames = 0;
                    constexpr std::chrono::duration<double> display_update_interval( 0.5 );

                    last_update_frames++;

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            handleCameraUpdate( state.params );
            handleResize( output_buffer, state.params );

            // Read geometry here?

            CUDA_CHECK( cudaSetDevice(cuda_device_idx) );
            //updateMeshAccel( state );
            launchSubframe( output_buffer, state );

            // sutil::ImageBuffer buffer;
            // buffer.data = output_buffer.getHostPointer();
            // buffer.width = output_buffer.width();
            // buffer.height = output_buffer.height();
            // buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage( out_file.c_str(), buffer, false );
        }

        CUDA_CHECK( cudaSetDevice(cuda_device_idx) );
        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

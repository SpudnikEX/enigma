// Default Development shader for Models

// Outputs vertex position
// Outputs UV corrected from -1,1 to 0,1 (calculated in screen space UV)
// Outputs custom vertex color

#version 460

#define v_i gl_VertexIndex

// Passed Vertex Values (Vertex_3D)
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color; // Dont nessecarily need this here, could be used for weight blending at some point. Colors could be input later straight into the fragment shader from a vector, to change on the fly
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

// Passed values from vertex shader to fragment shader
layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_uv;

// ❗ IMPORTANT! One uniform field MUST BE USED, or program will crash !!
layout(set = 0, binding = 0) uniform MvpData {
    mat4 model_matrix;
    mat4 view_matrix; //vec3 position;
    mat4 projection_matrix;
    float time; // Cumulative time (any other ways to do this?)
} uniforms; // struct name that holds the above fields

// Passed Texture as ImageView and Sampler
layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    float x = uniforms.time;
    //https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules
    // Vertex positions specified as Normalized Device Coordinates.
    // Output normalized device coordinates by outputting them as clip-space coordinates & setting last value to 1
    // gl_position: the clip-space output position of the current vertex

    // mat[col][row] or mat[col]
    // GLSL array starts at 0
    //gl_Position = uniforms.view_matrix[3].xyzw; 
    //gl_Position = vec4(position, 1.0);

    // Use MVP matrix to convert model to camera space
    // Multiplication order is clip=(P⋅V⋅M⋅v)
    // vec4 debug;
    // debug.x = uniforms.model_matrix[0][3] + position.x;
    // debug.y = uniforms.model_matrix[1][3] + position.y;
    // debug.z = uniforms.model_matrix[2][3] + position.z;
    // debug.w = 1.0;

    // Heightmap //
    vec3 adjusted_position = position;
    adjusted_position.y += texture(tex,uv).r * 5.0;
    gl_Position = uniforms.projection_matrix * uniforms.view_matrix * uniforms.model_matrix * vec4(position,1.0); // position
    
    //uniforms.projection_matrix * * 
    
    //gl_Position = vec4(position, 1.0);
    
    // Normalize from 0 - 1
    // Transform vertex to NDC ( / w), then transform to UV space (0-1)
    //out_uv = (gl_Position.xy / gl_Position.w) * 0.5 + vec2(0.5, 0.5); // The UV coordinates of the screen\

    // By default, vulkan uses -1,1 coordinates, so that 0,0 is the center of the window
    // -1,-1 ------- +1,-1
    //    |     |      |
    //    |    0,0     |
    //    |     |      |
    // -1,+1 ------- +1,+1

    out_uv = uv; // per-vertex uv
    out_color = color; // per-vertex color
}
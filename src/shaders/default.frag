// Default Development Fragment shader for Models

// Colors using either passed vertex color, or screen space UV

#version 460

//Custom passed values to fragment shader from vertex shader
layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 frag_color; //variable that is linked to the first (and only) framebuffer at index 0. (can be any name)

layout(set = 0, binding = 0) uniform MvpData {
    mat4 model_matrix;
    mat4 view_matrix;
    mat4 projection_matrix;
    float time;
} uniforms;

// Texture
layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    float x = uniforms.time;

    // pass the current position of the pixel as (rgb) in uv coordinates (0 to 1)
    //frag_color = vec4(in_uv, 0.0, 1.0);

    // Previous hard-coded color
    //frag_color = in_color; // hard-coded vertex color

    // Vulkan SPIRV supplied function (See vulkano image/main.rs)
        // Uses texture's alpha value for transparency if enabled in pipline (blend)
    vec4 tex_color = texture(tex, in_uv);
    frag_color = tex_color; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Extras
///////////////////////////////////////////////////////////////////////////////////////////////////

// float fresnel(float amount, vec3 normal, vec3 view)
// {
// 	return pow((1.0 - clamp(dot(normalize(normal), normalize(view)), 0.0, 1.0 )), amount);
// }

// void fragment()
// {
// 	vec3 base_color = vec3(0.0);
// 	float basic_fresnel = fresnel(3.0, NORMAL, VIEW);
// 	ALBEDO = base_color + basic_fresnel;
// }

#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(binding = 2) uniform sampler2D tex;
layout(binding = 3) uniform sampler2D edge_tex;

void main()
{
    if (uv.x < 0.5) {
        fragColor = texture(tex, uv);
    } else {
        float edge = texture(edge_tex, uv).r * 2.0;
        fragColor = vec4(vec3(edge), 1.0);
    }
}

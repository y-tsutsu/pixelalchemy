#version 330

in vec2 uv;
out vec4 fragColor;
uniform sampler2D tex;

void main() {
    vec2 fixed_uv = vec2(uv.x, 1.0 - uv.y);  // 180度反転を補正

    if (fixed_uv.x < 0.5) {  // 左：元画像
        vec4 c = texture(tex, fixed_uv);
        c.rgb = c.rgb / max(c.a, 0.0001);
        fragColor = vec4(c.rgb, 1.0);
    } else {
        vec2 texel = 1.0 / textureSize(tex, 0);

        float dx = 0.0;
        float dy = 0.0;

        dx += texture(tex, fixed_uv + vec2(-texel.x, 0)).r * -1.0;
        dx += texture(tex, fixed_uv + vec2( texel.x, 0)).r *  1.0;

        dy += texture(tex, fixed_uv + vec2(0, -texel.y)).r * -1.0;
        dy += texture(tex, fixed_uv + vec2(0,  texel.y)).r *  1.0;

        vec4 base = texture(tex, fixed_uv);
        base.rgb /= max(base.a, 0.0001);  // 色味補正

        float edge = length(vec2(dx, dy));
        edge *= 5.0;

        fragColor = vec4(vec3(edge), 1.0);

        // 赤で協調したエッジを合成した結果
        // vec3 edge_col = vec3(edge, 0.0, 0.0);
        // vec3 result = base.rgb + edge_col;
        // fragColor = vec4(result, 1.0);
    }
}

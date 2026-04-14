#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex;
uniform sampler2D edge_tex;

void main()
{
    vec2 fixed_uv = vec2(uv.x, 1.0 - uv.y); // 180度反転を補正

    if (fixed_uv.x < 0.5) // 左：元画像
    {
        vec4 c = texture(tex, fixed_uv);
        c.rgb = c.rgb / max(c.a, 0.0001); // 色味補正
        fragColor = vec4(c.rgb, 1.0);
    }
    else
    {
        float edge = texture(edge_tex, fixed_uv).r;
        edge *= 2.0;

        fragColor = vec4(vec3(edge), 1.0);

        // // 赤で協調したエッジを合成した結果
        // vec4 base = texture(tex, fixed_uv);
        // base.rgb /= max(base.a, 0.0001); // 色味補正
        // vec3 edge_col = vec3(edge, 0.0, 0.0);
        // vec3 result = base.rgb + edge_col;
        // fragColor = vec4(result, 1.0);
    }
}

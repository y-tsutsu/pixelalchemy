import moderngl
import moderngl_window as mglw
import numpy as np


class GLRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = 'GStreamer + OpenGL Edge detection'
    window_size = (800, 600)
    pipeline = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Quad
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            +1.0, -1.0, 1.0, 0.0,
            -1.0, +1.0, 0.0, 1.0,
            +1.0, +1.0, 1.0, 1.0,
        ], dtype='f4')

        # Vertex Buffer Object
        self.vbo = self.ctx.buffer(vertices.tobytes())

        # 仮テクスチャ（あとでサイズ更新）
        self.texture = None

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 uv;

                void main() {
                    uv = in_uv;
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
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
            '''
        )

        # Vertex Array Object
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

    def on_render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.0)

        frame = self.pipeline.get_frame()
        if frame is None:
            return

        h, w, _ = frame.shape
        # 初回 or サイズ変更時にテクスチャ作成
        if self.texture is None or self.texture.size != (w, h):
            self.texture = self.ctx.texture((w, h), 4)
            self.texture.build_mipmaps()

        self.texture.write(frame.tobytes())
        self.texture.use(location=0)
        self.prog['tex'] = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)

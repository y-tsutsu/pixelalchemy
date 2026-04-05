import moderngl
import moderngl_window as mglw
import numpy as np


class ComputeApp(mglw.WindowConfig):
    gl_version = (4, 3)  # ← Compute Shaderは4.3以上必要
    title = 'Compute Shader Step1'
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        w, h = self.window_size

        # 出力用テクスチャ
        self.texture = self.ctx.texture((w, h), 4)
        self.texture.bind_to_image(0, read=False, write=True)

        # Compute Shader
        self.compute = self.ctx.compute_shader('''
            #version 430
            layout(local_size_x = 16, local_size_y = 16) in;
            layout(rgba8, binding = 0) uniform image2D img;

            void main() {
                ivec2 id = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(img);

                vec2 uv = vec2(id) / vec2(size);
                vec3 color = vec3(uv.x, 0.0, uv.y);

                imageStore(img, id, vec4(color, 1.0));
            }
        ''')

        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # 表示用シェーダ（そのまま表示）
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
                    fragColor = texture(tex, uv);
                }
            '''
        )

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

        self.texture.use()
        self.prog['tex'] = 0

    def on_render(self, time, frame_time):
        w, h = self.window_size

        self.compute.run(
            group_x=(w + 15) // 16,
            group_y=(h + 15) // 16
        )

        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    mglw.run_window_config(ComputeApp)


if __name__ == '__main__':
    main()

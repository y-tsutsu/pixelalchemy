import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image


class ComputeApp(mglw.WindowConfig):
    gl_version = (4, 3)  # ← Compute Shaderは4.3以上必要
    title = 'Compute Shader Step1'
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        w, h = self.window_size

        # 画像読み込み
        img = Image.open('image/sample.png').transpose(Image.FLIP_TOP_BOTTOM)
        img = img.resize(self.window_size)
        img_data = img.tobytes()

        # 入力テクスチャ
        self.src_texture = self.ctx.texture(self.window_size, 4, img_data)
        self.src_texture.bind_to_image(0, read=True, write=False)

        # 出力用テクスチャ
        self.dst_texture = self.ctx.texture((w, h), 4)
        self.dst_texture.bind_to_image(1, read=False, write=True)

        # Compute Shader
        self.compute = self.ctx.compute_shader('''
            #version 430

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(rgba8, binding = 0) readonly uniform image2D src_img;
            layout(rgba8, binding = 1) writeonly uniform image2D dst_img;

            void main() {
                ivec2 id = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(src_img);

                if (id.x >= size.x || id.y >= size.y) return;

                vec4 color = imageLoad(src_img, id);
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                vec4 out_color = vec4(gray, gray, gray, 1.0);

                // vec4 out_color = vec4(1.0 - color.rgb, 1.0);

                // ivec2 flipped = ivec2(size.x - id.x - 1, id.y);
                // vec4 out_color = imageLoad(src_img, flipped);

                // vec2 uv = vec2(id) / vec2(size);
                // color.rgb *= uv.x;
                // vec4 out_color = color;

                imageStore(dst_img, id, out_color);
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

        self.dst_texture.use()
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

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image


class App(mglw.WindowConfig):
    gl_version = (3, 3)
    title = 'Texture Practice'
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # pos + uv
        vertices = np.array([
            # x,    y,     u, v
            -1.0, -1.0,  0.0, 0.0,
            1.0,  -1.0,  1.0, 0.0,
            -1.0, 1.0,   0.0, 1.0,
            1.0,  1.0,   1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        img = Image.open('image/sample.png').transpose(Image.FLIP_TOP_BOTTOM)
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.texture.build_mipmaps()

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
                    vec2 texel = 1.0 / textureSize(tex, 0);

                    float kernel[25] = float[](
                        1,  4,  6,  4, 1,
                        4, 16, 24, 16, 4,
                        6, 24, 36, 24, 6,
                        4, 16, 24, 16, 4,
                        1,  4,  6,  4, 1
                    );

                    vec3 col = vec3(0.0);
                    float sum = 0.0;
                    int idx = 0;
                    for (int y = -2; y <= 2; y++) {
                        for (int x = -2; x <= 2; x++) {
                            vec2 offset = vec2(x, y) * texel;
                            float w = kernel[idx++];
                            col += texture(tex, uv + offset).rgb * w;
                            sum += w;
                        }
                    }
                    col /= sum;
                    fragColor = vec4(col, 1.0);
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
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    mglw.run_window_config(App)


if __name__ == '__main__':
    main()

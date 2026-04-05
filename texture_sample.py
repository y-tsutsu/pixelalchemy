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
                // uniform float time;

                void main() {
                    fragColor = texture(tex, uv);
                    // fragColor = texture(tex, uv * 2.0);
                    // fragColor = texture(tex, vec2(uv.x, 1.0 - uv.y));
                    // fragColor = texture(tex, uv + vec2(time * 0.1, 0.0));
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
        # self.ctx.viewport = (0, 0, self.wnd.buffer_size[0], self.wnd.buffer_size[1])
        # self.prog['time'].value = time
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    mglw.run_window_config(App)


if __name__ == '__main__':
    main()

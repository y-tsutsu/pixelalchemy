import moderngl
import moderngl_window as mglw
import numpy as np


class App(mglw.WindowConfig):
    gl_version = (3, 3)
    title = 'Shader Practice'
    window_size = (800, 600)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # フルスクリーンQuad（画面全体を覆う板）
        # v2 ---- v3
        #  | \    |
        #  |  \   |
        #  |   \  |
        # v0 ---- v1
        vertices = np.array([
            -1.0, -1.0,  # 左下
            1.0, -1.0,   # 右下
            -1.0, 1.0,   # 左上
            1.0, 1.0,    # 右上
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;
                out vec2 uv;

                void main() {
                    uv = in_pos * 0.5 + 0.5;
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 uv;
                out vec4 fragColor;
                uniform float time;

                void main() {
                    // fragColor = vec4(1.0, 0.0, 0.0, 1.0);
                    // fragColor = vec4(uv.x, uv.y, 0.0, 1.0);
                    fragColor = vec4(uv, abs(sin(time)), 1.0);
                }
            '''
        )

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f', 'in_pos')]
        )

    def on_render(self, time, frame_time):
        self.prog['time'].value = time
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)


def main():
    mglw.run_window_config(App)


if __name__ == '__main__':
    main()

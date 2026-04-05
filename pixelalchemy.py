import moderngl_window as mglw

from gl_renderer import GLRenderer
from gst_pipeline import GStreamerPipeline


def main():
    pipeline = GStreamerPipeline('video/sample.mp4')
    pipeline.start()

    GLRenderer.pipeline = pipeline
    mglw.run_window_config(GLRenderer)


if __name__ == '__main__':
    main()

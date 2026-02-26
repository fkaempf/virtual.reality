"""Main Dear ImGui application window.

Provides a unified GUI that integrates stimulus control,
calibration, and mapping into a single application with a menu bar
and dockable panels.

Requires ``imgui[pygame]>=2.0``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VirtualRealityApp:
    """Main GUI application.

    Manages the pygame/OpenGL window, Dear ImGui context, and
    panel layout.
    """

    def __init__(self) -> None:
        self._running = False

    def run(self) -> None:
        """Launch the GUI main loop."""
        try:
            import imgui
            from imgui.integrations.pygame import PygameRenderer
        except ImportError:
            print(
                "Dear ImGui not installed. Install with:\n"
                "  pip install 'imgui[pygame]>=2.0'"
            )
            return

        import pygame
        from OpenGL import GL

        pygame.init()
        size = (1280, 720)
        pygame.display.set_mode(
            size,
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE,
        )
        pygame.display.set_caption("Virtual Reality")

        imgui.create_context()
        renderer = PygameRenderer()
        self._running = True
        clock = pygame.time.Clock()

        logger.info("GUI started")

        try:
            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    renderer.process_event(event)

                renderer.process_inputs()
                imgui.new_frame()

                self._draw_menu_bar()
                self._draw_panels()

                GL.glClearColor(0.1, 0.1, 0.1, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)

                imgui.render()
                renderer.render(imgui.get_draw_data())
                pygame.display.flip()
                clock.tick(60)
        finally:
            renderer.shutdown()
            pygame.quit()
            logger.info("GUI closed")

    def _draw_menu_bar(self) -> None:
        """Draw the main menu bar."""
        import imgui

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                clicked, _ = imgui.menu_item("Quit", "Ctrl+Q")
                if clicked:
                    self._running = False
                imgui.end_menu()
            if imgui.begin_menu("Panels"):
                imgui.menu_item("Stimulus", None, True)
                imgui.menu_item("Calibration", None, True)
                imgui.menu_item("Mapping", None, True)
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def _draw_panels(self) -> None:
        """Draw all GUI panels."""
        import imgui

        imgui.begin("Stimulus Control")
        imgui.text("Stimulus controls will appear here.")
        imgui.text("Use CLI entry points for now:")
        imgui.bullet_text("vr-fly3d")
        imgui.bullet_text("vr-fly2d")
        imgui.bullet_text("vr-warp-test")
        imgui.end()

        imgui.begin("Calibration")
        imgui.text("Calibration pipeline controls.")
        imgui.text("Requires camera + projector hardware.")
        imgui.end()

        imgui.begin("Mapping")
        imgui.text("Projector-camera mapping controls.")
        imgui.end()


def main() -> None:
    """CLI entry point for the GUI."""
    app = VirtualRealityApp()
    app.run()

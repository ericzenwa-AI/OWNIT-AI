"""Draw the raster icons from the same geometry as web/icon.svg.

Modern browsers take the SVG, but three things still want a bitmap: the bare
/favicon.ico a browser asks for whether or not the page says to, Windows and
Android shortcut tiles, and the iOS home screen, which has never read SVG.

The shapes are repeated here rather than rasterised from the SVG because
rasterising would mean a renderer on the build box - cairo, or a headless
browser - to produce four small files. The repetition is the cheaper debt, so
long as it is one function and it says so. If the SVG changes, change SIZE and
BARS to match and run this again:

    python backend/icons.py

Small sizes are drawn large and shrunk, because a 9-unit bar at 16px lands on
2.25 pixels and drawing it straight gives a hard-edged mess.
"""
import pathlib

from PIL import Image, ImageDraw

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"

# The viewBox of web/icon.svg. Every number below is in these units.
BOX = 64
TILE_RADIUS = 14
TILE = (18, 20, 28)           # #12141C, dark enough for a light tab strip,
                              # lifted just enough to still read on a dark one
MUTED = (242, 242, 244, 87)   # #F2F2F4 at 0.34, the two bars above
LIT = (61, 155, 255)          # #3D9BFF, the one thing in the way

# x, y, width, height - the question above, the gap underneath it. Two bars
# and not three: at 16px a third one closes the gaps and the whole mark turns
# into a single grey smudge with a blue line under it.
BARS = [(10, 14, 44, 13), (18, 37, 28, 13)]
BAR_RADIUS = 6.5

# 16 and 32 are the tab strip and the bookmark bar; 48 is the Windows
# shortcut; 180 is what iOS asks for by name.
ICO_SIZES = (16, 32, 48)
APPLE_SIZE = 180
SUPERSAMPLE = 8


def draw(size: int) -> Image.Image:
    """The mark at one size, drawn big and shrunk so the edges are clean."""
    big = size * SUPERSAMPLE
    scale = big / BOX

    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    pen = ImageDraw.Draw(tile)
    pen.rounded_rectangle(
        (0, 0, big - 1, big - 1), radius=TILE_RADIUS * scale, fill=(*TILE, 255))

    # The muted bars carry alpha, so they are drawn on their own layer and
    # composited - drawing them straight on would punch their alpha into the
    # tile instead of blending with it.
    over = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    brush = ImageDraw.Draw(over)
    for (x, y, width, height), fill in zip(BARS, (MUTED, (*LIT, 255))):
        brush.rounded_rectangle(
            (x * scale, y * scale, (x + width) * scale, (y + height) * scale),
            radius=BAR_RADIUS * scale, fill=fill)

    return Image.alpha_composite(tile, over).resize(
        (size, size), Image.LANCZOS)


def main() -> None:
    # One .ico holding all three sizes, so the browser picks rather than
    # scaling whichever one it was given. Largest first: Pillow drops any
    # requested size bigger than the image it is saving, so handing it the
    # 16px frame first silently writes a one-frame .ico.
    frames = [draw(size) for size in sorted(ICO_SIZES, reverse=True)]
    frames[0].save(WEB / "favicon.ico", sizes=[(s, s) for s in ICO_SIZES],
                   append_images=frames[1:])

    # iOS composites the home screen icon onto its own background and does
    # not honour alpha, so this one is flattened on the tile colour.
    apple = Image.new("RGB", (APPLE_SIZE, APPLE_SIZE), TILE)
    apple.paste(draw(APPLE_SIZE), (0, 0), draw(APPLE_SIZE))
    apple.save(WEB / "apple-touch-icon.png")

    for name in ("favicon.ico", "apple-touch-icon.png"):
        path = WEB / name
        print(f"  {name:22} {path.stat().st_size:>6} bytes")


if __name__ == "__main__":
    main()

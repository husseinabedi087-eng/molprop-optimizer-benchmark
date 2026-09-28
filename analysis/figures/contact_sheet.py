"""All figure PNGs on one review page."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W, COLS, PAD, LABEL_H = 1400, 3, 40, 60


def build(pngs: list[Path], out: Path, title: str) -> Path:
    tiles = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        # Same scale for every figure: a 7.2 in (double-column) figure fills TILE_W, a 3.5 in one about half,
        # so text sizes compare truthfully across tiles.
        scale = TILE_W / (7.2 * 600)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        tiles.append((p.stem, im))
    rows = [tiles[i:i + COLS] for i in range(0, len(tiles), COLS)]
    row_h = [max(im.height for _, im in r) + LABEL_H for r in rows]
    W = COLS * TILE_W + (COLS + 1) * PAD
    H = 120 + sum(row_h) + (len(rows) + 1) * PAD
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font, big = ImageFont.truetype("arial.ttf", 34), ImageFont.truetype("arialbd.ttf", 48)
    except OSError:
        font = big = ImageFont.load_default()
    draw.text((PAD, 40), title, fill="black", font=big)
    y = 120 + PAD
    for r, h in zip(rows, row_h):
        x = PAD
        for name, im in r:
            draw.text((x, y), name, fill="#333333", font=font)
            sheet.paste(im, (x, y + LABEL_H))
            draw.rectangle([x - 1, y + LABEL_H - 1, x + im.width, y + LABEL_H + im.height], outline="#DDDDDD")
            x += TILE_W + PAD
        y += h + PAD
    sheet.save(out, dpi=(150, 150))
    return out

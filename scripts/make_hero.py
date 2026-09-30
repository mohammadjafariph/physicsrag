"""Compose the PhysicsRAG README hero banner (one-off asset generator).

Takes docs/hero_raw.png (Pollinations flux art, no text) and renders the
title + tagline with Pillow so the banner typography is crisp and exact.
Re-run only when regenerating the artwork: python scripts/make_hero.py
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 1. Base art: center-crop to the 1280x400 banner
im = Image.open("docs/hero_raw.png").convert("RGB")
tw, th = 1280, 400
ratio = max(tw / im.width, th / im.height)
im = im.resize((round(im.width * ratio), round(im.height * ratio)))
left = (im.width - tw) // 2
top = (im.height - th) // 2
im = im.crop((left, top, left + tw, top + th))

# Remove the pollinations watermark (bottom-right): clone the sky patch
# just above it over the corner, then soften the seam.
wm_box = (tw - 170, th - 48, tw, th)
sky = im.crop((tw - 170, th - 96, tw, th - 48)).filter(
    ImageFilter.GaussianBlur(6)
)
im.paste(sky, (wm_box[0], wm_box[1]))

# 2. Left-to-right dark gradient overlay so the text pops
overlay = Image.new("L", (tw, 1), 0)
for x in range(tw):
    overlay.putpixel((x, 0), int(210 * max(0.0, 1.0 - x / (tw * 0.75))))
overlay = overlay.resize((tw, th))
im = Image.composite(Image.new("RGB", (tw, th), (8, 11, 19)), im, overlay)

# Feathered dark vignette over the bottom-right corner: hides the seam
# left by the watermark-removal clone and balances the composition.
vignette = Image.new("L", (tw, th), 0)
ImageDraw.Draw(vignette).rectangle([tw - 230, th - 100, tw, th], fill=110)
vignette = vignette.filter(ImageFilter.GaussianBlur(50))
im = Image.composite(Image.new("RGB", (tw, th), (8, 11, 19)), im, vignette)

# 3. Typography (matches the UI: Segoe UI Semibold, teal accent)
draw = ImageDraw.Draw(im)
title_font = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 88)
tag_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 27)
psi_font = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 84)

x0, y0 = 72, 118
box = (x0 - 14, y0 + 2, x0 + 92, y0 + 104)
draw.rounded_rectangle(box, radius=18, fill=(24, 42, 58), outline=(45, 212, 191), width=2)
draw.text((x0 + 22, y0 + 8), "\u03A8", font=psi_font, fill=(94, 234, 212))

tx = x0 + 130
draw.text((tx, y0), "PhysicsRAG", font=title_font, fill=(240, 244, 250))
draw.text(
    (tx + 4, y0 + 112),
    "Self-hosted RAG for physics papers  \u00b7  autonomous research  \u00b7  any LLM provider",
    font=tag_font,
    fill=(130, 141, 163),
)

im.save("docs/hero.png", optimize=True)
print("saved docs/hero.png", im.size)
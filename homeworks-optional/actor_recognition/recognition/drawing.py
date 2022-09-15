from PIL import ImageDraw
from PIL import Image as PILImage
import typing as t
from .types import Track


def draw_tracks(image: PILImage, tracks: t.Sequence[Track]) -> PILImage:
    draw = ImageDraw.Draw(image)
    for cur_track in tracks:
        x0, y0, x1, y1 = cur_track.tracking_box
        name = cur_track.person.name if cur_track.person is not None else 'Unknown'
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
        draw.text((x0, y1 + 2), '{}'.format(cur_track.track_id), fill=(255, 0, 0))
        draw.text((x0, y1 - 5), '{}'.format(name), fill=(230, 230, 230))
    return image

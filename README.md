# Photo booth

Green screen photo booth for Windows: live keyed preview, countdown, full-res
photo, background library, optional rate-limited printing, and an M5StickC
Plus (or Plus2) as a two-button Bluetooth remote.

- **Button A** (front): take a photo
- **Button B** (side): next background; hold for previous

## Setup (booth laptop)

1. Install Python 3.11 or newer from python.org ("Add to PATH" ticked).
2. Run `setup.bat` (creates `.venv` and installs everything).
3. Run `run.bat --windowed` and press **F2** to open the settings panel
   (or browse to http://localhost:8765/settings). Everything below is done
   there; it applies live, and **Save** writes it to `settings.json` (kept out
   of git: it holds the panel PIN and printer name).
4. **Camera** tab: pick the capture card by name, its native resolution and frame
   rate, and press *Use this camera*. Pick by name, not number: OBS, vMix,
   DroidCam etc. install virtual cameras that show a placeholder picture. Lock
   exposure, white balance and focus on the camera itself.
5. **Key** tab: tap the green screen in the preview to set its colour, then use
   the Matte view to set the clip levels (below).
6. **Backgrounds** tab: drop pictures in (or *Add pictures*, which on a phone
   opens the photo library). Drag or use the arrows to set the order, tap a name
   to rename, *Use now* to show one in the booth. Uploads are scaled down to
   3840 px and saved as `NN_Name.jpg` in `backgrounds/`; deleted ones go to
   `backgrounds/deleted/`.
7. Press **Save**, then run `run.bat` for the fullscreen booth.

## Dashboard

http://localhost:8765 (or the phone address, see below) shows the live result
with a **Take photo** button (same as the remote's big button; press again
during the countdown to cancel), the backgrounds to tap and switch to, the
printer's state as guests see it ("Ready to print", "Printer resting 0:42"
counting down, "Printer needs help" with the error), print and photo counts
against the limits, and a feed of recent activity (photo taken, sent to
printer, printed, skipped and why, failed, cancelled, deleted).

## Gallery

http://localhost:8765/gallery has every photo, newest first. Click a photo for
a big view (arrow keys or swipe, Download); **Fullscreen** and **Slideshow**
turn it into a display for a TV: controls hide after a few seconds and each new
photo is shown as soon as it's taken. The **×** on a photo (or Delete in the big
view) removes it from the gallery and the slideshows; it's moved, with its
original, into a `deleted` folder inside the photos folder, so it can be put back.

The **printer** button on a photo (or Print in the big view) prints it again.
Reprints are for the operator, so they always print: the guest limits and
*Print every photo* don't apply, and reprints don't use up the guests'
allowance (they're logged as `<time> reprint` in `print_log.txt` and counted in
the dashboard's "printed" total). The page says when it's been sent or if the
printer failed.

## Second screen

*Booth & print → Second screen* puts a clean full-screen slideshow (no text or
buttons) on another monitor or a TV. It plays every photo oldest first, so new
photos join the end. A photo jumps straight on the moment it's taken and stays
up until nobody has taken one for *Show each new photo for* seconds (60); if
more are taken in that time the newest is shown. Then the slideshow carries on
where it left off. Before the first photo it shows the backgrounds.

It runs as its own window process, so it can't slow down the booth, and it
never takes the keyboard from the booth window. If its monitor is unplugged it
closes, and comes back when it's plugged in again.

## Settings panel

Live preview (Result / Matte / Camera views) beside five tabs: **Key**,
**Colour**, **Camera**, **Backgrounds**, **Booth & print**. Revert undoes
everything since the last save.

To use it from a phone or tablet (tuning while standing where the guests
stand), set a PIN under *Booth & print → Phone & tablet access* and Save. The
booth then listens on the network (Windows asks once to allow Python through
the firewall: allow *Private networks*); the address is shown there and in the
booth's help screen (H). The phone asks for the PIN once. With no PIN, only the
booth laptop can open the panel.

### Key controls

| Control | What it does |
|---|---|
| Balance | Colour difference against `max(R,B)` (100%) or their average (0%). 100% keeps yellow, blond hair and cyan solid; lower only for a very clean screen. |
| Clip black / white | Matte levels. Raise black until the screen is solid black in the Matte view; lower white until people are solid white. |
| Matte blur | Blurs the difference matte before clipping. Helps with noisy cameras and 4:2:0 capture cards. |
| Choke | Shrinks (+) or grows (−) the matte, in pixels at 1080p. Removes dark/green outlines from soft or sharpened cameras. |
| Softness | Edge feather after the choke. |
| Despill / Despill balance | How much green is removed from people, and how hard (0% = limit green to the average of red and blue, 100% = to the larger of them). |
| Luma restore | Adds removed spill back as grey so despilled areas don't go dark/magenta. |
| Clean semi-transparent edges | Removes the screen colour from semi-transparent edges (hair, motion blur) so they take on the new background instead of a green fringe. |
| Light wrap / size | Lets the background's light bleed slightly onto people's edges. |
| Garbage matte | Forces the frame edges to background (stands, the room past the edge of the screen). |

All spatial settings are in pixels at 1080 lines and scale with resolution, so
the preview and the full-res photo key identically.

### Colour (fill layer)

Exposure, contrast, midtones (gamma), saturation, temperature and tint for the
**people only**; the background picture is untouched. The grade is applied to
the un-premultiplied subject colour, so hair and soft edges grade cleanly.
Light wrap is added after the grade.

### Clean plate (screen correction)

*Capture clean plate* on the Key tab (or `C` in the booth) waits 5 s for
everyone to leave the shot, averages 30 frames of the empty screen and saves
`clean_plate.png`. The key then compares
each pixel with the screen's own brightness at that spot, which fixes light
falloff, hot spots and uneven fabric (in testing it cut ghosting on a dim,
unevenly lit screen from 37% to 3.5%).

It makes things **worse** when people stand close enough to cast shadows on the
screen (the screen behind them is darker than in the plate). Keep people 1.5 m+
from the screen if you use it; otherwise leave it off and use clip black.
*Match plate brightness* compensates for cameras on auto exposure.

### What it was tested with

- Synthetic frames with known ground truth (`tests/`): yellow shirt, skin,
  dark and blond hair strands, cyan, semi-transparent fabric, spill, uneven
  light, noise, 4:2:0 chroma. Edge subtraction cuts edge colour error ~28%.
- A Logitech C110 with a creased, room-lit screen: clean with clip black 0.55,
  choke 8, softness 2 (`settings.example.json` has this rig's settings; copy it
  to `settings.json` to start from them, then re-tune for your camera and
  lights, which should need far gentler settings).

Speed (Ryzen 7 3700X, Numba): 720p 11 ms, 1080p 24 ms, 4K 92 ms per frame.

## Running the booth (`run.bat`)

Guests only need the remote (or Space / arrow keys). Operator keys:

| Key | |
|---|---|
| Space / Enter | take photo |
| Right / Left | next / previous background |
| P | printing on/off (for this session) |
| M | show the matte |
| C / X | capture / clear clean plate |
| S | camera driver settings dialog (DirectShow) |
| R | reload backgrounds and settings.json |
| F2 | open the settings panel in a browser |
| F11 | fullscreen / window |
| H | help and status |
| Esc | quit |

`run.bat --windowed` starts in a window.

During the countdown a bobbing yellow arrow points up at the camera with
"Look at the camera!" under it (*Booth & print → Countdown message*; empty
turns the arrow off). The printer badge steps aside while it's showing.

Photos go to `~/Pictures/Photobooth/<date>/IMG_*.jpg`, with the un-keyed camera
frame in `originals/` (for re-keying later) when `save_original` is on.
`booth.log` and `print_log.txt` are kept in the output folder.

An `overlay` PNG with transparency (event logo, frame) is drawn over the
preview and every photo.

## Printing

Off by default. In `settings.json`:

```json
"print": {
  "enabled": true,
  "printer": "",
  "copies": 1,
  "fit": "fit",
  "min_seconds_between": 60,
  "max_per_hour": 20,
  "max_per_day": 100
}
```

- `printer`: printer name; `""` = Windows default printer
- `fit`: `fit` shows the whole photo; `fill` crops to the page (borderless photo printers)
- `min_seconds_between`: cooldown between prints
- `max_per_hour`, `max_per_day`: caps; `0` = no limit

Photos are always saved; when a limit is hit the guest sees e.g. "Printing
paused - next print in 0:45" and that photo just isn't printed. Limits come from
`print_log.txt`, so restarting the app doesn't reset them.

## The remote (M5StickC Plus / Plus2)

Flash `firmware/photobooth_remote` (needs the ESP32 core 3.x and M5Unified).
Use `m5stack_stickc_plus` for a Plus (ESP32-PICO-D4, 4 MB) or
`m5stack_stickc_plus2` for a Plus2; the code detects the board at runtime. The
stick shows up as an FTDI "USB Serial Port"; upload at 115200, faster speeds
fail with "The chip stopped responding":

```bash
arduino-cli compile --fqbn esp32:esp32:m5stack_stickc_plus firmware/photobooth_remote
```

```bash
arduino-cli upload --fqbn esp32:esp32:m5stack_stickc_plus:UploadSpeed=115200 -p COM4 firmware/photobooth_remote
```

The stick logs connections, presses and status text on its USB serial port
(115200) for troubleshooting. No Windows pairing: the app finds the remote by its
Bluetooth service, connects, and reconnects automatically. The stick's screen
shows the link, battery, countdown and "SAVED"; it beeps on each press. To save
its small battery the screen turns fully off after 20 seconds with no press or
movement, and comes back on when it's picked up or a button is pressed (the booth
updates its text quietly, so a countdown never lights it by itself). To use two
booths in one room, set `remote.address` to pin each app to its own stick.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

`camera.index = "test"` (with `camera.name` empty) runs the booth on a
synthetic green-screen camera, for trying it without a camera.

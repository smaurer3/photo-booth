"""BLE link to the M5StickC Plus2 remote (firmware in firmware/photobooth_remote).

No Windows pairing: the app scans for the remote's service, connects, and
reconnects whenever it drops. Button presses arrive as notifications; the app
writes short status text back for the remote's screen.

Protocol (custom 128-bit service):
  BUTTON  notify  [event, seq]   event 1 = photo (A), 2 = next background (B),
                                 3 = previous background (B held), 4 = the remote moved
  STATUS  write   UTF-8 text, lines split by "\\n" (first line shown large)
  Battery level from the standard Battery Service.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading

log = logging.getLogger(__name__)

SERVICE_UUID = "6b913008-8f62-4fe5-8c46-3a3886474a74"
BUTTON_UUID = "77f040a3-7b3e-4b53-91d8-4cd35a27c289"
STATUS_UUID = "2f2d24b2-bf7a-4da4-b433-3a8f9046f1e7"
BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

PHOTO, NEXT, PREVIOUS, MOTION = 1, 2, 3, 4


class Remote(threading.Thread):
    """Runs bleak on its own asyncio loop; events go to a queue as ("remote", code)."""

    def __init__(self, cfg: dict, events: queue.Queue):
        super().__init__(daemon=True)
        self.address = (cfg.get("address") or "").upper()
        self.events = events
        self.connected = False
        self.battery = None
        self.name = None
        self._loop = None
        self._dirty = None
        self._status = ""
        self._halt = False

    # -- called from the UI thread --------------------------------------
    def send_status(self, text: str):
        """Show text on the remote (latest wins; sent when connected)."""
        if text == self._status:
            return
        self._status = text
        if self._loop and self._dirty:
            self._loop.call_soon_threadsafe(self._dirty.set)

    def stop(self):
        self._halt = True
        if self._loop and self._dirty:
            self._loop.call_soon_threadsafe(self._dirty.set)

    # -- BLE thread -------------------------------------------------------
    def run(self):
        try:
            asyncio.run(self._main())
        except Exception:
            log.exception("Remote thread crashed")

    def _matches(self, device, adv):
        if self.address and device.address.upper() != self.address:
            return False
        return SERVICE_UUID in [u.lower() for u in adv.service_uuids]

    async def _main(self):
        from bleak import BleakScanner

        self._loop = asyncio.get_running_loop()
        self._dirty = asyncio.Event()
        while not self._halt:
            try:
                device = await BleakScanner.find_device_by_filter(self._matches, timeout=8.0)
                if device is not None:
                    await self._session(device)
            except Exception as e:
                log.warning("Remote: %s", e)
                await asyncio.sleep(2)
            if self.connected:
                log.info("Remote disconnected")
            self.connected = False

    async def _session(self, device):
        from bleak import BleakClient

        lost = asyncio.Event()

        def on_button(_char, data: bytearray):
            if data:
                self.events.put(("remote", int(data[0])))

        def on_battery(_char, data: bytearray):
            if data:
                self.battery = int(data[0])

        async with BleakClient(device, disconnected_callback=lambda _c: lost.set()) as client:
            await client.start_notify(BUTTON_UUID, on_button)
            try:
                self.battery = int((await client.read_gatt_char(BATTERY_UUID))[0])
                await client.start_notify(BATTERY_UUID, on_battery)
            except Exception as e:
                log.info("Remote battery unavailable: %s", e)
            self.name = device.name
            self.connected = True
            log.info("Remote connected: %s (%s)", device.name, device.address)
            self._dirty.set()   # push the current status
            while not lost.is_set() and not self._halt:
                try:
                    await asyncio.wait_for(self._dirty.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                self._dirty.clear()
                if lost.is_set() or self._halt:
                    break
                text = self._status.encode("utf-8")[:120]
                await client.write_gatt_char(STATUS_UUID, text, response=True)

// Photo booth remote for the M5StickC Plus / Plus2.
//
//   Button A (front, big):  take a photo
//   Button B (side):        next background; hold for previous background
//
// The booth app connects over BLE by itself (no Windows pairing), and writes
// short status text back that is shown on the screen (countdown, "SAVED"...).
// UUIDs must match photobooth/remote.py.
//
// Build (pick the board; M5Unified detects it at runtime either way):
//   arduino-cli compile --fqbn esp32:esp32:m5stack_stickc_plus  firmware/photobooth_remote   (Plus, ESP32-PICO-D4)
//   arduino-cli compile --fqbn esp32:esp32:m5stack_stickc_plus2 firmware/photobooth_remote   (Plus2)
// Needs the M5Unified library.

#include <M5Unified.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>

static const char *DEVICE_NAME = "BoothRemote";
#define SERVICE_UUID "6b913008-8f62-4fe5-8c46-3a3886474a74"
#define BUTTON_UUID  "77f040a3-7b3e-4b53-91d8-4cd35a27c289"
#define STATUS_UUID  "2f2d24b2-bf7a-4da4-b433-3a8f9046f1e7"

enum : uint8_t { EVT_PHOTO = 1, EVT_NEXT = 2, EVT_PREVIOUS = 3, EVT_MOTION = 4 };

static const uint8_t ROTATION = 0;          // 0 = portrait, button A below the screen; 2 if upside down
static const uint8_t BRIGHT = 120;
static const uint32_t OFF_AFTER_MS = 20000;      // screen off after this long with no press or movement
static const bool BEEP = true;
static const float MOTION_G = 0.12f;        // change in acceleration (g) that wakes the screen
static const uint32_t IMU_EVERY_MS = 50;
static const uint32_t MOTION_EVENT_EVERY_MS = 1500;   // tell the app about movement at most this often

static BLECharacteristic *buttonChar;
static BLECharacteristic *batteryChar;
static volatile bool connected = false;
static volatile bool redraw = true;
static portMUX_TYPE statusMux = portMUX_INITIALIZER_UNLOCKED;
static char statusText[128] = "";
static uint8_t seq = 0;
static int battery = -1;
static uint32_t lastActivity = 0;
static uint32_t lastBatteryCheck = 0;
static uint32_t flashUntil = 0;
static bool screenOn = true;
static uint32_t screenOffAt = 0;
static float baseX, baseY, baseZ;           // resting acceleration, follows slow changes
static bool imuReady = false;
static uint32_t lastImu = 0;
static uint32_t lastMotionSent = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override {
    connected = true;
    redraw = true;
    Serial.println("connected");
  }
  void onDisconnect(BLEServer *) override {
    connected = false;
    Serial.println("disconnected");
    portENTER_CRITICAL(&statusMux);
    statusText[0] = 0;
    portEXIT_CRITICAL(&statusMux);
    redraw = true;
    BLEDevice::startAdvertising();
  }
};

class StatusCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String v = c->getValue();
    portENTER_CRITICAL(&statusMux);
    size_t n = v.length() < sizeof(statusText) - 1 ? v.length() : sizeof(statusText) - 1;
    memcpy(statusText, v.c_str(), n);
    statusText[n] = 0;
    portEXIT_CRITICAL(&statusMux);
    redraw = true;              // text updates quietly; only presses and movement light the screen (battery)
    Serial.printf("status: %s\n", v.c_str());
  }
};

static void draw();

static void wake() {
  lastActivity = millis();
  if (screenOn) return;
  screenOn = true;
  uint32_t asleep = millis() - screenOffAt;
  if (asleep < 120) delay(120 - asleep);   // the ST7789 ignores sleep-out this soon after sleep-in
  M5.Display.getPanel()->setSleep(false);
  delay(5);
  redraw = false;
  draw();                                  // fresh text before the light comes on, not the stale screen
  M5.Display.setBrightness(BRIGHT);
}

// Wake the screen when the stick is picked up or moved.
static void checkMotion(uint32_t now) {
  if (now - lastImu < IMU_EVERY_MS) return;
  lastImu = now;
  float x, y, z;
  if (!M5.Imu.getAccel(&x, &y, &z)) return;
  if (!imuReady) {
    baseX = x; baseY = y; baseZ = z;
    imuReady = true;
    return;
  }
  float d = fabsf(x - baseX) + fabsf(y - baseY) + fabsf(z - baseZ);
  baseX += (x - baseX) * 0.2f;
  baseY += (y - baseY) * 0.2f;
  baseZ += (z - baseZ) * 0.2f;
  if (d > MOTION_G) {
    if (!screenOn) Serial.printf("motion wake (%.2f g)\n", d);
    wake();
    // Movement also wakes the booth from its slideshow (silent, rate limited).
    if (connected && now - lastMotionSent > MOTION_EVENT_EVERY_MS) {
      lastMotionSent = now;
      uint8_t v[2] = {EVT_MOTION, ++seq};
      buttonChar->setValue(v, 2);
      buttonChar->notify();
    }
  }
}

static void sendEvent(uint8_t evt) {
  wake();
  Serial.printf("button event %d (%s)\n", evt, connected ? "sent" : "not connected");
  if (BEEP) M5.Speaker.tone(evt == EVT_PHOTO ? 3000 : 2000, 40);
  if (!connected) {
    flashUntil = millis() + 1500;   // show "not connected" briefly
    redraw = true;
    return;
  }
  uint8_t v[2] = {evt, ++seq};
  buttonChar->setValue(v, 2);
  buttonChar->notify();
}

static void updateBattery() {
  int level = M5.Power.getBatteryLevel();
  if (level < 0 || level == battery) return;
  battery = level;
  uint8_t b = (uint8_t)level;
  batteryChar->setValue(&b, 1);
  if (connected) batteryChar->notify();
  redraw = true;
}

static void draw() {
  auto &d = M5.Display;
  char text[128];
  portENTER_CRITICAL(&statusMux);
  strcpy(text, statusText);
  portEXIT_CRITICAL(&statusMux);

  d.startWrite();
  d.fillScreen(TFT_BLACK);

  // Top bar: link and battery.
  d.setFont(&fonts::Font2);
  d.setTextDatum(top_left);
  d.setTextColor(connected ? TFT_GREEN : TFT_ORANGE);
  d.drawString(connected ? "Linked" : "Searching", 4, 4);
  if (battery >= 0) {
    d.setTextDatum(top_right);
    d.setTextColor(battery < 15 ? TFT_RED : TFT_WHITE);
    d.drawString(String(battery) + "%", d.width() - 4, 4);
  }

  // Message: first line big, the rest small.
  const char *big = text;
  const char *small = "";
  char *nl = strchr(text, '\n');
  if (nl) {
    *nl = 0;
    small = nl + 1;
  }
  if (millis() < flashUntil && !connected) {
    big = "No link";
    small = "Is the booth running?";
  } else if (!connected) {
    big = "Booth";
    small = "waiting for app";
  }
  int cy = d.height() / 2 - 10;
  d.setTextDatum(middle_center);
  d.setTextColor(TFT_WHITE);
  if (strlen(big) <= 2) {
    d.setFont(&fonts::FreeSansBold24pt7b);   // countdown digits
    d.setTextSize(2);
  } else {
    d.setFont(&fonts::FreeSansBold12pt7b);
    d.setTextSize(1);
    while (d.textWidth(big) > d.width() - 6 && d.getTextSizeX() > 0.5f) d.setTextSize(d.getTextSizeX() - 0.1f);
  }
  d.drawString(big, d.width() / 2, cy);
  d.setTextSize(1);
  if (*small) {
    d.setFont(&fonts::Font2);
    d.setTextColor(TFT_LIGHTGREY);
    String s(small);
    while (s.length() > 1 && d.textWidth(s) > d.width() - 6) s.remove(s.length() - 1);
    d.drawString(s, d.width() / 2, cy + 46);
  }

  // Footer: what the buttons do.
  d.setFont(&fonts::Font0);
  d.setTextColor(TFT_DARKGREY);
  d.setTextDatum(bottom_center);
  d.drawString("A: photo  B: background", d.width() / 2, d.height() - 14);
  d.drawString("hold B: previous", d.width() / 2, d.height() - 3);
  d.endWrite();
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);                 // detects StickC / Plus / Plus2; latches the Plus2 power hold pin
  Serial.begin(115200);
  Serial.printf("photobooth remote: board %d, display %dx%d\n", (int)M5.getBoard(), M5.Display.width(), M5.Display.height());
  M5.Display.setRotation(ROTATION);
  M5.Display.setBrightness(BRIGHT);
  M5.Speaker.setVolume(80);
  M5.BtnB.setHoldThresh(600);

  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(185);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *svc = server->createService(SERVICE_UUID);
  buttonChar = svc->createCharacteristic(BUTTON_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  buttonChar->addDescriptor(new BLE2902());
  BLECharacteristic *statusChar = svc->createCharacteristic(
      STATUS_UUID, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  statusChar->setCallbacks(new StatusCallbacks());
  svc->start();

  BLEService *batt = server->createService(BLEUUID((uint16_t)0x180F));
  batteryChar = batt->createCharacteristic(BLEUUID((uint16_t)0x2A19),
                                           BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  batteryChar->addDescriptor(new BLE2902());
  batt->start();

  // Service UUID in the advert (that's what the app scans for), name in the scan response.
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  BLEAdvertisementData advData;
  advData.setFlags(0x06);
  advData.setCompleteServices(BLEUUID(SERVICE_UUID));
  adv->setAdvertisementData(advData);
  BLEAdvertisementData scanData;
  scanData.setName(DEVICE_NAME);
  adv->setScanResponseData(scanData);
  BLEDevice::startAdvertising();

  updateBattery();
  lastActivity = millis();
}

void loop() {
  M5.update();
  if (M5.BtnA.wasPressed()) sendEvent(EVT_PHOTO);
  if (M5.BtnB.wasHold()) sendEvent(EVT_PREVIOUS);
  else if (M5.BtnB.wasClicked()) sendEvent(EVT_NEXT);

  uint32_t now = millis();
  if (now - lastBatteryCheck > 10000) {
    lastBatteryCheck = now;
    updateBattery();
  }
  if (flashUntil && now > flashUntil) {
    flashUntil = 0;
    redraw = true;
  }
  checkMotion(now);
  if (redraw && screenOn) {      // while off, wake() draws the latest text
    redraw = false;
    draw();
  }
  // millis(), not now: wake() above may have set lastActivity after now was read.
  if (screenOn && millis() - lastActivity > OFF_AFTER_MS) {
    screenOn = false;
    screenOffAt = millis();
    M5.Display.sleep();          // backlight supply off and the panel asleep
    Serial.println("screen off");
  }
  delay(10);
}

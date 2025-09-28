// arduino_serial_csv.ino (demo)
// Prints: ISO8601, temp_c, bpm, ibi_ms. Replace BPM/IBI with your sensor libs.
#include <Arduino.h>

const int THERM_PIN = A0;
const float SERIES_RESISTOR = 10000.0;
const float NOMINAL_RESISTANCE = 10000.0;
const float NOMINAL_TEMPERATURE = 25.0;
const float BETA_COEFFICIENT = 3950.0;

float thermistorTempC() {
  int adc = analogRead(THERM_PIN);
  float v = (adc / 1023.0) * 5.0;
  float r = SERIES_RESISTOR * (5.0 / v - 1.0);
  float steinhart = r / NOMINAL_RESISTANCE;
  steinhart = log(steinhart);
  steinhart /= BETA_COEFFICIENT;
  steinhart += 1.0 / (NOMINAL_TEMPERATURE + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;
  return steinhart;
}

int bpm_sim = 72;
int ibi_sim = 830;

String iso8601_now() {
  unsigned long ms = millis();
  char buf[40];
  snprintf(buf, sizeof(buf), "1970-01-01T00:%02lu:%02luZ", (ms/60000)%60, (ms/1000)%60);
  return String(buf);
}

void setup() {
  Serial.begin(115200);
  analogReference(DEFAULT);
  delay(1000);
  Serial.println("# iso_datetime, temp_c, bpm, ibi_ms");
}

void loop() {
  float tempC = thermistorTempC();
  bpm_sim = 72 + (int)(6.0 * sin(millis()/1000.0));
  ibi_sim = (int)(60000.0 / max(40, bpm_sim));
  String ts = iso8601_now();
  Serial.print(ts); Serial.print(", ");
  Serial.print(tempC, 2); Serial.print(", ");
  Serial.print(bpm_sim); Serial.print(", ");
  Serial.println(ibi_sim);
  delay(200);
}

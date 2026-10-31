# Projektteendők a következő játéknapra (tulajdonos)

A projekt nyitott próbái és a kiadás előtti mérések, amelyeket csak a tulajdonos tud elvégezni a referenciagépen (UMC1820, ASIO). A játéknap menete a nyilvános [SESSION_CHECKLIST_HU.md](../SESSION_CHECKLIST_HU.md)-ban van. Ez a lista a [RELEASING.md](../RELEASING.md) v0.1.0-kapujához tartozik.

A `recordings/`, a `bench/refs/` és a `rig.toml` git-ignorált, személyes adat: soha ne kerüljön commitba, és a tartalmuk (szöveg, nevek) ne kerüljön issue-ba vagy naplóba. Visszajelzésként csak számok és `[OK]`/`[FAIL]` sorok mennek.

## 1. M5 nyitott próbák (a játék előtt, külön próbamappába)

- [ ] **Halk csatorna:** az egyik csatorna gainjét erősen tekerd le, majd futtasd: `uv run ttrpg record check --rig rig.toml`. Erre a csatornára `quiet` riasztásnak kell jönnie. **Utána tekerd vissza a gaint!**
- [ ] **Kihúzás:**
  1. Indítsd el: `uv run ttrpg record start --rig rig.toml --session recordings/teszt-kihuzas --no-check --seconds 180`.
  2. Közben húzd ki a UMC1820 USB-kábelét kb. 10 s-ra, majd dugd vissza.
  3. Elvárt: `device_lost` riasztás, utána `reconnected`, és a felvétel folytatódik.
  4. Végül `uv run ttrpg record verify --session recordings/teszt-kihuzas`: egy `device_lost` kimaradás, egyéb hiba nélkül.
- [ ] **WASAPI `audio_hw`:** a `record devices` WASAPI-sorából vedd az eszköznevet, majd PowerShellben:
  ```powershell
  $env:TTRPG_RECORDER_TEST_DEVICE = "<WASAPI-eszköznév>"; $env:TTRPG_RECORDER_TEST_HOST_API = "wasapi"
  uv run pytest -m audio_hw
  ```

## 2. A játéknapon

- [ ] **Mikrofonpróba úgy, hogy mindenki beszél:** a session elején a vezetett próba (ellenőrzőlista, 2. pont). Elvárt: minden résztvevő `OK`, téves riasztás nélkül.
- [ ] **Teljes session és `record verify`** (ellenőrzőlista, 4. pont). Elvárt: `0 problem(s)`, egyforma csatornahosszak.
- [ ] **Visszajelzés:** a leállításkori `[OK] stopped: …` és a `verify` `[OK]`/`[FAIL]` sorát másold be nekem; szöveget nem kell.

## 3. Utána: clip02 a triage-hez

- [ ] **Részlet kiválasztása:** legalább **5 perces**, élénk beszélgetés: többen beszélnek egymás után, van egymásra beszélés, és megszólal halkabb beszélő is. Az időpontot az átirat időbélyegeiből veheted.
- [ ] **Kivágás:** minden sávot egyszerre vág ki, a session időtengelyén igazítva; az eredeti fájlokhoz nem nyúl:
  ```bash
  uv run ttrpg bench clip cut bench/refs/clip02 --session recordings/2026-11-05 --start 1:23:00 --minutes 5 --language hu
  ```
- [ ] **Triage-címkézés** a clip01-hez hasonlóan (útmutató: [REFERENCE_GUIDE.md](../REFERENCE_GUIDE.md), kb. 15–20 perc):
  ```bash
  uv run ttrpg bench ref triage bench/refs/clip02
  uv run ttrpg bench ref triage bench/refs/clip02 --stats
  ```
- [ ] **Bleed-mérés:** ha a státusz `complete`, futtasd, és küldd el a számokat (szöveget ne):
  ```bash
  uv run ttrpg bench bleed bench/refs/clip02
  ```
  Ebből lesznek véglegesek az M3 ideiglenes tűrései és az átfedés-jelölés, és ebből erősítjük meg az alapértelmezett ASR-módot (M1-benchmark a ≥ 5 perces klipen).

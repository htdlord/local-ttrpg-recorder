# Session-napi ellenőrzőlista

Egy játéknap felvételéhez, a referenciagépen (UMC1820, ASIO). A példákban a session mappája `recordings/2026-11-05`; minden parancs a repó gyökeréből fut. A `recordings/`, a `bench/refs/` és a `rig.toml` git-ignorált, személyes adat: soha ne kerüljön commitba.

## 1. Mielőtt a játékosok megérkeznek (kb. 20 perc)

- [ ] A UMC1820 be van kapcsolva és USB-n csatlakozik. Más program (Audacity, DAW, Discord ASIO-n) **nem** fogja az ASIO-drivert.
- [ ] A Windows Update szüneteltetve van, vagy a használati idő lefedi az estét (egy kényszerített újraindítás megszakítaná a felvételt). Az alvást a felvevő maga tiltja le.
- [ ] Elég hely van a lemezen. A `record start` induláskor ellenőrzi, hogy 8 óra tömörítetlen 24 bites hang + 10% elfér-e; ha nem, el sem indul.
- [ ] A rig rendben van: `rig.toml` a bekötött csatornákkal és a résztvevőkkel. Ha a mai ülésrend vagy a résztvevők mások, mentsd újra:
  ```bash
  uv run sessionscribe record devices
  uv run sessionscribe record rig save rig.toml --device "UMC ASIO Driver" --host-api asio --channels 1,4-8 --names gm,anna,... --labels "Mesélő,Anna,..."
  ```
- [ ] Minden csíptetős mikrofon a helyén (mellkas közepén, kb. 15–20 cm-re a szájtól), a kábelek nem feszülnek.
- [ ] Gyors szintpróba: `uv run sessionscribe record meters --rig rig.toml`. Mindenki mondjon pár szót, és a saját csatornája mozduljon (Ctrl+C kilép).
- [ ] **Tartalék felvétel:** egy telefon vagy kézi felvevő az asztal közepén, a mikrofonpróba előtt elindítva. Ha a gép vagy az interfész menet közben kiesik, ebből megvan a hang.

**M5 nyitott próbák, ha ma van rá idő (a játék előtt, külön próbamappába):**

- [ ] **Halk csatorna:** az egyik csatorna gainjét erősen tekerd le, majd `uv run sessionscribe record check --rig rig.toml`. Erre a csatornára `quiet` riasztásnak kell jönnie. **Utána tekerd vissza a gaint!**
- [ ] **Kihúzás:** `uv run sessionscribe record start --rig rig.toml --session recordings/teszt-kihuzas --no-check --seconds 180`. Közben húzd ki a UMC1820 USB-kábelét kb. 10 s-ra, majd dugd vissza. Elvárt: `device_lost` riasztás, utána `reconnected`, a felvétel folytatódik. Végül `record verify --session recordings/teszt-kihuzas`: egy `device_lost` kimaradás, egyéb hiba nélkül.
- [ ] **WASAPI:** a `record devices` WASAPI-sorából vedd az eszköznevet, majd PowerShellben:
  ```powershell
  $env:SESSIONSCRIBE_TEST_DEVICE = "<WASAPI-eszköznév>"; $env:SESSIONSCRIBE_TEST_HOST_API = "wasapi"
  uv run pytest -m audio_hw
  ```

## 2. Indítás: mikrofonpróba és felvétel

```bash
uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05
```

- [ ] Először csend (5 s): mindenki hallgat. Utána a program név szerint szólít mindenkit, és mindenki kb. 5 s-ig beszél (bármit, természetes hangerővel). *(Ez az M5 nyitott próbája is: mikrofonpróba úgy, hogy mindenki beszél.)*
- [ ] Nézd meg a riasztásokat (alább). Ha minden OK vagy csak WARN van, a felvétel magától folytatódik; ettől kezdve minden a lemezre kerül.
- [ ] A terminál ablakát hagyd nyitva, ne kattints bele (a kijelölés a Windows-konzolban megállíthatja a kiírást). A felvételt **Ctrl+C** állítja le.

### A riasztások jelentése és teendői

| Riasztás | Mit jelent | Teendő |
|---|---|---|
| `no_signal` (FAIL) | a csatornán a saját körében sem emelkedett a jel, vagy digitális csend van | kábel, mikrofon bekapcsolva? Phantomtáp a csatornán? Jó csatornára van dugva? Javítás után indítsd újra (lásd 3.) |
| `wrong_channel` (FAIL) | valaki más csatornája emelkedett jobban a körében: felcserélt bekötés | cseréld a kábeleket, vagy javítsd a rigben a sorrendet, majd indítsd újra |
| `quiet` (WARN) | a beszéd nagyon halk, vagy sokkal kisebb, mint a többieké | gain feljebb, mikrofon közelebb a szájhoz; mehet tovább is, de rosszabb lesz az átirat |
| `clipping` (WARN) | a csúcs elérte a −1 dBFS-t | gain lejjebb; a torzított hang rontja az átiratot |
| `idle_channel` (WARN, felvétel közben) | egy csatorna 10 perce csak alapzajt ad, miközben mások beszélnek | lehúzódott kábel, kikapcsolt mikrofon? Ha valaki csak csendben van, nincs teendő |
| `device_lost` (FAIL) → `reconnected` (INFO) | megszakadt a hang az interfészről, majd visszajött | kábel/táp ellenőrzése. A kiesett idő kimaradásként (gap) kerül a manifestbe; a tartalék felvétel lefedi |
| `disk_low` (WARN) | kb. 30 percnyi hely maradt | szabadíts fel helyet most; 2 perc alatt a felvétel tisztán leáll (`disk_full`) |
| `fatal` (FAIL) | a felvevő nem tud tovább írni | lásd 3., „összeomlás” |

## 3. Ha valami elromlik

- **FAIL a mikrofonpróbán:** a program leáll. Javítsd a hibát, és futtasd **ugyanazt** a parancsot újra: ugyanabba a mappába folytatja, a közte eltelt időt kimaradásként jelöli, és újra lefut a próba.
  - Ha nem lehet megjavítani (pl. egy mikrofon ma nem működik), és így is el akarod indítani: `--force` kapcsolóval a FAIL ellenére folytatja. A hibás csatorna attól még felvételre kerül; jegyezd fel, kinek nem ment.
- **Megszakadt a hang (`device_lost`), és 1 percen belül nem jön `reconnected`:** Ctrl+C, ellenőrizd a kábelt és az interfészt, majd folytasd ugyanabba a mappába, próba nélkül:
  ```bash
  uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05 --no-check
  ```
- **Összeomlás, lefagyás, újraindult a gép:** először helyreállítás, csak utána folytatás:
  ```bash
  uv run sessionscribe record recover --session recordings/2026-11-05
  uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05 --no-check
  ```
  A helyreállítás a félbemaradt fájlok olvasható részét menti; az eredetit `.partial` néven megtartja.
- **Semmi nem segít:** a tartalék felvétel fut tovább. Írd fel, mikor (óra:perc) állt le a gépes felvétel, hogy később tudjuk, mi hiányzik.

## 4. A session végén

- [ ] Ctrl+C a felvevő terminálban; megvárni, amíg kiírja: `[OK] stopped: … gap(s) …`.
- [ ] Ellenőrzés:
  ```bash
  uv run sessionscribe record verify --session recordings/2026-11-05
  ```
  Elvárt: `0 problem(s)`, és minden csatorna ugyanolyan hosszú. *(Ez az M5 nyitott próbája: teljes session felvétele és `record verify`.)*
- [ ] A tartalék felvételt is mentsd el a session mappája mellé.
- [ ] A két `[OK]`/`[FAIL]` sort másold be nekem; szöveget nem kell.

## 5. Utána: átírás és a clip02 a triage-hez

- [ ] Átírás (vagy a GUI-ban: Session megnyitása → Futtatás):
  ```bash
  uv run sessionscribe transcribe --session recordings/2026-11-05 -l hu
  ```
- [ ] Válassz ki egy **legalább 5 perces** részt, ahol élénk a beszélgetés: többen beszélnek egymás után, van egymásra beszélés, és szólal meg halkabb beszélő is. Az időpontot az átirat időbélyegeiből veheted.
- [ ] Kivágás, minden sáv egyszerre, a session időtengelyén igazítva (az eredeti fájlokhoz nem nyúl):
  ```bash
  uv run sessionscribe bench clip cut bench/refs/clip02 --session recordings/2026-11-05 --start 1:23:00 --minutes 5 --language hu
  ```
- [ ] Triage-címkézés, ahogy a clip01-nél (útmutató: `docs/REFERENCE_GUIDE.md`, kb. 15–20 perc):
  ```bash
  uv run sessionscribe bench ref triage bench/refs/clip02
  uv run sessionscribe bench ref triage bench/refs/clip02 --stats
  ```
- [ ] Ha a státusz `complete`, futtasd a bleed-mérést, és küldd el a számokat (szöveget ne):
  ```bash
  uv run sessionscribe bench bleed bench/refs/clip02
  ```
  Ebből rögzítjük véglegesre az M3 ideiglenes tűréseit és az átfedés-jelölést.

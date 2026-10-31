# Session-napi ellenőrzőlista

Egy játéknap felvételéhez, egy többcsatornás hangkártyával és résztvevőnként egy csiptetős mikrofonnal. A példákban:
- a session mappája `recordings/2026-11-05`;
- a hangkártya ASIO-drivere `"Az interfész ASIO-drivere"` (a pontos nevet a `record devices` írja ki);
- minden parancs a klónozott SessionScribe-mappából fut (lásd [INSTALL.md](INSTALL.md)).

A felvételek és a rig-fájl személyes adatot tartalmaznak (hangok, nevek): ne oszd meg és ne commitold őket. A `recordings/` és a `rig.toml` ezért git-ignorált.

## 1. Mielőtt a játékosok megérkeznek (kb. 20 perc)

- [ ] **Hangkártya:** be van kapcsolva és csatlakozik. Más program (DAW, hangszerkesztő, ASIO-t használó csevegőprogram) **nem** fogja az ASIO-drivert.
- [ ] **Windows Update:** szüneteltetve, vagy a használati idő lefedi az estét, mert egy kényszerített újraindítás megszakítaná a felvételt. Az alvást a felvevő maga tiltja le.
- [ ] **Lemezhely:** a `record start` induláskor ellenőrzi, hogy a tervezett hossz (alapból 8 óra) tömörítetlen 24 bites hangja + 10% elfér-e; ha nem, el sem indul.
- [ ] **Rig:** a `rig.toml` a bekötött csatornákkal és a résztvevőkkel. Ha a mai ülésrend vagy a résztvevők mások, mentsd újra:
  ```bash
  uv run sessionscribe record devices
  uv run sessionscribe record rig save rig.toml --device "Az interfész ASIO-drivere" --host-api asio --channels 1-4 --names gm,anna,bela,cili --labels "Mesélő,Anna,Béla,Cili"
  ```
- [ ] **Mikrofonok:** mindegyik a helyén (a mellkas közepén, kb. 15–20 cm-re a szájtól), a kábelek nem feszülnek.
- [ ] **Gyors szintpróba:** `uv run sessionscribe record meters --rig rig.toml`. Mindenki mondjon pár szót, és a saját csatornája mozduljon (Ctrl+C kilép).
- [ ] **Tartalék felvétel:** egy telefon vagy kézi felvevő az asztal közepén, a mikrofonpróba előtt elindítva. Ha a gép vagy a hangkártya menet közben kiesik, ebből megvan a hang.

## 2. Indítás: mikrofonpróba és felvétel

```bash
uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05
```

A GUI-ban ugyanez: **Felvétel** fül, rig betöltése, session-mappa, **Felvétel indítása**.

- [ ] **Csend:** először 5 s-ig mindenki hallgat.
- [ ] **Körök:** utána a program név szerint szólít mindenkit, és mindenki kb. 5 s-ig beszél (bármit, természetes hangerővel).
- [ ] **Riasztások:** nézd meg őket (alább). Ha minden OK, vagy csak WARN van, a felvétel magától folytatódik; ettől kezdve minden a lemezre kerül.
- [ ] **Terminál:** hagyd nyitva, és ne kattints bele, mert a kijelölés a Windows-konzolban megállíthatja a kiírást. A felvételt **Ctrl+C** állítja le (a GUI-ban a **Leállítás** gomb). A GUI ablaka bezárható, a felvétel a háttérben fut tovább.

### A riasztások jelentése és teendői

| Riasztás | Mit jelent | Teendő |
|---|---|---|
| `no_signal` (FAIL) | a csatornán a saját körében sem emelkedett a jel, vagy digitális csend van | Ellenőrizd a kábelt, hogy be van-e kapcsolva a mikrofon, van-e fantomtáp a csatornán, és jó csatornára van-e dugva. Javítás után indítsd újra (lásd 3.) |
| `wrong_channel` (FAIL) | valaki más csatornája emelkedett jobban a körében: felcserélt bekötés | Cseréld meg a kábeleket, vagy javítsd a sorrendet a rigben, majd indítsd újra |
| `quiet` (WARN) | a beszéd nagyon halk, vagy sokkal halkabb, mint a többieké | Gain feljebb, mikrofon közelebb a szájhoz. Folytathatod így is, de rosszabb lesz az átirat |
| `clipping` (WARN) | a csúcs elérte a −1 dBFS-t | Gain lejjebb; a torzított hang rontja az átiratot |
| `idle_channel` (WARN, felvétel közben) | egy csatornán 10 perce csak alapzaj van, miközben mások beszélnek | Lehúzódott a kábel, vagy kikapcsolt a mikrofon? Ha az illető csak hallgat, nincs teendő |
| `device_lost` (FAIL), majd `reconnected` (INFO) | megszakadt a hang a hangkártyáról, majd visszajött | Ellenőrizd a kábelt és a tápot. A kiesett idő kimaradásként (gap) kerül a manifestbe, ezt a részt a tartalék felvétel fedi le |
| `disk_low` (WARN) | kb. 30 percnyi hely maradt | Szabadíts fel helyet most; 2 perc múlva a felvétel szabályosan leáll (`disk_full`) |
| `fatal` (FAIL) | a felvevő nem tud tovább írni | Lásd 3., „Összeomlás” |

## 3. Ha valami elromlik

- **FAIL a mikrofonpróbán:** a program leáll. Javítsd a hibát, és futtasd **ugyanazt** a parancsot újra. Ugyanabba a mappába folytatja, a közben eltelt időt kimaradásként jelöli, és újra lefut a próba.
  - Ha nem lehet megjavítani (pl. egy mikrofon ma nem működik), de így is indítanád: a `--force` kapcsolóval a FAIL ellenére folytatja (a GUI-ban: „Sikertelen mikrofonpróba esetén is folytassa”). A hibás csatorna attól még felvételre kerül; jegyezd fel, kinek nem ment.
- **Megszakadt a hang (`device_lost`), és 1 percen belül nem jön `reconnected`:** Ctrl+C, ellenőrizd a kábelt és a hangkártyát, majd folytasd ugyanabba a mappába, próba nélkül:
  ```bash
  uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05 --no-check
  ```
- **Összeomlás, lefagyás, újraindult a gép:** előbb helyreállítás, csak utána folytatás. A GUI ilyenkor maga ajánlja fel a **Helyreállítás** gombot.
  ```bash
  uv run sessionscribe record recover --session recordings/2026-11-05
  uv run sessionscribe record start --rig rig.toml --session recordings/2026-11-05 --no-check
  ```
  A helyreállítás a félbemaradt fájlok olvasható részét menti, az eredetit `.partial` néven megtartja.
- **Semmi nem segít:** a tartalék felvétel fut tovább. Írd fel, mikor (óra:perc) állt le a gépes felvétel, hogy később tudd, mi hiányzik.

## 4. A session végén

- [ ] **Leállítás:** Ctrl+C a felvevő terminálban (vagy **Leállítás** a GUI-ban), és várd meg, amíg kiírja: `[OK] stopped: … gap(s) …`.
- [ ] **Ellenőrzés:**
  ```bash
  uv run sessionscribe record verify --session recordings/2026-11-05
  ```
  Elvárt: `0 problem(s)`, és minden csatorna ugyanolyan hosszú. Ha van kimaradás, a `verify` megmondja, hol és mennyi.
- [ ] **Tartalék felvétel:** mentsd el a session mappája mellé.

## 5. Utána: átírás

- [ ] **Átírás** (a GUI-ban: Session megnyitása → **Futtatás**):
  ```bash
  uv run sessionscribe transcribe --session recordings/2026-11-05 -l hu
  ```
  Megszakítás után ugyanez a parancs onnan folytatja, ahol abbahagyta.
- [ ] **Áttekintés:** az **Eredmény** fülön szűrj a jelölt tételekre (Átnézendő, Átfedés, Kiszűrt, Lehetséges kimaradt beszéd). Egy szegmensre kattintva meghallgathatod az eredeti sávból, és jobb klikkel másik résztvevőhöz sorolhatod.
- [ ] **Újraexport:** nevek, offsetek vagy átsorolás változása után nem kell újra átírni:
  ```bash
  uv run sessionscribe export --session recordings/2026-11-05
  ```
- [ ] **Takarítás:** ha már nincs szükség rá, a `work/` gyorsítótár (benne átiratszöveg) törölhető: `uv run sessionscribe clean --session recordings/2026-11-05`.

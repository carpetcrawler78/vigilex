# Claude-Code-Auftrag: Triage-Balken auf rot/gelb/gruen (Live-Grafana)

Ziel: im Panel "Triage Categories (Post-Fix Records)" des Dashboards
`sentinelai-coding-v1` die drei Balken faerben:
  01 Flag for Review = rot | 02 Uncertain = gelb | 03 Accept = GRUEN
Aktuell ist "03 Accept" tuerkis (Default-Palette).

## KRITISCH -- erst lesen
Die **lokale** Datei `grafana/provisioning/dashboards/sentinelai_coding.json`
ist VERALTET und entspricht NICHT dem Live-Dashboard (andere Panels). Das echte
Dashboard wurde per Grafana-API editiert (DEVLOG 2026-05-31). Daher:
- NIEMALS die lokale Repo-JSON nach Hetzner provisionieren -- das wuerde das
  gute Live-Dashboard mit der alten Version ueberschreiben.
- Aenderung NUR am LIVE-Dashboard via Grafana-API.
- IMMER zuerst Backup. Bei Unklarheit ueber die Panel-Struktur: STOPP, Thomas fragen.

## Voraussetzungen
- Grafana laeuft auf Hetzner. Zugriff entweder direkt auf dem Server
  (`ssh cap@46.225.109.99`) gegen `http://localhost:3000`, oder lokal ueber den
  bestehenden SSH-Tunnel (`localhost:3000`).
- Admin-Credentials NICHT raten: aus der Hetzner-`.env` lesen
  (Variable wie GF_SECURITY_ADMIN_USER / GF_SECURITY_ADMIN_PASSWORD; in
  `docker-compose.yml` pruefen wie Grafana konfiguriert ist). Wenn nicht
  auffindbar -> STOPP, Thomas fragen.

## Schritte
1. **Backup (Pflicht):**
   `curl -s -u <user>:<pass> http://localhost:3000/api/dashboards/uid/sentinelai-coding-v1 > /tmp/dash_backup_$(date +%Y%m%d_%H%M).json`
   Pruefen, dass die Datei das Panel "Triage Categories" enthaelt. Wenn nicht ->
   STOPP (dann ist die UID/Instanz anders als erwartet).

2. **Panel-Struktur inspizieren (nicht raten):**
   Im Backup-JSON das Triage-Panel finden. Feststellen, WIE die Balkenfarben
   aktuell gesetzt sind (eine der Moeglichkeiten):
   - `fieldConfig.defaults.color.mode` (palette-classic o.ae.), ODER
   - `fieldConfig.overrides` mit matcher `byName`/`byField` je Kategorie, ODER
   - value `mappings` mit `color`.
   Den BESTEHENDEN Mechanismus weiterverwenden, nicht einen neuen erfinden.

3. **Farben setzen:**
   Overrides so setzen, dass die drei Kategorien fix sind:
   01 Flag for Review -> `red`, 02 Uncertain -> `yellow`, 03 Accept -> `green`
   (Grafana-Standardfarbnamen oder fixedColor-Hex, je nach bestehendem Stil).
   Nur dieses eine Panel anfassen, sonst nichts.

4. **Zurueckschreiben via API:**
   Das modifizierte `dashboard`-Objekt (mit unveraenderter `uid`, `id`,
   inkrementierter/`overwrite:true`) posten:
   `curl -s -u <user>:<pass> -H "Content-Type: application/json" \
     -d '{"dashboard": <modified>, "overwrite": true}' \
     http://localhost:3000/api/dashboards/db`
   Auf `"status":"success"` pruefen.

5. **Verifizieren:**
   Dashboard erneut per API GETen, bestaetigen dass die drei Farb-Overrides
   drin sind und die Version hochgezaehlt wurde. Optional: im Browser (Tunnel)
   Slide 12 / `localhost:3000` reloaden -> 03 Accept ist gruen.

6. **Revert verhindern (wie am 31.05.):**
   Pruefen, ob Grafana das Dashboard aus einer Provisioning-Datei AUF HETZNER
   laedt (Pfad aus `docker-compose.yml` Grafana-Volume ableiten). Falls ja:
   das NEU exportierte Live-JSON in genau diese Hetzner-Provisioning-Datei
   schreiben (nicht die Repo-Datei!) und Grafana-Reload triggern, damit ein
   Container-Restart die Farben nicht zuruecksetzt.

7. **Repo nachziehen (optional, sauber):**
   Das finale Live-JSON nach `grafana/provisioning/dashboards/sentinelai_coding.json`
   exportieren, damit Repo == Live. Branch `work`:
   `git checkout work && git add ... && git commit -m "grafana: triage bars red/yellow/green" && git push origin work`.

## Sicherheits-Stopps
- Kein Schritt 4/6 ohne erfolgreiches Backup aus Schritt 1.
- Wenn das Triage-Panel value-mappings/overrides nutzt, die du nicht eindeutig
  zuordnen kannst -> STOPP, Struktur an Thomas zeigen, nicht raten.
- Lokale Repo-JSON NIE nach Hetzner kopieren/provisionieren.

## DEVLOG (vigilex/DEVLOG.md) am Ende appenden
```
## 2026-06-XX HH:MM -- Grafana Triage-Farben
- DONE: 03 Accept auf gruen (rot/gelb/gruen), via API am Live-Dashboard, Backup gesichert, Provisioning-File synchronisiert
```

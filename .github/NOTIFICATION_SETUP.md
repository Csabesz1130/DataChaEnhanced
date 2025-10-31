# 🔔 Release Notification System Setup

Ez a dokumentum leírja, hogyan állítsd be a release értesítési rendszert a GitHub Actions workflow-ban.

## 📧 Email Értesítések

### Szükséges GitHub Secrets:

```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

### GitHub Variables:

```bash
ENABLE_EMAIL_NOTIFICATIONS=true
```

### Gmail App Password beállítása:

1. Menj a Google Account beállításokhoz
2. Biztonság → 2 lépcsős ellenőrzés
3. Alkalmazás jelszavak → Új alkalmazás jelszó generálása
4. Használd ezt a jelszót az `EMAIL_PASSWORD` secret-ben

## 🔗 Webhook Értesítések

### Szükséges GitHub Secrets:

```bash
WEBHOOK_URLS=https://webhook1.example.com/endpoint,https://webhook2.example.com/endpoint
```

### GitHub Variables:

```bash
ENABLE_WEBHOOK_NOTIFICATIONS=true
WEBHOOK_TYPE=generic  # generic, slack, discord
WEBHOOK_TIMEOUT=30
WEBHOOK_RETRY_ATTEMPTS=3
WEBHOOK_RETRY_DELAY=5
WEBHOOK_CUSTOM_HEADERS={"Authorization": "Bearer token123"}
```

### Webhook típusok:

- **generic**: Általános HTTP POST kérések
- **slack**: Slack-compatible formátum
- **discord**: Discord-compatible formátum

## 📱 Telegram Bot Értesítések

### Szükséges GitHub Secrets:

```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_IDS=-1001234567890,987654321
```

### GitHub Variables:

```bash
ENABLE_TELEGRAM_NOTIFICATIONS=true
TELEGRAM_NOTIFICATION_TYPE=standard  # standard, channel
TELEGRAM_TIMEOUT=30
TELEGRAM_RETRY_ATTEMPTS=3
TELEGRAM_RETRY_DELAY=5
```

### Telegram Bot létrehozása:

1. Beszélj a [@BotFather](https://t.me/botfather) bot-tal
2. `/newbot` parancs
3. Add meg a bot nevét és felhasználónevét
4. Mentsd el a bot tokent
5. Add hozzá a bot-ot a csatornához/csoporthoz
6. Kérj chat ID-t a [@userinfobot](https://t.me/userinfobot) bot-tól

## 🚀 Használat

### 1. Secrets beállítása

A GitHub repository-ban:
1. Settings → Secrets and variables → Actions
2. Add meg a fenti secret-eket
3. Add meg a fenti variable-okat

### 2. Értesítések engedélyezése/tiltása

```bash
# Minden értesítés engedélyezése
ENABLE_EMAIL_NOTIFICATIONS=true
ENABLE_WEBHOOK_NOTIFICATIONS=true
ENABLE_TELEGRAM_NOTIFICATIONS=true

# Egyes értesítések tiltása
ENABLE_EMAIL_NOTIFICATIONS=false
```

### 3. Release létrehozása

```bash
# Git tag létrehozása
git tag v1.0.0
git push origin v1.0.0

# Vagy GitHub webes felületen
# Releases → Create a new release
```

## 🔧 Hibaelhárítás

### Email problémák:

- Ellenőrizd az SMTP beállításokat
- Gmail esetén használj App Password-t
- Ellenőrizd a tűzfal beállításokat

### Webhook problémák:

- Ellenőrizd a webhook URL-eket
- Teszteld a webhook endpoint-okat
- Nézd meg a GitHub Actions logokat

### Telegram problémák:

- Ellenőrizd a bot tokent
- Ellenőrizd a chat ID-kat
- Győződj meg róla, hogy a bot hozzá van adva a csatornához

## 📊 Monitoring

A workflow minden értesítési típusról jelentést ad:

```bash
📧 Email notifications: Enabled
🔗 Webhook notifications: Enabled
📱 Telegram notifications: Enabled
```

## 🛡️ Biztonság

- **Soha ne commitolj secret-eket** a kódba
- Használj **App Password-okat** email szolgáltatásokhoz
- **Korlátozd a webhook hozzáférési jogokat**
- **Rendszeresen frissítsd a bot tokeneket**

## 📝 Példa konfiguráció

```yaml
# .github/workflows/release.yml részlet
env:
  ENABLE_EMAIL_NOTIFICATIONS: true
  ENABLE_WEBHOOK_NOTIFICATIONS: true
  ENABLE_TELEGRAM_NOTIFICATIONS: true
  WEBHOOK_TYPE: slack
  TELEGRAM_NOTIFICATION_TYPE: channel
```

## 🆘 Támogatás

Ha problémák merülnek fel:

1. Ellenőrizd a GitHub Actions logokat
2. Teszteld a konfigurációt lokálisan
3. Ellenőrizd a secret-ek és variable-ok beállításait
4. Nézd meg a script-ek debug üzeneteit

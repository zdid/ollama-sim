# ollama-sim

Faux serveur **Ollama** qui implémente fidèlement le protocole Ollama v0.5.x
et route toutes les requêtes vers **Mistral API**.

Home Assistant (ou tout client Ollama) n'y voit que du feu.

---

## Démarrage rapide

```bash
# 1. Cloner / dézipper le projet
cd ollama-sim

# 2. Configurer la clé Mistral
cp .env.example .env
# Éditer .env et coller votre clé MISTRAL_API_KEY

# 3. Construire et démarrer
docker compose up -d --build

# 4. Vérifier
curl http://localhost:11434/api/version
# → {"version":"0.5.1"}

curl http://localhost:11434/api/tags
# → {"models":[{"name":"mistral",...}, ...]}
```

---

## Configuration Home Assistant

Dans **Paramètres → Assistants → Ollama** :

| Champ | Valeur |
|-------|--------|
| URL   | `http://<IP_DE_LA_MACHINE>:11434` |
| Modèle | `mistral` (ou `mistral-large`, `codestral`, …) |

> Si Home Assistant tourne dans Docker sur la même machine :
> utilisez `http://ollama-sim:11434` et ajoutez le service
> au même réseau Docker.

---

## Modèles disponibles

| Nom Ollama          | Modèle Mistral réel       |
|---------------------|---------------------------|
| `mistral`           | mistral-small-latest      |
| `mistral:7b`        | mistral-small-latest      |
| `mistral-large`     | mistral-large-latest      |
| `mistral-nemo`      | open-mistral-nemo         |
| `codestral`         | codestral-latest          |

Modifier `MODEL_MAP` dans `app/main.py` pour en ajouter.

---

## Endpoints implémentés

| Méthode | Route            | Description                        |
|---------|------------------|------------------------------------|
| GET     | `/api/version`   | Handshake Ollama                   |
| GET     | `/api/tags`      | Liste des modèles                  |
| POST    | `/api/show`      | Détails d'un modèle                |
| POST    | `/api/chat`      | Chat (streaming NDJSON)            |
| POST    | `/api/generate`  | Generate (ancien format, délégué)  |
| POST    | `/api/embeddings`| Stub embeddings                    |
| GET     | `/api/ps`        | Modèles en mémoire (liste vide)    |
| DELETE  | `/api/delete`    | Suppression modèle (no-op)         |

---

## Logs

```bash
# Temps réel
docker compose logs -f

# Fichier persistant
tail -f logs/ollama-sim.log
```

Chaque échange est logué avec blocs séparés :
```
────────────────────────────────────────────────────────────────
📨 HA → PROXY  [/api/chat]
{ "model": "mistral", "messages": [...] }
────────────────────────────────────────────────────────────────
🚀 PROXY → MISTRAL  [mistral-small-latest]
{ "model": "mistral-small-latest", "messages": [...], "stream": true }
────────────────────────────────────────────────────────────────
```

---

## Arrêt / redémarrage

```bash
docker compose down      # arrêter
docker compose restart   # redémarrer
docker compose up -d     # relancer en arrière-plan
```

---

## Dépannage

| Symptôme | Cause probable | Solution |
|----------|---------------|----------|
| `503` dans HA | `MISTRAL_API_KEY` vide | Vérifier `.env` |
| Modèle absent | Nom non présent dans `MODEL_MAP` | Ajouter dans `main.py` |
| Timeout | Réseau vers api.mistral.ai | Vérifier firewall sortant |
| HA ne trouve pas l'hôte | URL mal configurée | Vérifier l'IP dans HA |

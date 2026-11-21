#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/usr/sbin:/usr/bin:/sbin:/bin
umask 077


# Recevoir le SHA Git depuis GitHub Actions
if ! IFS= read -r -t 15 DEPLOY_SHA; then
    echo "Erreur : aucun SHA Git reçu." >&2
    exit 1
fi

# Vérifier le format du SHA (40 caractères hexadécimaux)
if [[ ! "$DEPLOY_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Erreur : SHA Git invalide." >&2
    exit 1
fi

# Utiliser l'image Docker correspondant au commit
IMAGE="ghcr.io/imanemekkaoui1/predictive-maintenance-mlops:sha-${DEPLOY_SHA}"

echo "Déploiement du commit : $DEPLOY_SHA"

APP="maintenance-api"
BACKUP="maintenance-api-rollback"

# Empêcher deux déploiements simultanés
exec 9>/run/lock/predictive-maintenance-deploy.lock
flock -n 9 || { echo "Déploiement déjà en cours"; exit 1; }

# Éviter d'écraser une ancienne sauvegarde
if docker container inspect "$BACKUP" >/dev/null 2>&1; then
    echo "Un conteneur de sauvegarde existe déjà."
    exit 1
fi

backup_made=0
app_attempted=0

rollback() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "Échec du déploiement."

        if [ "$backup_made" -eq 1 ]; then
            docker rm -f "$APP" >/dev/null 2>&1 || true
            docker rename "$BACKUP" "$APP"
            docker start "$APP" >/dev/null
            echo "Ancienne version restaurée."
        elif [ "$app_attempted" -eq 1 ]; then
            docker rm -f "$APP" >/dev/null 2>&1 || true
        fi
    fi
}
trap rollback EXIT

echo "Téléchargement de l'image..."
docker pull "$IMAGE"

# Sauvegarder le conteneur existant
if docker container inspect "$APP" >/dev/null 2>&1; then
    docker rename "$APP" "$BACKUP"
    backup_made=1
    docker stop "$BACKUP" >/dev/null
fi

echo "Démarrage de la nouvelle version..."
app_attempted=1

docker run -d \
    --name "$APP" \
    --cpus="1.0" \
    --memory="2g" \
    --memory-swap="2g" \
    --restart unless-stopped \
    -p 127.0.0.1:18000:8000 \
    "$IMAGE"

echo "Vérification de l'API..."

for i in {1..20}; do
    if curl -fsS http://127.0.0.1:18000/health \
        | grep -q '"model_loaded":true'; then

        echo "API opérationnelle !"

        if [ "$backup_made" -eq 1 ]; then
            docker rm "$BACKUP" >/dev/null
        fi

        echo "Déploiement réussi."

        # Nettoyage uniquement après déploiement réussi
        CLEANUP="/opt/predictive-maintenance/cleanup_docker_images.sh"

        if [[ -f "$CLEANUP" ]]; then
            echo "Nettoyage des anciennes images Docker..."

            if ! bash "$CLEANUP" --apply; then
                echo "AVERTISSEMENT : nettoyage échoué, API conservée." >&2
            fi
        else
            echo "Script de nettoyage absent : étape ignorée."
        fi

        exit 0
    fi

    sleep 2
done

echo "L'API ne répond pas correctement."
docker logs --tail 30 "$APP" || true
exit 1

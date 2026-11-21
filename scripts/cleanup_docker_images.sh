
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="ghcr.io/imanemekkaoui1/predictive-maintenance-mlops"
APP="maintenance-api"
BACKUP="maintenance-api-rollback"

MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--apply" ]]; then
    echo "Usage: $0 [--dry-run|--apply]" >&2
    exit 1
fi

# Ne jamais nettoyer pendant un rollback
if docker container inspect "$BACKUP" >/dev/null 2>&1; then
    echo "Sauvegarde détectée : nettoyage annulé."
    exit 1
fi

# Vérifier que la production fonctionne
docker container inspect "$APP" >/dev/null

RUNNING=$(docker inspect -f '{{.State.Running}}' "$APP")

if [[ "$RUNNING" != "true" ]]; then
    echo "Production arrêtée : nettoyage annulé."
    exit 1
fi

ACTIVE=$(docker inspect -f '{{.Config.Image}}' "$APP")

if [[ ! "$ACTIVE" =~ ^ghcr.io/imanemekkaoui1/predictive-maintenance-mlops:sha-[0-9a-f]{40}$ ]]; then
    echo "Image active inattendue : nettoyage annulé."
    exit 1
fi

docker info >/dev/null

# Conserver l'image active et deux versions récentes
declare -A KEEP=()
KEEP["${ACTIVE##*:}"]=1
COUNT=1

mapfile -t IMAGES < <(
    docker image ls --format '{{.Repository}}|{{.Tag}}' |
    while IFS='|' read -r repo tag; do
        [[ "$repo" == "$REPO" ]] || continue
        [[ "$tag" =~ ^sha-[0-9a-f]{40}$ ]] || continue

        created=$(docker image inspect \
            --format '{{.Created}}' "$REPO:$tag")

        printf '%s|%s\n' "$created" "$tag"
    done | LC_ALL=C sort -r
)

for item in "${IMAGES[@]}"; do
    tag="${item#*|}"

    if [[ -n "${KEEP[$tag]+x}" ]]; then
        continue
    fi

    if (( COUNT < 3 )); then
        KEEP["$tag"]=1
        COUNT=$((COUNT + 1))
    fi
done

echo "Mode : $MODE"
echo "Image active : ${ACTIVE##*:}"
echo "Politique : image active + 2 versions récentes"

for item in "${IMAGES[@]}"; do
    tag="${item#*|}"

    if [[ -n "${KEEP[$tag]+x}" ]]; then
        echo "CONSERVER : $tag"
    else
        if [[ "$MODE" == "--dry-run" ]]; then
            echo "SUPPRIMER (simulation) : $tag"
        else
            echo "SUPPRESSION : $tag"
            docker image rm "$REPO:$tag"
        fi
    fi
done

echo "Nettoyage terminé."

# Historique technique du projet

**Maintenance prédictive industrielle — Master SDIA, ENSET Mohammedia**

Ce document synthétise les étapes du travail en trinôme. Il conserve la chronologie technique de l'ancien README, mise à jour avec les développements de supervision, de sécurité et de reprise sur échec validés en novembre 2026.

## Phase 1 — Cadrage et préparation des données

1. Définir le besoin industriel : identifier les observations associées à des défaillances de machines.
2. Délimiter les trois sorties : décision de défaillance, probabilité, niveau de risque ; exclure la prédiction RUL.
3. Sélectionner AI4I 2020 pour ses caractéristiques de fonctionnement industriel et sa cible binaire.
4. Examiner le déséquilibre : 339 pannes sur 10 000 observations.
5. Exclure les identifiants et les indicateurs de panne susceptibles d'introduire une fuite d'information.
6. Réaliser un split stratifié 80/20 avec graine 42.
7. Préparer les variables numériques et encoder `Type`.
8. Versionner les jeux de données via DVC avec stockage distant Google Drive.

## Phase 2 — Expérimentation et sélection du modèle

9. Définir les métriques adaptées aux pannes rares : précision, rappel, F1, ROC-AUC et PR-AUC.
10. Évaluer Decision Tree comme modèle de référence interprétable.
11. Évaluer Random Forest et des stratégies de prise en compte de la classe minoritaire.
12. Entraîner et analyser plusieurs configurations XGBoost.
13. Examiner les variantes baseline, pondération et SMOTE en évitant la contamination du test.
14. Tracer les expériences et artefacts avec MLflow.
15. Sélectionner le modèle final en tenant compte du rappel des pannes et des faux positifs.
16. Optimiser le seuil binaire et retenir `0.15` pour l'API.
17. Définir les niveaux de risque `Faible`, `Moyen` et `Élevé`.
18. Exporter le modèle final sous `models/final_model/`.

## Phase 3 — Industrialisation de l'API

19. Implémenter FastAPI et la validation des données reçues.
20. Exposer `/predict`, `/health` et la documentation Swagger.
21. Ajouter les tests de l'API avec pytest.
22. Ajouter les contrôles statiques Ruff dans la CI.
23. Conteneuriser l'application avec Docker et Docker Compose.
24. Instrumenter l'API avec `/metrics` et un compteur de prédictions par risque.

## Phase 4 — Publication continue et cloud

25. Implémenter la CI GitHub Actions avec Python 3.12, Ruff, pytest et build Docker.
26. Publier les images dans GHCR pour `linux/amd64` et `linux/arm64`.
27. Mettre en place des tags Docker fondés sur le SHA Git du commit.
28. Configurer le serveur Oracle Cloud Linux sans exposer l'API publiquement.
29. Créer les accès SSH et les secrets nécessaires au CD.
30. Automatiser le déploiement par `workflow_run` après une CI réussie sur `main`.
31. Ajouter un contrôle de santé applicatif et une sauvegarde temporaire du conteneur précédent.
32. Vérifier le déploiement de l'image correcte sur Oracle.

## Phase 5 — Observabilité et résilience

33. Déployer localement Prometheus et Grafana avec Docker Compose.
34. Exposer les métriques FastAPI, les latences, le débit et les erreurs.
35. Configurer le dashboard Grafana et sa datasource par provisioning.
36. Mettre en place un tunnel SSH pour la collecte des métriques Oracle.
37. Compléter les 12 panneaux : six indicateurs locaux et six indicateurs Oracle.
38. Tester trois prédictions réelles sur Oracle, puis confirmer leur comptabilisation dans Grafana.
39. Déclencher un échec volontaire de health check dans un laboratoire isolé et valider la restauration automatique par comparaison des identifiants Docker.
40. Nettoyer l'environnement de test, préserver l'API de production et vérifier `/health`.

## Phase 6 — Maîtrise des ressources et documentation

41. Mesurer l'espace disque Oracle et identifier les anciennes images Docker.
42. Libérer de l'espace en supprimant explicitement les tags devenus inutiles.
43. Versionner `scripts/deploy.sh` et `scripts/cleanup_docker_images.sh`.
44. Mettre en place une rétention d'images : version active plus deux versions SHA récentes.
45. Tester le nettoyage en simulation, puis l'intégrer après un déploiement réussi.
46. Vérifier le CD sur le commit `5faeb54`, l'état sain de FastAPI et les trois images conservées.
47. Mettre à jour le README, les diagrammes, les captures et le guide d'exploitation.

## Références de validation

- [CI/CD #20 — commit `5faeb54`](https://github.com/ImaneMekkaoui1/predictive-maintenance-mlops/actions/runs/44902713425).
- [Historique CD Oracle](https://github.com/ImaneMekkaoui1/predictive-maintenance-mlops/actions/workflows/deploy.yml).
- Captures du dashboard, du CD et du rollback dans [`assets/`](assets/).

Pour une vue synthétique des composants, consulter le [README principal](../README.md).

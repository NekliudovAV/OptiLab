# Сохранение докер-контейнеров:
1. docker commit runner1 runner_backup
2. docker save -o runner.tar runner_backup
3. docker commit gitlab gitlab_backup
4. docker csave -o gitlab.tar gitlab_backup
# Загрузка докер-контейнеров:
5. docker load < runner.tar
6. docker load < gitlab.tar
7. docker run -d --name runner runner1:latest
8. docker run -d --name gitlab gitlab:latest


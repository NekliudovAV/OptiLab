# Установка gitlab
1. docker-compose up -d
2. После того, как всё установится и настроится, необходимо откорректировать файл: /etc/gitlab/gitlab.rb
(1256-1258 строки)

   puma['worker_processes'] = 2

   puma['min_threads'] = 1

   puma['max_threads'] = 4

3. Перезапустить
4. Можно отключить логирование prometheus, чтобы не разрастась папка data: /etc/gitlab/gitlab.rb
(2591 стока)

   prometheus_monitoring['enable'] = false

5. Настройка автоматического бэкапирования:
   
   apt-get update
   
   apt-get install crontab (первый вызов)
   
   Добавляем запись вызова:

   /tmp/crontab.wRRRid/crontab
   
   (Каждый день в 11 чвасов выпоняется backup)

   0 11 * * * /opt/gitlab/bin/gitlab-backup create

   березагружаем docker
   
   
   ## Комманды для справки:
   Остановка gitlab: gitlab-ctl stop
   
   Запуск gitlab: gitlab-ctl start   

# Контейнеры, которые должны быть загружены в Docker для старта CICD:

1. docker:dind
   
2. docker scip_pyomo_9_2
   
3. kemgres_opt2:latest

## Сохранение контейнеров:

1. docker save -o docker.tar docker:dind
   
2. docker save -o scip_pyomo9_2.tar scip_pyomo_9_2
   
3. docker save -o kemgres_opt2.tar kemgres_opt2:latest
   Скаченные образы
   
4. docker save -o runner.tar gitlab/gitlab-runner:alpine
   
5. docker save -o gitlab.tar gitlab/gitlab-ce:latest

## Загрузка контейнеров:

1. docker load < docker.tar
   
2. docker load < kemgres_opt2.tar
   
3. docker load < scip_pyomo9_2.tar

4. docker load < runner.tar

5. docker load < gitlab.tar

## Старт контейнеров

1. sudo docker run -d --name gitlab -p 8929:8929 -p 2424:22 -p 443:443 -v /home/master/gitlab/config:/etc/gitlab -v /home/master/gitlab/logs:/var/log/gitlab -v /home/master/gitlab/data:/var/opt/gitlab -v /home/master/gitlab/backup:/var/opt/backups gitlab/gitlab-ce:latest
   
2. sudo docker run -d --name gitlab
   
# BackUp Gitlab
1. Создани BackUp:
   gitlab-backup create
2. Резервное копирование конфигурационных файлов:
   sudo tar -czvf gitlab_config_backup_$(date +%F).tar.gz /etc/gitlab
3. Проверка резервной копии:
   ls -l /var/opt/gitlab/backups
4. Автоматизация резервного копирования:
   Вы можете настроить автоматическое резервное копирование с помощью cron. Например, чтобы создавать резервную    копию каждый день в 2:00, добавьте задачу в cron:
   sudo crontab -e
   Добавьте строку:
   0 2 * * * /opt/gitlab/bin/gitlab-backup create
5. Восстановление из резервной копии:
   sudo gitlab-backup restore BACKUP=название_резервной_копии
6. Восстановление конфигурационных файлов:
   sudo tar -xzvf gitlab_config_backup_дата.tar.gz -C /
7. Перезапуск Gitlab:
   sudo gitlab-ctl restart

# Сохранение инфраструктурных докер-контейнеров:
1. docker commit gitlab-runner1 runner_backup
2. docker save -o runner.tar runner_backup
   
4. docker commit gitlab gitlab_backup
5. docker save -o gitlab.tar gitlab_backup
   
7. docker commit mongo mongo_backup
8. docker save -o mongo.tar mongo_backup

9. docker commit influx influx_backup
10. docker save -o influx.tar influx_backup

# Загрузка докер-контейнеров:
1. docker load < runner.tar
2. docker run -d --name runner runner_backup:latest
   
3. docker load < gitlab.tar
4. sudo docker run -d --name gitlab -p 8929:8929 -p 2424:22 -p 443:443 -v /home/master/gitlab_b/config:/etc/gitlab -v /home/master/gitlab_b/logs:/var/log/gitlab -v /home/master/gitlab_b/data:/var/opt/gitlab -v /home/master/gitlab_b/backup:/var/opt/backups gitlab_backup:latest


6. docker load < mongo.tar
7. docker run -d --name mongo -p 27017:27017 mongo_backup:latest   

8. docker load < influx.tar
9. docker run -d --name influx -p 8086:8086 influx_backup:latest



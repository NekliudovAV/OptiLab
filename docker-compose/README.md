# Установка gitlab
1. docker-compose up -d
2. После того, как всё установится и настроится, необходимо откорректировать файл: /etc/gitlab/gitlab.rb
(1256-1258 строки)   
puma['worker_processes'] = 2
puma['min_threads'] = 1
puma['max_threads'] = 4


# Сохранение докер-контейнеров:
1. docker commit gitlab-runner1 runner_backup
2. docker save -o runner.tar runner_backup
   
4. docker commit gitlab gitlab_backup
5. docker csave -o gitlab.tar gitlab_backup
   
7. docker commit mongo mongo_backup
8. docker save -o mongo.tar mongo_backup

9. docker commit influx influx_backup
10. docker save -o influx.tar influx_backup

# Загрузка докер-контейнеров:
1. docker load < runner.tar
2. docker run -d --name runner runner_backup:latest
   
3. docker load < gitlab.tar
4. sudo docker run -d --name gitlab1 -p 8929:8929 -p 2424:22 -p 443:443 -v /home/master/gitlab/config:/etc/gitlab -v /home/master/gitlab/logs:/var/log/gitlab -v /home/master/gitlab/data:/var/opt/gitlab gitlab_backup:latest

5. docker load < mongo.tar
6. docker run -d --name mongo -p 27017:27017 mongo_backup:latest   

7. docker load < influx.tar
8. docker run -d --name influx -p 8086:8086 influx_backup:latest

# BackUp Gitlab
1. Остановка сервера
   gitlab-ctl stop
2. Создани BackUp
   gitlab-backup create
3. Резервное копирование конфигурационных файлов
   sudo tar -czvf gitlab_config_backup_$(date +%F).tar.gz /etc/gitlab
4. Проверка резервной копии
   ls -l /var/opt/gitlab/backups
5. Автоматизация резервного копирования
   Вы можете настроить автоматическое резервное копирование с помощью cron. Например, чтобы создавать резервную    копию каждый день в 2:00, добавьте задачу в cron:
   sudo crontab -e
   Добавьте строку:
   0 2 * * * /opt/gitlab/bin/gitlab-backup create
6. Восстановление из резервной копии
   sudo gitlab-backup restore BACKUP=название_резервной_копии
7. Восстановление конфигурационных файлов:
   sudo tar -xzvf gitlab_config_backup_дата.tar.gz -C /
8. Перезапуск Gitlab
   sudo gitlab-ctl restart

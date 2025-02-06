# Подготовка WLSI win10

1. dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

2. dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

3. Скачиваем и устанавливаем пакет обновления ядра Linux https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi

4. wsl --set-default-version 2

5. Выбрать и устоновать Ubuntu

    [https://tretyakov.net/post/ustanovit-docker-na-windows-10-wsl2/?ysclid=m6tdd4d877367942673]

# Используемые образы (Images):

| №     | Контейнер | Описание |
|-------| ------- | -------- |
| 1.1 | mongo | Хранение Датафреймов |
| 1.2 | influx | Хранение Временных рядов |
| 1.3 | postgres | Замена на единое храниилще в перспективе |
|2	| Grafana	| Визуализация расчётов | 
|3.1| 	gitlab | Хранение проектов | 
|3.2|	runner | Формирование образов | 
|3.3|	docker | Докер-контейнер для запуска сборки | 
|3.4|	helper | Докер-контейнер для запуска сборки | 
|4|	scip_pyomo9_2	| Образ солвера | 
|5|	jupyter	| Юпитер ноутбук | 
|6|	airflow	| Запуск задач по расписанию | 

   
  
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

   перезагрузить gitlab
   
   
   ## Комманды для справки:
   Остановка gitlab: gitlab-ctl stop
   
   Запуск gitlab: gitlab-ctl start   

# Контейнеры, которые должны быть загружены в Docker для старта CICD:

1. docker:dind
   
2. docker scip_pyomo_9_2
   
3. gitlab-ce:latest

4. gitlab-runner:alpine

## Сохранение контейнеров:

1. docker save -o docker.tar docker:dind
   
2. docker save -o scip_pyomo9_2.tar scip_pyomo_9_2
    
3. docker save -o runner.tar gitlab/gitlab-runner:alpine
   
4. docker save -o gitlab.tar gitlab/gitlab-ce:latest

## Загрузка контейнеров:

1. docker load < docker.tar
   
2. docker load < scip_pyomo9_2.tar

3. docker load < runner.tar

4. docker load < gitlab.tar

## Сбор сонтейнеров:

docker-compose docker-compose_CICD.yml up -d

## Старт контейнеров (в ручную)

1. sudo docker run -d --name gitlab -p 8929:8929 -p 2424:22 -p 443:443 -v /home/master/gitlab/config:/etc/gitlab -v /home/master/gitlab/logs:/var/log/gitlab -v /home/master/gitlab/data:/var/opt/gitlab -v /home/master/gitlab/backup:/var/opt/backups gitlab/gitlab-ce:latest
   
2. sudo docker run -d --name gitlab
   
# BackUp Gitlab

0. Заходим в образ
   sudo docker exec -ti gitlab_b /bin/bash

1. Создани BackUp:
   gitlab-backup create

2. Резервное копирование конфигурационных файлов (если требуется):
   sudo tar -czvf gitlab_config_backup_$(date +%F).tar.gz /etc/gitlab

3. Проверка резервной копии:
   ls -l /var/opt/gitlab/backups

4. Автоматизация резервного копирования:
   Вы можете настроить автоматическое резервное копирование с помощью cron. Например, чтобы создавать резервную    копию каждый день в 2:00, добавьте задачу в cron:
   sudo crontab -e
   Добавьте строку:
   0 2 * * * /opt/gitlab/bin/gitlab-backup create

5. Копируем в образ   
   sudo docker cp 1738763953_2025_02_05_17.7.0_gitlab_backup.tar gitlab_b:/var/opt/gitlab/backups

6. Заходим в образ    
   sudo docker exec -ti gitlab_b /bin/bash

7. Восстановление из резервной копии:
   sudo gitlab-backup restore #(если восстановить не последнюю версию) BACKUP=название_резервной_копии

8. Восстановление конфигурационных файлов (если требуется):
   sudo tar -xzvf gitlab_config_backup_дата.tar.gz -C /

9. Копирвоание   

   
10. Восстановление образа
   gitlab-backup restore

11. Перезапуск Gitlab:
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
4. sudo docker run -d --name gitlab -p 8929:8929 -p 2424:22 -p 443:443 -v /home/master/gitlab_b/config:/etc/gitlab -v /home/master/gitlab_b/logs:/var/log/gitlab -v /home/master/gitlab_b/data:/var/opt/gitlab -v /home/master/gitlab_b/backup:/var/opt/gitlab/backups gitlab_backup:latest


6. docker load < mongo.tar
7. docker run -d --name mongo -p 27017:27017 mongo_backup:latest   

8. docker load < influx.tar
9. docker run -d --name influx -p 8086:8086 influx_backup:latest



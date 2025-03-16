# Пример сборки контейнеров

docker-compose -p "databases" -f=docker-compose_influx.yml up -d

docker-compose -p "databases" -f=docker-compose_mongo.yml up -d

docker-compose -p "databases" -f=docker-compose_postgress.yml up -d

download file scip:

https://github.com/scipopt/scip/releases



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

   
  



# Контейнеры, которые должны быть загружены в Docker для старта CICD:

1. docker:dind
   
2. docker scip_pyomo_9_2
   
3. gitlab-ce:latest

4. gitlab-runner:alpine
   
5. gitlab/gitlab-runner-helper:ubuntu-x86_64-v17.7.0


## Сохранение контейнеров:

1. docker save -o docker.tar docker:dind
   
2. docker save -o scip_pyomo9_2.tar scip_pyomo_9_2
    
3. docker save -o runner.tar gitlab/gitlab-runner:alpine
   
4. docker save -o gitlab.tar gitlab/gitlab-ce:latest

5. docker pull gitlab/gitlab-runner-helper:ubuntu-x86_64-v17.7.0

   docker save -o helper.tar gitlab/gitlab-runner-helper:ubuntu-x86_64-v17.7.0

## Загрузка контейнеров:

1. docker load < docker.tar
   
2. docker load < scip_pyomo9_2.tar

3. docker load < runner.tar

4. docker load < gitlab.tar
   
5. docker load < helper.tar

# Установка gitlab

1. docker-compose docker-compose_CICD.yml up -d

2. После того, как всё установится и настроится, необходимо откорректировать файл: /etc/gitlab/gitlab.rb
(1256-1258 строки) Сократится используемая оперативная память

   puma['worker_processes'] = 2

   puma['min_threads'] = 1

   puma['max_threads'] = 4

3. Можно отключить логирование prometheus, чтобы не разрастась папка data: /etc/gitlab/gitlab.rb
(2591 стока)

   prometheus_monitoring['enable'] = false

4. Настройка автоматического бэкапирования (ТРЕБУЕТСЯ ДОРАБОКА):
   
   apt-get update
   
   apt-get install crontab (первый вызов)
   
   Добавляем запись вызова:

   /tmp/crontab.wRRRid/crontab
   
   (Каждый день в 11 чвасов выпоняется backup)

   0 11 * * * /opt/gitlab/bin/gitlab-backup create  
  
   ## Комманды для справки:
   Остановка gitlab: gitlab-ctl stop
   
   Запуск gitlab: gitlab-ctl start

## Настройка Runner:

1. Необходимо зайти в файл "/etc/gitlab-runner/config.toml" и поменять строчку
volumes = ["/cache"] на
volumes = ["/var/run/docker.sock:/var/run/docker.sock", "/cache"]

2. Добавить строчку
pull_policy = "if-not-present" 

## Пример старта контейнера gitlab (в ручную)

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

Как показал эксперемент, при восстановлении проектов слетает админка CICD. Возможный выход - миграция

## Миграция репозитория проекта

Проще создать новые контейнеры, выполнить нвстройку образов и миграцию баз.

    git clone --mirror [http://10.251.0.106:8929/root/kemgres.git] # Старый репозиторий проекта
    
    cd kemgres.git/
    
    git remote rm origin
    
    git remote add origin [http://10.16.0.153:8929/root/kemgres2.git] # Новый репозиторий проекта
    
    git push origin --all


# Бэкапирование и восстановление баз данных

## Сохранение докер-контейнеров баз данный:
   
1. docker commit mongo mongo_backup

2. docker save -o mongo.tar mongo_backup

3. docker commit influx influx_backup

4. docker save -o influx.tar influx_backup

## Загрузка докер-контейнеров:
   
1. docker load < mongo.tar

2. docker run -d --name mongo -p 27017:27017 mongo_backup:latest   

3. docker load < influx.tar

4. docker run -d --name influx -p 8086:8086 influx_backup:latest

## Бэкапирование и восстановление данных:

## InfluxDB

1. Заходим в контейнер

docker exec -ti  influx88 /bin/bash

2. Бэкапирование 

influxd backup -portable /backup/influxdb1

3. Запаковка

tar -czvf  /backup/backup_2025_01_30.tar.gz -C /backup influxdb1

4. Копирование на 10.16.0.157

docker cp influx88:/backup/backup_2025_01_30.tar.gz  /home/master/

5. Копирование на локальную машину

   Использовать WinSCP
6. копирование на докер-машину

   mkdir backup
   
   docker cp /docker/BackUpImages/2025_01/influx_backup/backup_2025_01_30.tar.gz influxdb:/backup/backup_2025_01_30.tar.gz 

7. Заходим в контейнер

   docker exec -ti  influxdb /bin/bash
   
8. распаковка

    tar -xzvf  /backup/backup_2025_01_30.tar.gz -C /backup/

9. Восстановление базы
    
    influxd restore -portable /backup/influxdb1

10. Удаление лишнего 

    rm -r backup
## Mongo

1.Заходим в контейнер

    docker exec -ti mongo  /bin/bash

2. Устанавливается тулза

    apt-get install mongodb-database-tool
   
3. Бэкапирование 

    mongodump --uri="mongodb://mongo:mongo@localhost:27017" --out=/backup/mongodb
   
4. Формирование архива

    tar -czvf  /backup/backup_mongo_2025_01_30.tar.gz -C /backup mongodb
   
5. Копирование на 10.16.0.157

    docker cp mongo:/backup/backup_mongo_2025_01_30.tar.gz  /home/master/

6. Копирование на локальную машину

    Использовать WinSCP

7. Создание папки

   docker exec -ti mongodb mkdir backup

8. Копирование на докер-машину
   
   docker cp /docker/BackUpImages/2025_01/mongo_backup/backup_mongo_2025_01_30.tar.gz mongodb:/backup/backup_mongo_2025_01_30.tar.gz

9. Вход в контейнер
    
   docker exec -ti  mongodb /bin/bash

10 Распаковка

    tar -xzvf  /backup/backup_mongo_2025_01_30.tar.gz -C /backup/ 

11 Восстановление базы

    mongorestore --uri="mongodb://mongo:mongo@localhost:27017" /backup/mongodb


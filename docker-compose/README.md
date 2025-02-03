# Сохранение докер-контейнеров:
1. docker commit runner1 runner_backup
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
4. docker run -d --name gitlab gitlab_backup:latest

5. docker load < mongo.tar
6. docker run -d --name mongo -p 27017:27017 mongo_backup:latest   

7. docker load < influx.tar
8. docker run -d --name influx -p 8086:8086 influx_backup:latest

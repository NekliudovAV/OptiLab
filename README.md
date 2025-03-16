# OptiLab
Optimization models for operating modes of technological process

Downkoad Optilab:

git clone https://github.com/NekliudovAV/OptiLab.git

# Установка баз данных для запуска примеров в Docker: 

cd docker-compose

## Start InfluxDBstart

docker compose -f docker-compose_influx.yml up -d

## Start MongoDB

docker compose -f docker-compose_mongo.yml up -d

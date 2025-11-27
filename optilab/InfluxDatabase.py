from influxdb import DataFrameClient,InfluxDBClient
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import config
from tqdm import tqdm
import time
from datetime import datetime, timedelta
@dataclass
class InfluxConfig:
    """Конфигурация подключения к InfluxDB"""
    db_name: str
    ip: str
    port: int
        

class InfluxDBManager:
    """Класс для управления операциями с InfluxDB"""
    
    def __init__(self, config_data: Optional[Dict] = None):
        self.config = self._load_config(config_data)
        self.client = None
        
    def _load_config(self, config_data: Optional[Dict] = None) -> InfluxConfig:
        """Загрузка конфигурации InfluxDB"""
        if config_data is None:
            config_data = config.INFLUX
            
        return InfluxConfig(
            db_name=config_data['DB_name'],
            ip=config_data['IP_'],
            port=config_data['port_']
        )
    
    def connect(self, database: Optional[str] = None) -> None:
        """Установка соединения с InfluxDB"""
        db_name = database or self.config.db_name
        self.client = DataFrameClient(
            host=self.config.ip,
            port=self.config.port,
            database=db_name
        )
    
    def disconnect(self) -> None:
        """Закрытие соединения с InfluxDB"""
        if self.client:
            self.client.close()
    
    def __enter__(self):
        """Контекстный менеджер"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер"""
        self.disconnect()
    
    def create_database(self, database: str) -> bool:
        """
        Создание новой базы данных
        
        Args:
            database: Имя базы данных
            
        Returns:
            bool: Успешность операции
        """
        try:
            client = InfluxDBClient(
                host=self.config.ip,
                port=self.config.port
            )
            existing_dbs = client.get_list_database()
            
            if database not in [db['name'] for db in existing_dbs]:
                client.create_database(database)
                print(f"База данных {database} создана успешно")
                return True
            else:
                print(f"База данных {database} уже существует")
                return False
                
        except Exception as e:
            print(f"Ошибка при создании БД: {e}")
            return False
        finally:
            client.close()
    
    def drop_measurement(self, measurement: str, database: Optional[str] = None) -> bool:
        """
        Удаление измерения
        
        Args:
            measurement: Имя измерения
            database: Имя базы данных
            
        Returns:
            bool: Успешность операции
        """
        try:
            db_name = database or self.config.db_name
            client = InfluxDBClient(
                host=self.config.ip,
                port=self.config.port,
                database=db_name
            )
            client.drop_measurement(measurement)
            return True
        except Exception as e:
            print(f"Ошибка при удалении измерения: {e}")
            return False
        finally:
            client.close()




@dataclass
class InfluxDataPoint:
    measurement: str
    fields: Dict[str, float]
    tags: Dict[str, str]
    timestamp: Optional[datetime] = None
    
    def to_influx_format(self) -> Dict[str, Any]:
        return {
            "measurement": self.measurement,
            "tags": self.tags,
            "fields": self.fields,
            "time": self.timestamp or datetime.utcnow()
        }

@dataclass
class InfluxBatch:
    points: List[InfluxDataPoint]
    
    def to_influx_format(self) -> List[Dict[str, Any]]:
        return [point.to_influx_format() for point in self.points]
    

class InfluxDataBuilder:
    def __init__(self, measurement: str):
        self.measurement = measurement
        self.fields = {}
        self.tags = {}
        self.timestamp = None
    
    def with_field(self, name: str, value: float) -> 'InfluxDataBuilder':
        self.fields[name] = value
        return self
    
    def with_tag(self, name: str, value: str) -> 'InfluxDataBuilder':
        self.tags[name] = value
        return self
    
    def with_timestamp(self, timestamp: datetime) -> 'InfluxDataBuilder':
        self.timestamp = timestamp
        return self
    
    def build(self) -> InfluxDataPoint:
        return InfluxDataPoint(
            measurement=self.measurement,
            fields=self.fields,
            tags=self.tags,
            timestamp=self.timestamp
        )            

class BaseTags:
    FLEET: str = "fleet"
    EQUIPMENT: str = "equipment"
    TYPE_CALC: str = "type_calc"
    SCENARIO: str = "scenario"
    MODEL: str = "model"
    VERSION: str = "version"

class DefaultTagValues:
    FLEET = "none"
    EQUIPMENT = "All"
    TYPE_CALC = "calc"
    SCENARIO = "Base"
    MODEL = "Base"
    VERSION = "1"

class TagPreset:
    @staticmethod
    def basic_preset() -> Dict[str, str]:
        return {
            BaseTags.MODEL: DefaultTagValues.MODEL,
            BaseTags.EQUIPMENT: DefaultTagValues.EQUIPMENT,
            BaseTags.TYPE_CALC: DefaultTagValues.TYPE_CALC,
            BaseTags.SCENARIO: DefaultTagValues.SCENARIO,
            BaseTags.FLEET: DefaultTagValues.FLEET,
            BaseTags.VERSION: DefaultTagValues.VERSION,
        }
    
    @staticmethod
    def custom_preset(**kwargs) -> Dict[str, str]:
        preset = TagPreset.basic_preset()
        preset.update({k: str(v) for k, v in kwargs.items()})
        print('preset:',preset)
        return preset        
        
class EnhancedInfluxDBManager(InfluxDBManager):
    
    def write_points(self, points: List[InfluxDataPoint], 
                    database: Optional[str] = None) -> bool:
        """Запись списка точек данных"""
        try:
            db_name = database or self.config.db_name
            client = InfluxDBClient(
                host=self.config.ip,
                port=self.config.port,
                database=db_name
            )
            
            influx_data = [point.to_influx_format() for point in points]
            success = client.write_points(influx_data)
            return success
            
        except Exception as e:
            print(f"Ошибка при записи точек: {e}")
            return False
        #finally:
        #    client.close()
    
    def write_dataframe_enhanced(self, dataframe: pd.DataFrame, 
                               measurement: str,
                               tag_columns: List[str] = None,
                               field_columns: List[str] = None,
                               timestamp_column: str = None,
                               additional_tags: Dict[str, str] = None,
                               database: Optional[str] = None) -> bool:
        """
        Улучшенная запись DataFrame с автоматическим определением структуры
        
        Args:
            dataframe: DataFrame с данными
            measurement: Имя измерения
            tag_columns: Столбцы, которые следует использовать как теги
            field_columns: Столбцы, которые следует использовать как поля
            timestamp_column: Столбец с временными метками
            database: Имя базы данных
        """
        if additional_tags is not None:
            for add_tag in additional_tags.keys():
                #print('add_tag:',add_tag,'=',additional_tags[add_tag])
                dataframe[add_tag]=additional_tags[add_tag]
        
        if field_columns is None:
            # Автоматическое определение числовых колонок как полей
            field_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
            
        if tag_columns is None:
            tag_columns = list(set(dataframe.keys())-set(field_columns)-set(['TimeWrite2DB']))
        
        points = []
        
        for idx, row in dataframe.iterrows():
            # Определение временной метки
            if timestamp_column and timestamp_column in dataframe.columns:
                timestamp = row[timestamp_column]
            elif hasattr(dataframe.index, 'to_pydatetime'):
                timestamp = idx.to_pydatetime()
            else:
                timestamp = datetime.utcnow()
            
            # Сбор тегов
            tags = {}
            for tag_col in tag_columns:
                if tag_col in dataframe.columns:
                    tags[tag_col] = str(row[tag_col])
            #print('tags:',tags)
            
            # Сбор полей
            fields = {}
            for field_col in field_columns:
                if field_col in dataframe.columns:
                    fields[field_col] = float(row[field_col])
            
            points.append(InfluxDataPoint(
                measurement=measurement,
                fields=fields,
                tags=tags,
                timestamp=timestamp
            ))
        #print(f'{points}')
        
        return self.write_points(points, database)
    
    def write_with_preset(self, dataframe: pd.DataFrame, 
                         measurement: str,
                         preset_name: str = "custom",
                         database: Optional[str] = None,
                         **preset_kwargs) -> bool:
        """
        Запись с использованием предустановленных конфигураций тегов
        """
        tag_presets = {
            "basic": TagPreset.basic_preset(),
            "custom": TagPreset.custom_preset(**preset_kwargs)
        }
        tags = tag_presets.get(preset_name, TagPreset.basic_preset())
        field_columns=[col for col in dataframe.columns 
                         if col not in tags and pd.api.types.is_numeric_dtype(dataframe[col])]
        #print('field_columns: ',field_columns)
        # Добавление 
        
        return self.write_dataframe_enhanced(
            dataframe=dataframe,
            database=database,
            measurement=measurement,
            additional_tags=tags,
            field_columns=field_columns
        )
    def read_data(self, 
                 measurement: str,                  #1
                 start_time: Optional[str] = None,  #2
                 end_time: Optional[str] = None,    #3
                 fields: Optional[List[str]] = None,#4
                 tags: Optional[Dict[str, str]] = None,#5
                 database: Optional[str] = None,       #6 
                 time_zone: str = 'Etc/GMT-3',         #7
                 batch_size: str = '1d',               #8
                 max_retries: int = 3,                 #9 
                 delay_between_batches: float = 0.1) -> pd.DataFrame:
        """
        Чтение данных из InfluxDB с фильтрацией
        
        Args:
            measurement: Имя измерения
            start_time: Начальное время (можно строку или datetime)
            end_time: Конечное время (опционально)
            fields: Список полей для выборки
            tags: Фильтры по тегам
            database: Имя базы данных
            time_zone: Часовой пояс
            batch_size: Размер батча ('1d', '6h', '1h' и т.д.)
            max_retries: Максимальное количество повторных попыток
            delay_between_batches: Задержка между батчами в секундах          
        Returns:
            pd.DataFrame: Данные из InfluxDB
        """
        if start_time is None:
            print('start_time: None')
        else:
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time) if end_time else pd.Timestamp(start_time)
        
        db_name = database or self.config.db_name
        self.connect(db_name)
        return self._read_single_batch(measurement, start_time, end_time, fields, tags, database, time_zone)    

    def read_data_batch(self, 
                       measurement: str,                    #1
                       start_time: Optional[str] = None,    #2
                       end_time: Optional[str] = None,      #3
                       fields: Optional[List[str]] = None,  #4
                       tags: Optional[Dict[str, str]] = None,#5
                       database: Optional[str] = None,       #6
                       time_zone: str = 'Etc/GMT-3',         #7
                       batch_size: str = '365d',              #8
                       max_retries: int = 3,                 #9
                       delay_between_batches: float = 0.05) -> pd.DataFrame:
        """
        Чтение данных из InfluxDB с пакетной обработкой
        
        Args:
            measurement: Имя измерения
            start_time: Начальное время (можно строку или datetime)
            end_time: Конечное время (опционально)
            fields: Список полей для выборки
            tags: Фильтры по тегам
            database: Имя базы данных
            time_zone: Часовой пояс
            batch_size: Размер батча ('1d', '6h', '1h' и т.д.)
            max_retries: Максимальное количество повторных попыток
            delay_between_batches: Задержка между батчами в секундах
            
        Returns:
            pd.DataFrame: Объединенные данные из всех батчей
        """
        if start_time is None:
            print('start_time: None')
        else:
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time) if end_time else pd.Timestamp(start_time)
        
        print('read_data_batch!!!!!',measurement,start_time,end_time,fields,tags)
        # Обработка временных параметров
        if start_time is None:
            print('start_time: None')
            return self._read_single_batch(measurement, None, None, fields, tags, database, time_zone)
        
        start_dt = pd.Timestamp(start_time)
        end_dt = pd.Timestamp(end_time) if end_time else pd.Timestamp.now()
        
        # Параметры для единичного запроса
        if start_dt == end_dt:
            return self._read_single_batch(measurement, start_time, end_time, fields, tags, database, time_zone)
        
        # Генерация временных интервалов для батчей
        time_ranges = self._generate_time_ranges(start_dt, end_dt, batch_size)
        
        print(f"Выполняется пакетное чтение: {len(time_ranges)} батчей")
        
        all_data = []
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            with tqdm(total=len(time_ranges), desc=f"Чтение {measurement}") as pbar:
                for batch_start, batch_end in time_ranges:
                    #print('In batch steps...','+'*50)
                    batch_data = self._read_single_batch(measurement, batch_start, batch_end, fields, tags, database, time_zone)
                    #print('In batch steps...','+-'*50)
                    #print(batch_data)
                    #batch_data = self._read_batch_with_retry( measurement, batch_start, batch_end, fields, tags,db_name, time_zone, max_retries)
                    
                    if not batch_data.empty:
                        all_data.append(batch_data)
                        pbar.set_postfix({
                            'batch': f"{batch_start.strftime('%Y-%m-%d %H:%M')}",
                            'records': len(batch_data)
                        })
                    
                    pbar.update(1)
                    #print('In batch steps...','...'*50)
                    # Задержка между запросами
                    if delay_between_batches > 0:
                        time.sleep(delay_between_batches)
            
            # Объединение всех данных
            if all_data:
                #result_df = pd.concat(all_data, ignore_index=True)
                result_df = pd.concat(all_data)
                print(f"Успешно прочитано {len(result_df)} записей за {len(time_ranges)} батчей")
                return result_df
            else:
                print("Нет данных для указанных параметров")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Ошибка при пакетном чтении из InfluxDB: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()

    def _generate_time_ranges(self, start_dt: datetime, end_dt: datetime, 
                             batch_size: str) -> List[tuple]:
        """Генерация временных интервалов для батчей"""
        
        # Парсинг размера батча
        size_value = int(batch_size[:-1])
        size_unit = batch_size[-1]
        
        # Создание временных интервалов
        time_ranges = []
        current_start = start_dt
        
        while current_start < end_dt:
            if size_unit == 'd':  # дни
                current_end = current_start + timedelta(days=size_value)
            elif size_unit == 'h':  # часы
                current_end = current_start + timedelta(hours=size_value)
            elif size_unit == 'm':  # минуты
                current_end = current_start + timedelta(minutes=size_value)
            else:
                raise ValueError(f"Неизвестная единица измерения: {size_unit}")
            
            # Убедимся, что не выходим за конечное время
            current_end = min(current_end, end_dt)
            time_ranges.append((current_start, current_end))
            
            current_start = current_end
        
        return time_ranges        
        
            
    def _read_single_batch(self, measurement: str, 
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      fields: Optional[List[str]] = None,
                      tags: Optional[Dict[str, str]] = None,
                      database: Optional[str] = None,
                      time_zone: str = 'Etc/GMT-3') -> pd.DataFrame:
        """
        Чтение одного батча данных (оригинальная логика)
        """
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            # Формирование условия для тегов
            tags_condition = ''
            if tags:
                tags_conditions = [f"{k}='{str(v)}'" for k, v in tags.items()]
                tags_condition = ' AND '.join(tags_conditions) + ' AND '
            
            # Формирование списка полей
            fields_select = '*' if not fields else '"'+'", "'.join(fields)+'"'
            print('_read_single_batch'+'_'*50)
            if start_time is None:
                if len(tags_condition) == 0:
                    query = f"SELECT {fields_select} FROM {measurement}"
                else:
                    query = f"SELECT {fields_select} FROM {measurement} WHERE {tags_condition}"
            #
            else:
                query = f"SELECT {fields_select} FROM {measurement} WHERE {tags_condition}time >= '{start_time}' AND time <= '{end_time}'"
                if time_zone:
                    query += f" tz('{time_zone}')"
                    
            print(f"БД:{db_name}. Выполняем запрос: {query}")
            result = self.client.query(query)
            
            if measurement in result:
                df = result[measurement]
                if time_zone and not df.empty:
                    df = df.tz_convert(time_zone)
                #df=df.set_index('time')
                return df
            else:
                print('Результат запроса - пустая таблица')
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Ошибка при чтении батча: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()       
            
            
    def read_data_last_point(self, measurement: str, 
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 fields: Optional[List[str]] = None,
                 tags: Optional[Dict[str, str]] = None,
                 database: Optional[str] = None,
                 time_zone: str = 'Etc/GMT-3') -> pd.DataFrame:
        """
        Чтение данных из InfluxDB с фильтрацией
        
        Args:
            measurement: Имя измерения
            start_time: Начальное время (можно строку или datetime)
            end_time: Конечное время (опционально)
            fields: Список полей для выборки
            tags: Фильтры по тегам
            database: Имя базы данных
            time_zone: Часовой пояс
            
        Returns:
            pd.DataFrame: Данные из InfluxDB
        """
        if start_time is None:
            print('start_time: None')
        else:
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time) if end_time else pd.Timestamp(start_time)
        
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            # Формирование условия для тегов
            tags_condition = ''
            if tags:
                tags_conditions = [f"{k}='{str(v)}'" for k, v in tags.items()]
                tags_condition = ' AND '.join(tags_conditions) + ' AND '
            
            # Формирование списка полей
            fields_select = '*' if not fields else '"'+'", "'.join(fields)+'"'
            
            if start_time is None:
                query = f"""
                    SELECT {fields_select} FROM {measurement}  
                    WHERE {tags_condition} ORDER BY time DESC  LIMIT 1
                """            
            else:
                query = f"""
                    SELECT {fields_select} FROM {measurement} 
                    WHERE {tags_condition}time >= '{start_time}' AND time <= '{end_time}' ORDER BY time DESC  LIMIT 1
                """
            
            if time_zone:
                query += f" tz('{time_zone}')"
            
            print(f"БД:{db_name}. Выполняем запрос: {query}")
            result = self.client.query(query)
            
            if measurement in result:
                df = result[measurement]
                if time_zone and not df.empty:
                    df = df.tz_convert(time_zone)
                return df
            else:
                print('Результат запроса - пустая таблица')
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Ошибка при чтении из InfluxDB: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()
    
    def read_last_point1(self, measurement: str,
                       tags: Optional[Dict[str, str]] = None,
                       database: Optional[str] = None) -> pd.DataFrame:
        """
        Чтение последней точки данных
        
        Args:
            measurement: Имя измерения
            tags: Фильтры по тегам
            database: Имя базы данных
            
        Returns:
            pd.DataFrame: Последняя точка данных
        """
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            tags_condition = ''
            if tags:
                tags_conditions = [f"{k}='{str(v)}'" for k, v in tags.items()]
                tags_condition = ' AND '.join(tags_conditions) + ' AND '
            
            query = f"""
                SELECT * FROM {measurement} 
                WHERE {tags_condition}time > now() - 1d
                ORDER BY time DESC 
                LIMIT 1
            """
            
            result = self.client.query(query)
            
            if measurement in result:
                return result[measurement]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Ошибка при чтении последней точки: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()
    
    def read_aggregated_data(self, measurement: str,
                           start_time: str,
                           end_time: str,
                           aggregation: str = 'mean',
                           window: str = '1h',
                           fields: Optional[List[str]] = None,
                           tags: Optional[Dict[str, str]] = None,
                           database: Optional[str] = None) -> pd.DataFrame:
        """
        Чтение агрегированных данных
        
        Args:
            measurement: Имя измерения
            start_time: Начальное время
            end_time: Конечное время
            aggregation: Тип агрегации (mean, sum, count, max, min)
            window: Окно агрегации (1h, 30m, 1d)
            fields: Поля для агрегации
            tags: Фильтры по тегам
            database: Имя базы данных
            
        Returns:
            pd.DataFrame: Агрегированные данные
        """
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            tags_condition = ''
            if tags:
                tags_conditions = [f"{k}='{str(v)}'" for k, v in tags.items()]
                tags_condition = ' AND '.join(tags_conditions) + ' AND '
            
            fields_select = '*' if not fields else ', '.join(fields)
            
            query = f"""
                SELECT {aggregation}({fields_select}) 
                FROM {measurement} 
                WHERE {tags_condition}time >= '{start_time}' AND time <= '{end_time}'
                GROUP BY time({window})
            """
            
            #print(f"Выполняем агрегирующий запрос: {query}")
            result = self.client.query(query)
            
            if measurement in result:
                return result[measurement]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Ошибка при чтении агрегированных данных: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()
    def get_measurement_info(self, measurement: str,
                           database: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение информации об измерении
        
        Args:
            measurement: Имя измерения
            database: Имя базы данных
            
        Returns:
            Dict: Информация об измерении
        """
        db_name = database or self.config.db_name
        self.connect(db_name)
        
        try:
            # Получение временного диапазона
            time_range_query = f"""
                SELECT FIRST(*), LAST(*) FROM {measurement}
            """
            
            # Получение уникальных тегов
            tag_keys_query = f"""
                SHOW TAG KEYS FROM {measurement}
            """
            
            # Получение уникальных полей
            field_keys_query = f"""
                SHOW FIELD KEYS FROM {measurement}
            """
            
            result = {
                'measurement': measurement,
                'time_range': self.client.query(time_range_query),
                'tag_keys': self.client.query(tag_keys_query),
                'field_keys': self.client.query(field_keys_query)
            }
            
            return result
            
        except Exception as e:
            print(f"Ошибка при получении информации об измерении: {e}")
            return {}
        finally:
            self.disconnect()
    
    def get_measurements_list(self, database: Optional[str] = None) -> List[str]:
        """
        Получение списка всех измерений в базе данных
        
        Args:
            database: Имя базы данных
            
        Returns:
            List[str]: Список измерений
        """
        db_name = database or self.config.db_name
        
        try:
            client = InfluxDBClient(
                host=self.config.ip,
                port=self.config.port,
                database=db_name
            )
            
            result = client.query("SHOW MEASUREMENTS")
            measurements = [list(point.values())[0] for point in result.get_points()]
            
            return measurements
            
        except Exception as e:
            print(f"Ошибка при получении списка измерений: {e}")
            return []
        finally:
            client.close()           

class InfluxDataFactory:
    
    @staticmethod
    def from_dataframe(df: pd.DataFrame, measurement: str, 
                      tag_columns: List[str] = None) -> List[InfluxDataPoint]:
        """Создание точек данных из DataFrame"""
        points = []
        
        for idx, row in df.iterrows():
            # Автоматическое определение структуры
            fields = {}
            tags = {}
            
            for col, value in row.items():
                if tag_columns and col in tag_columns:
                    tags[col] = str(value)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    fields[col] = float(value)
            
            points.append(InfluxDataPoint(
                measurement=measurement,
                fields=fields,
                tags=tags,
                timestamp=df.index[idx] if hasattr(df.index, 'to_pydatetime') else None
            ))
        
        return points
    
    @staticmethod
    def from_dict_list(data_list: List[Dict], measurement: str,
                      field_keys: List[str], tag_keys: List[str] = None) -> List[InfluxDataPoint]:
        """Создание точек данных из списка словарей"""
        points = []
        
        for data in data_list:
            fields = {k: float(data[k]) for k in field_keys if k in data}
            tags = {k: str(data[k]) for k in (tag_keys or []) if k in data}
            
            points.append(InfluxDataPoint(
                measurement=measurement,
                fields=fields,
                tags=tags,
                timestamp=data.get('timestamp')
            ))
        
        return points            
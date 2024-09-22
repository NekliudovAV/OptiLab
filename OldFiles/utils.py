def write_pyomo_results_2_influxDB(resdf, calc_type='Optimize', database_='TES',  time_zone_ = None, tags_=None, host='127.0.0.1'):
    # tags_={'Scenarii':'MinD0'}
    influxDataFrameClient_client = DataFrameClient(host=host, port=8086, database=database_)
    influx_DBname = calc_type
    influxDataFrameClient_client.write_points(resdf.astype(float), influx_DBname, tags=tags_, batch_size=1000)
    influxDataFrameClient_client.close()

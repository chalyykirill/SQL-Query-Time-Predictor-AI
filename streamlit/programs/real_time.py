import cx_Oracle
import time
import pandas as pd

def create_connection():
    dsn = cx_Oracle.makedsn('localhost', 1521, service_name='xe')  # 'xe' - это пример названия сервиса, измените его на ваше
    connection = cx_Oracle.connect('c##avia', 'qwerty', dsn)
    return connection

def execute_query_with_timing(query):
    connection = create_connection()
    cursor = connection.cursor()

    # Записываем время начала выполнения запроса
    start_time = time.time()

    # Выполняем запрос
    cursor.execute(query)

    # Завершаем выполнение запроса
    cursor.fetchall()

    # Записываем время окончания выполнения запроса
    end_time = time.time()

    # Вычисляем время выполнения
    elapsed_time = end_time - start_time

    # Закрываем соединение
    cursor.close()
    connection.close()

    data_real = pd.DataFrame({'SQL_TEXT': [query], 'Real_ELAPSED_TIME': [elapsed_time]})
    # Возвращаем время выполнения
    return data_real

# Пример использования
#query = "SELECT ticket_no, MIN(seat_no) FROM boarding_passes GROUP BY ticket_no"
#data_real = execute_query_with_timing(query)

# Вывод времени выполнения
#print(data_real)
import random

columns = {
    'bookings': ['book_ref', 'book_date', 'total_amount'],
    'tickets': ['ticket_no', 'book_ref', 'passenger_id', 'passenger_name', 'contact_data'],
    'flights': ['flight_id', 'flight_no', 'scheduled_departure', 'scheduled_arrival', 'departure_airport', 'arrival_airport', 'status', 'aircraft_code', 'actual_departure', 'actual_arrival'],
    'ticket_flights': ['ticket_no', 'flight_id', 'fare_conditions', 'amount'],
    'airports_data': ['airport_code', 'airport_name', 'city', 'coordinates', 'timezone'],
    'boarding_passes': ['ticket_no', 'flight_id', 'boarding_no', 'seat_no'],
    'aircrafts_data': ['aircraft_code', 'model', 'range'],
    'seats': ['aircraft_code', 'seat_no', 'fare_conditions']
}

# Словарь с данными о типах столбцов для каждой таблицы
column_types = {
    'tickets': {
        'ticket_no': 'varchar2',
        'book_ref': 'varchar2',
        'passenger_id': 'varchar2',
        'passenger_name': 'varchar2',
        'contact_data': 'varchar2'
    },
    'boarding_passes': {
        'ticket_no': 'varchar2',
        'flight_id': 'number',
        'boarding_no': 'number',
        'seat_no': 'varchar2'
    },
    'bookings': {
        'book_ref': 'varchar2',
        'book_date': 'timestamp',
        'total_amount': 'number'
    },
    'flights': {
        'flight_id': 'number',
        'flight_no': 'varchar2',
        'scheduled_departure': 'timestamp',
        'scheduled_arrival': 'timestamp',
        'departure_airport': 'varchar2',
        'arrival_airport': 'varchar2',
        'status': 'varchar2',
        'aircraft_code': 'varchar2',
        'actual_departure': 'timestamp',
        'actual_arrival': 'timestamp'
    },
    'ticket_flights': {
        'ticket_no': 'varchar2',
        'flight_id': 'number',
        'fare_conditions': 'varchar2',
        'amount': 'number'
    },
    'airports_data': {
        'airport_code': 'varchar2',
        'airport_name': 'varchar2',
        'city': 'varchar2',
        'coordinates': 'varchar2',
        'timezone': 'varchar2'
    },
    'aircrafts_data': {
        'aircraft_code': 'varchar2',
        'model': 'varchar2',
        'range': 'number'
    },
    'seats': {
        'aircraft_code': 'varchar2',
        'seat_no': 'varchar2',
        'fare_conditions': 'varchar2'
    }
}


def random_table():
    return random.choice(list(columns.keys()))


def random_column(table_name, exclude_columns=[], data_type=None):
    # Начинаем с полного списка столбцов для данной таблицы
    if data_type:
        available_columns = [col for col, dtype in column_types[table_name].items() if dtype == data_type]
    else:
        available_columns = list(columns[table_name])

    # Далее исключаем столбцы из списка, если они есть в exclude_columns
    filtered_columns = [col for col in available_columns if col not in exclude_columns]

    # Возвращаем None, если отфильтрованный список пуст
    if not filtered_columns:
        return None

    # Выбираем случайный столбец из отфильтрованного списка
    return random.choice(filtered_columns)


def random_condition(column):
    if column in ['book_date', 'scheduled_departure', 'scheduled_arrival']:
        dates = ['2017-08-01', '2017-07-01', '2017-06-01']
        return f"{column} > DATE '{random.choice(dates)}'"
    elif column in ['total_amount']:
        return f"{column} > {random.randint(3400, 1204500)}"
    elif column in ['flight_id']:
        return f"{column} < {random.randint(1, 33120)}"
    elif column in ['boarding_no']:
        return f"{column} > {random.randint(1, 374)}"
    elif column in ['amount']:
        return f"{column} < {random.randint(3000, 203300)}"
    elif column in ['range']:
        return f"{column} > {random.randint(1200, 11100)}"
    else:
        return f"1=1"


def generate_select_query():
    table = random_table()
    col1 = random_column(table)
    col2 = random_column(table)
    condition = random_condition(col1)
    query = f"SELECT {col1}, {col2} FROM {table} WHERE {condition}"
    return query


#print(generate_select_query())

def random_order_by(table):
    # Выбираем случайный столбец из доступных в таблице
    column = random.choice(columns[table])
    # Случайно выбираем направление сортировки
    order = random.choice(['ASC', 'DESC'])
    # Формируем часть запроса ORDER BY
    order_by_clause = f"ORDER BY {column} {order}"
    return order_by_clause

def generate_query_with_order_by():
    table = random_table()
    # Генерируем базовый запрос SELECT для всех столбцов таблицы
    query = f"SELECT * FROM {table} "
    # Добавляем ORDER BY часть
    order_by_clause = random_order_by(table)
    # Комбинируем запрос с ORDER BY
    full_query = query + order_by_clause
    return full_query

# Пример вызова функции
#print(generate_query_with_order_by())

def generate_aggregate_query():
    table = random_table()
    # Выбираем случайный столбец для агрегации
    agg_col = random_column(table)
    # Выбираем другой случайный столбец для группировки
    group_by_col = random_column(table)
    while agg_col == group_by_col:
        group_by_col = random_column(table)  # Убедимся, что столбцы разные

    if column_types[table][agg_col] == 'number':
        agg_func = random.choice(['COUNT', 'AVG', 'SUM', 'MAX', 'MIN'])
    else:
        agg_func = 'COUNT'
    # Создаем запрос с агрегатной функцией и группировкой
    query = f"SELECT {group_by_col}, {agg_func}({agg_col}) FROM {table} GROUP BY {group_by_col}"
    return query

#print(generate_aggregate_query())

def generate_aggregate_query_having():
    table = random_table()
    # Выбираем случайный столбец для агрегации
    agg_col = random_column(table)
    # Выбираем другой случайный столбец для группировки
    group_by_col = random_column(table)
    while agg_col == group_by_col:
        group_by_col = random_column(table)  # Убедимся, что столбцы разные

    if column_types[table][agg_col] == 'number':
        agg_func = random.choice(['COUNT', 'AVG', 'SUM', 'MAX', 'MIN'])
    else:
        agg_func = 'COUNT'
    # Создаем запрос с агрегатной функцией и группировкой
    query = f"SELECT {group_by_col}, {agg_func}({agg_col}) FROM {table} GROUP BY {group_by_col} HAVING {agg_func}({agg_col}) > {random.randint(1, 500)}"
    return query

#print(generate_aggregate_query_having())

def random_column_agg(table_name, data_type=None):
    if data_type:
        filtered_columns = [col for col, dtype in column_types[table_name].items() if dtype == data_type]
        if filtered_columns:
            return random.choice(filtered_columns)
    return random.choice(list(columns[table_name]))

def generate_aggregate_query_advanced():
    table = random_table()
    # Выбираем случайный столбец для агрегации, предпочтительно числовой
    agg_col = random_column_agg(table, data_type='number')
    # Выбираем другие столбцы для группировки, могут быть любого типа
    group_by_cols = random.sample(columns[table], random.randint(1, len(columns[table]) - 1))
    group_by_cols = [col for col in group_by_cols if col != agg_col]

    # Выбор агрегатной функции, соответствующей типу данных столбца агрегации
    if column_types[table][agg_col] == 'number':
        agg_func = random.choice(['COUNT', 'AVG', 'SUM', 'MAX', 'MIN'])
    else:
        agg_func = 'COUNT'

    # Формируем часть запроса с условием WHERE, если нужно
    where_condition = random_condition(random_column(table))

    group_by_clause = ', '.join(group_by_cols)
    query = f"SELECT {group_by_clause}, {agg_func}({agg_col}) FROM {table} WHERE {where_condition} GROUP BY {group_by_clause}"
    return query

# Пример вызова функции
#print(generate_aggregate_query_advanced())

def compatible_columns(table1, table2):
    cols1 = set(columns[table1])
    cols2 = set(columns[table2])
    compatible = list(cols1.intersection(cols2))
    return [(col, col) for col in compatible]

def generate_join_query():
    max_attempts = 10  # Максимальное количество попыток найти подходящие столбцы
    for attempt in range(max_attempts):
        table1 = random_table()
        table2 = random_table()
        while table1 == table2:
            table2 = random_table()

        compatible_cols = compatible_columns(table1, table2)
        if compatible_cols:
            join_col1, join_col2 = random.choice(compatible_cols)
            select_col1 = random_column(table1, exclude_columns=[join_col1])
            select_col2 = random_column(table2, exclude_columns=[join_col2])
            query = f"SELECT t1.{select_col1}, t2.{select_col2} FROM {table1} t1 JOIN {table2} t2 ON t1.{join_col1} = t2.{join_col2}"
            return query
    return "Failed to find compatible columns for JOIN after several attempts"

# Пример вызова функции
#print(generate_join_query())

def random_function(column, table):
    # Получаем тип данных для столбца
    column_type = column_types[table][column]
    # Определение функций на основе типа данных
    if column_type == 'number':
        functions = [f"ROUND({column})"]
    elif column_type in ['varchar2', 'varchar']:
        functions = [f"LOWER({column})", f"UPPER({column})"]
    else:
        functions = []
    return random.choice(functions) if functions else None

def generate_function_query():
    table = random_table()
    column = random_column(table)
    func = random_function(column, table)
    if func:
        query = f"SELECT {func} FROM {table}"
        return query
    else:
        return "No applicable function for the selected column type"

# Пример вызова функции
#print(generate_function_query())

def generate_advanced_join_query():
    max_attempts = 10  # Максимальное количество попыток найти подходящие столбцы
    for attempt in range(max_attempts):
        table1 = random_table()
        table2 = random_table()
        while table1 == table2:
            table2 = random_table()

        compatible_cols = compatible_columns(table1, table2)
        if compatible_cols:
            join_col1, join_col2 = random.choice(compatible_cols)
            join_type = random.choice(['JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL OUTER JOIN'])

            # Выбор случайного количества столбцов для вывода в результате, исключая столбцы join
            select_cols1 = [col for col in columns[table1] if col != join_col1]
            select_cols2 = [col for col in columns[table2] if col != join_col2]
            random.shuffle(select_cols1)
            random.shuffle(select_cols2)
            select_cols1 = select_cols1[:random.randint(1, len(select_cols1))]  # Выбор 1+ столбца
            select_cols2 = select_cols2[:random.randint(1, len(select_cols2))]  # Выбор 1+ столбца

            select_part = ', '.join([f"t1.{col}" for col in select_cols1] + [f"t2.{col}" for col in select_cols2])
            query = f"SELECT {select_part} FROM {table1} t1 {join_type} {table2} t2 ON t1.{join_col1} = t2.{join_col2}"
            return query
    return "Failed to find compatible columns for JOIN after several attempts"


# Пример вызова функции
#print(generate_advanced_join_query())

# Генератор случайных SQL запросов
def generate_random_query():
    functions = [generate_select_query, generate_join_query, generate_aggregate_query, generate_function_query, generate_advanced_join_query, generate_aggregate_query_advanced, generate_query_with_order_by, generate_aggregate_query_having]
    return random.choice(functions)()

#print(generate_random_query())
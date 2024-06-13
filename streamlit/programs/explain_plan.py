import cx_Oracle
import pandas as pd
import re

class OracleExplainPlan:
    def __init__(self, user, password, host, port, service_name):
        self.dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        self.user = user
        self.password = password

    def create_connection(self):
        connection = cx_Oracle.connect(self.user, self.password, self.dsn)
        return connection

    def explain_plan(self, query):
        connection = self.create_connection()
        cursor = connection.cursor()

        # Выполняем EXPLAIN PLAN
        cursor.execute(f"EXPLAIN PLAN FOR {query}")

        # Получаем план выполнения
        cursor.execute("SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)")
        plan = cursor.fetchall()

        # Закрываем соединение
        cursor.close()
        connection.close()

        return plan

    def extract_execution_time(self, plan):
        time_pattern = re.compile(r"\|\s+(\d{2}:\d{2}:\d{2})\s+\|")
        execution_times = []

        for row in plan:
            match = time_pattern.search(row[0])
            if match:
                execution_times.append(match.group(1))

        return execution_times

    def time_to_seconds(self, time_str):
        h, m, s = map(int, time_str.split(':'))
        total_seconds = h * 3600 + m * 60 + s
        return total_seconds

    def get_execution_time(self, query):
        plan = self.explain_plan(query)
        execution_times = self.extract_execution_time(plan)
        if execution_times:
            return self.time_to_seconds(execution_times[0])
        else:
            return None

    def create_data_plan(self, query):
        execution_time = self.get_execution_time(query)
        data_plan = pd.DataFrame({'SQL_TEXT': [query], 'Plan_ELAPSED_TIME': [execution_time]})
        return data_plan

# Пример использования
#oracle_explain_plan = OracleExplainPlan(user='c##avia', password='qwerty', host='localhost', port=1521, service_name='xe')
#query = "SELECT * FROM boarding_passes ORDER BY boarding_no ASC"
#data_plan = oracle_explain_plan.create_data_plan(query)
#print(data_plan)
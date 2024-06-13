import streamlit as st
import importlib.util
import pandas as pd
import numpy as np
from programs.ml_predict import Net
#from pathlib import Path


# Функция для загрузки модуля
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Загрузка программ
generator_query = load_module("generator_query", r"C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\programs\generator_query.py")
ml_predict = load_module("ml_predict", r"C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\programs\ml_predict.py")
explain_plan = load_module("explain_plan", r"C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\programs\explain_plan.py")
real_time = load_module("real_time", r"C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\programs\real_time.py")

def main():
    st.title('Прогнозирование времени выполнения SQL запроса')

    st.header('Описание базы данных')
    st.markdown('[Перейти на сайт описания базы данных](https://www.postgrespro.ru/education/demodb)')
    image_path = r'C:\Users\Zver\magistr_study\it_academy_samsung\main_project\streamlit_programm\images\chema_avia.png'
    st.image(image_path, caption='Схема базы данных "авиа"', use_column_width=True)

    # Заголовок приложения
    st.header('Генератор случайных запросов')
    # Кнопка
    if st.button('Получить запрос'):
        st.write(generator_query.generate_random_query())

    st.header('Прогноз на основе нейросети')
    query_1 = st.text_input('Введите SQL запрос', key='query1')
    if query_1:
        data_ml = ml_predict.process_sql_query(query_1)
        st.write('Результат:')
        st.dataframe(data_ml)

    st.header('Прогноз на основе EXPLAIN PLAN')
    query_2 = st.text_input('Введите SQL запрос', key='query2')
    if query_2:
        data_plan = explain_plan.OracleExplainPlan(user='c##avia', password='qwerty', host='localhost', port=1521, service_name='xe').create_data_plan(query_2)
        st.write('Результат:')
        st.dataframe(data_plan)

    st.header('Реальное время выполнения запроса')
    query_3 = st.text_input('Введите SQL запрос', key='query3')
    if query_3:
        data_real = real_time.execute_query_with_timing(query_3)
        st.write('Результат:')
        st.dataframe(data_real)

    st.header('Оценка прогноза')
    query_4 = st.text_input('Введите SQL запрос', key='query4')
    if query_4:
        ml_data = ml_predict.process_sql_query(query_4)
        plan_data = explain_plan.OracleExplainPlan(user='c##avia', password='qwerty', host='localhost', port=1521, service_name='xe').create_data_plan(query_4)
        real_data = real_time.execute_query_with_timing(query_4)
        ml = ml_data['Predicted_ELAPSED_TIME'][0]
        plan = plan_data['Plan_ELAPSED_TIME'][0]
        real = real_data['Real_ELAPSED_TIME'][0]

        if real != 0:
            y_true = np.array([real])

            # Предсказанные значения (например, от разных моделей)
            predictions = {
                'ML': np.array([ml]),
                'EXPLAIN PLAN': np.array([plan])
            }

            # Функции для вычисления метрик
            def absolute_error(y_true, y_pred):
                return abs(y_true - y_pred)

            def squared_error(y_true, y_pred):
                return (y_true - y_pred) ** 2

            def relative_error(y_true, y_pred):
                return (y_true - y_pred) / y_true

            def percentage_error(y_true, y_pred):
                return relative_error(y_true, y_pred) * 100

            # Список метрик и их названий
            metrics = {
                'Абсолютная ошибка': absolute_error,
                'Квадратичная ошибка': squared_error,
                'Относительная ошибка': relative_error,
                'Процентная ошибка': percentage_error
            }

            # Создаем пустой DataFrame
            df = pd.DataFrame(index=metrics.keys(), columns=predictions.keys())

            # Заполняем DataFrame вычисленными метриками
            for model_name, y_pred in predictions.items():
                for metric_name, metric_func in metrics.items():
                    metric_values = metric_func(y_true, y_pred)
                    df.at[metric_name, model_name] = np.mean(metric_values)

            st.write('Результат:')
            st.write(query_4)
            st.write('Predict_ELAPSED_TIME:', ml, 'c')
            st.write('Plan_ELAPSED_TIME:', plan, 'c')
            st.write('Real_ELAPSED_TIME:', real, 'c')
            st.dataframe(df)

            st.write('Абсолютная ошибка')
            st.markdown("$$ |y_{true} - y_{pred}| $$")

            st.write('Квадратичная ошибка')
            st.markdown("$$ (y_{true} - y_{pred})^2 $$")

            st.write('Относительная ошибка')
            st.markdown("$$ \\frac{y_{true} - y_{pred}}{y_{true}} $$")

            st.write('Процентная ошибка')
            st.markdown("$$ \\frac{y_{true} - y_{pred}}{y_{true}} \\times 100 $$")
        else:
            st.write('Пустой запрос')

if __name__ == "__main__":
    main()

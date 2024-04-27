# SQL-Query-Time-Predictor-AI
This is a repository dedicated to a project to predict the execution time of SQL queries in a database using artificial intelligence (AI)
## Creating a dataset (see the Creating_dataset file):
1) Creating an EXTENDED_SQL_MONITORING table in the DBMS based on the V$SQL service table and the Scheduler task to collect statistics on the execution of SQL queries in the DBMS
1) Creating a random sql query generator for the database
2) Connecting to the database
3) Running the query generator in the DBMS
4) Collecting statistics from the EXTENDED_SQL_MONITORING table for further work with it

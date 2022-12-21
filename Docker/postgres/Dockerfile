FROM postgres:13

COPY create-multiple-postgresql-databases.sh /docker-entrypoint-initdb.d/

RUN sed -i -e 's/\r$//' /docker-entrypoint-initdb.d/create-multiple-postgresql-databases.sh
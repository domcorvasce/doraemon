FROM python:3.12

WORKDIR /code

RUN apt update && apt install -y build-essential
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:$PATH
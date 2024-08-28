FROM python:3.10

ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV GOOGLE_API_KEY=""

WORKDIR /var/www/chainlit

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/*.py ./

# chainlit config書き換え
COPY ./config/config.py /usr/local/lib/python3.10/site-packages/chainlit/config.py

RUN mkdir data-pdf

EXPOSE 8000

CMD ["chainlit","run","app.py","-w","--host","0.0.0.0"]

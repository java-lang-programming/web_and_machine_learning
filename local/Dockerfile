FROM python:3.7.0-slim
ENV PYTHONUNBUFFERED 1
RUN mkdir app
WORKDIR /app
ADD requirements.txt /app/
ADD jupyter_notebook_config.py /root/.jupyter/
ADD jupyter_notebook_config.json /root/.jupyter/
RUN pip install -r requirements.txt
ADD ./notebook/ /app/
EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root"]